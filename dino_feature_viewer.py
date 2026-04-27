#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


def load_frames(rgb_dir: Path, max_frames: Optional[int] = None) -> np.ndarray:
    frame_paths = sorted(rgb_dir.glob("frame_*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {rgb_dir}")

    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]

    frames = []
    for p in frame_paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed reading frame: {p}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)

    return np.stack(frames, axis=0)


def discover_cameras(dataset_root: Path, requested: Optional[list[str]]) -> list[str]:
    rgb_root = dataset_root / "rgb"
    if not rgb_root.exists():
        raise FileNotFoundError(f"Missing RGB root: {rgb_root}")

    available = sorted([p.name for p in rgb_root.iterdir() if p.is_dir() and p.name.startswith("cam_")])
    if not available:
        raise RuntimeError(f"No camera folders found in {rgb_root}")

    if requested:
        missing = [name for name in requested if name not in available]
        if missing:
            raise RuntimeError(f"Requested camera(s) not found: {', '.join(missing)}")
        return requested

    return available


def load_fg_masks(mask_dir: Path, num_frames: int, target_hw: tuple[int, int]) -> np.ndarray:
    mask_paths = sorted(mask_dir.glob("frame_*.png"))
    if not mask_paths:
        raise FileNotFoundError(f"No masks found in {mask_dir}")

    if len(mask_paths) < num_frames:
        raise RuntimeError(f"Not enough masks in {mask_dir}: need {num_frames}, found {len(mask_paths)}")

    h, w = target_hw
    masks = []
    for p in mask_paths[:num_frames]:
        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed reading mask: {p}")
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        masks.append(mask > 127)

    return np.stack(masks, axis=0)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_dino_model(
    dino_repo: str = "facebookresearch/dinov2",
    dino_model: str = "dinov2_vits14",
    device: Optional[str] = None,
) -> torch.nn.Module:
    if device is None:
        device = pick_device()

    model = torch.hub.load(dino_repo, dino_model)
    return model.to(device).eval()


def extract_patch_features(frame_rgb: np.ndarray, model: torch.nn.Module, device: str) -> torch.Tensor:
    h, w = frame_rgb.shape[:2]
    patch_size = getattr(getattr(model, "patch_embed", None), "patch_size", (14, 14))
    if isinstance(patch_size, int):
        patch_h = patch_w = int(patch_size)
    else:
        patch_h = int(patch_size[0])
        patch_w = int(patch_size[1])

    # DINO ViT expects input spatial sizes to be multiples of patch size.
    target_h = max(patch_h, (h // patch_h) * patch_h)
    target_w = max(patch_w, (w // patch_w) * patch_w)
    if target_h != h or target_w != w:
        frame_rgb = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    x = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    x = x.to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = (x - mean) / std

    with torch.no_grad():
        out = model.forward_features(x)

    if isinstance(out, dict):
        tokens = out.get("x_norm_patchtokens")
        if tokens is None:
            tokens = out.get("x_prenorm")
    else:
        tokens = out

    if tokens is None or tokens.ndim != 3:
        raise RuntimeError("Unexpected DINO output format.")

    n_tokens = tokens.shape[1]
    feat_h = frame_rgb.shape[0] // patch_h
    feat_w = frame_rgb.shape[1] // patch_w
    if feat_h * feat_w != n_tokens:
        feat_h = int(round(np.sqrt(n_tokens)))
        feat_w = n_tokens // max(1, feat_h)
        if feat_h * feat_w != n_tokens:
            raise RuntimeError(f"Cannot reshape patch tokens: n_tokens={n_tokens}")

    feats = tokens[0].reshape(feat_h, feat_w, -1)
    feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats


def resize_mask_to_feat(mask_hw: np.ndarray, feat_hw: tuple[int, int]) -> np.ndarray:
    feat_h, feat_w = feat_hw
    small = cv2.resize(mask_hw.astype(np.uint8), (feat_w, feat_h), interpolation=cv2.INTER_NEAREST)
    return small > 0


def fit_pca_basis(feature_maps: list[torch.Tensor], masks: np.ndarray, fit_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    samples = []
    use_frames = min(fit_frames, len(feature_maps))
    for i in range(use_frames):
        feats = feature_maps[i].detach().cpu()
        mhw = resize_mask_to_feat(masks[i], (feats.shape[0], feats.shape[1]))
        vecs = feats[mhw]
        if vecs.numel() > 0:
            samples.append(vecs)

    if not samples:
        raise RuntimeError("No foreground DINO features found to fit PCA.")

    x = torch.cat(samples, dim=0)
    if x.shape[0] > 30000:
        idx = torch.randperm(x.shape[0])[:30000]
        x = x[idx]

    mu = x.mean(dim=0, keepdim=True)
    xc = x - mu
    _, _, v = torch.pca_lowrank(xc, q=3)
    basis = v[:, :3]
    return mu.squeeze(0), basis


def collect_feature_samples(
    feature_maps: list[torch.Tensor],
    masks: np.ndarray,
    fit_frames: int,
    max_samples: int = 30000,
) -> torch.Tensor:
    samples = []
    use_frames = min(fit_frames, len(feature_maps))
    for i in range(use_frames):
        feats = feature_maps[i].detach().cpu()
        mhw = resize_mask_to_feat(masks[i], (feats.shape[0], feats.shape[1]))
        vecs = feats[mhw]
        if vecs.numel() > 0:
            samples.append(vecs)

    if not samples:
        raise RuntimeError("No foreground DINO features found to collect PCA samples.")

    x = torch.cat(samples, dim=0)
    if x.shape[0] > max_samples:
        idx = torch.randperm(x.shape[0])[:max_samples]
        x = x[idx]
    return x


def collect_track_feature_samples(
    feature_groups: list[np.ndarray],
    max_samples: int = 60000,
    seed: int = 0,
) -> np.ndarray:
    if not feature_groups:
        raise RuntimeError("No DINO track feature groups provided.")

    rng = np.random.default_rng(seed)
    samples = []
    per_group_budget = max(1000, max_samples // max(1, len(feature_groups)))

    for features in feature_groups:
        flat = np.asarray(features, dtype=np.float32).reshape(-1, features.shape[-1])
        norms = np.linalg.norm(flat, axis=-1)
        valid = np.isfinite(norms) & (norms > 1e-8)
        vecs = flat[valid]
        if vecs.shape[0] == 0:
            continue
        if vecs.shape[0] > per_group_budget:
            idx = rng.choice(vecs.shape[0], size=per_group_budget, replace=False)
            vecs = vecs[idx]
        samples.append(vecs)

    if not samples:
        raise RuntimeError("No valid DINO track features found to fit a shared transform.")

    x = np.concatenate(samples, axis=0)
    if x.shape[0] > max_samples:
        idx = rng.choice(x.shape[0], size=max_samples, replace=False)
        x = x[idx]
    return x.astype(np.float32, copy=False)


def fit_shared_track_feature_transform(
    feature_groups: list[np.ndarray],
    output_dim: Optional[int] = None,
    max_samples: int = 60000,
    seed: int = 0,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = collect_track_feature_samples(feature_groups, max_samples=max_samples, seed=seed)
    mu = x.mean(axis=0, dtype=np.float64)
    xc = x.astype(np.float64) - mu[None, :]

    cov = (xc.T @ xc) / max(1, xc.shape[0] - 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    full_dim = evecs.shape[1]
    if output_dim is None:
        output_dim = full_dim
    output_dim = max(1, min(int(output_dim), full_dim))

    basis = evecs[:, :output_dim].astype(np.float32)
    scales = np.sqrt(np.maximum(evals[:output_dim], eps)).astype(np.float32)

    total_var = float(np.maximum(evals.sum(), eps))
    explained = (evals[:output_dim] / total_var).astype(np.float32)
    return mu.astype(np.float32), basis, scales, explained


def apply_track_feature_transform(
    features: np.ndarray,
    mu: np.ndarray,
    basis: np.ndarray,
    scales: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    flat = features.reshape(-1, features.shape[-1])
    norms = np.linalg.norm(flat, axis=-1)
    valid = np.isfinite(norms) & (norms > eps)

    transformed = np.zeros((flat.shape[0], basis.shape[1]), dtype=np.float32)
    if np.any(valid):
        centered = flat[valid] - mu[None, :]
        projected = centered @ basis
        projected = projected / np.maximum(scales[None, :], eps)
        proj_norms = np.linalg.norm(projected, axis=-1, keepdims=True)
        transformed[valid] = projected / np.maximum(proj_norms, eps)

    return transformed.reshape(features.shape[:-1] + (basis.shape[1],))


def fit_pca_visualization_basis(
    feature_map_groups: list[list[torch.Tensor]],
    mask_groups: list[np.ndarray],
    fit_frames: int,
    max_samples: int = 30000,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    if len(feature_map_groups) != len(mask_groups):
        raise ValueError("feature_map_groups and mask_groups must have the same length.")

    samples = []
    per_group_budget = max(1000, max_samples // max(1, len(feature_map_groups)))
    for feature_maps, masks in zip(feature_map_groups, mask_groups):
        samples.append(
            collect_feature_samples(
                feature_maps=feature_maps,
                masks=masks,
                fit_frames=fit_frames,
                max_samples=per_group_budget,
            )
        )

    x = torch.cat(samples, dim=0)
    if x.shape[0] > max_samples:
        idx = torch.randperm(x.shape[0])[:max_samples]
        x = x[idx]

    mu = x.mean(dim=0, keepdim=True)
    xc = x - mu
    _, _, v = torch.pca_lowrank(xc, q=3)
    basis = v[:, :3]

    proj = (x - mu) @ basis
    proj_np = proj.numpy()
    lo = np.percentile(proj_np, 2, axis=0).astype(np.float32)
    hi = np.percentile(proj_np, 98, axis=0).astype(np.float32)
    return mu.squeeze(0), basis, lo, hi


def colorize_features(
    feat_map: torch.Tensor,
    mask_hw: np.ndarray,
    mu: torch.Tensor,
    basis: torch.Tensor,
    out_hw: tuple[int, int],
    lo: Optional[np.ndarray] = None,
    hi: Optional[np.ndarray] = None,
) -> np.ndarray:
    feat_h, feat_w, _ = feat_map.shape
    proj = (feat_map.detach().cpu() - mu) @ basis
    proj_np = proj.numpy()

    if lo is None or hi is None:
        fg_small = resize_mask_to_feat(mask_hw, (feat_h, feat_w))
        if np.any(fg_small):
            fg_vals = proj_np[fg_small]
            lo = np.percentile(fg_vals, 2, axis=0)
            hi = np.percentile(fg_vals, 98, axis=0)
        else:
            lo = proj_np.reshape(-1, 3).min(axis=0)
            hi = proj_np.reshape(-1, 3).max(axis=0)

    denom = np.maximum(hi - lo, 1e-6)
    norm = np.clip((proj_np - lo) / denom, 0.0, 1.0)
    rgb_small = (norm * 255.0).astype(np.uint8)

    h, w = out_hw
    rgb = cv2.resize(rgb_small, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb[~mask_hw] = 0
    return rgb


def render_video(
    frames_rgb: np.ndarray,
    masks: np.ndarray,
    color_maps: list[np.ndarray],
    out_video: Path,
    out_preview: Path,
    fps: int,
    alpha: float,
    bg_dim: float,
) -> None:
    t, h, w, _ = frames_rgb.shape
    out_video.parent.mkdir(parents=True, exist_ok=True)
    out_preview.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {out_video}")

    last_bgr = None
    alpha = float(np.clip(alpha, 0.0, 1.0))
    bg_dim = float(np.clip(bg_dim, 0.0, 1.0))

    for i in range(t):
        out = blend_feature_overlay(frames_rgb[i], masks[i], color_maps[i], alpha=alpha, bg_dim=bg_dim)
        bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
        last_bgr = bgr

    writer.release()
    if last_bgr is not None:
        cv2.imwrite(str(out_preview), last_bgr)


def extract_feature_maps(
    frames_rgb: np.ndarray,
    model: torch.nn.Module,
    device: str,
    log_prefix: str = "[DINO Viewer]",
) -> list[torch.Tensor]:
    feature_maps = []
    for i, frame in enumerate(frames_rgb):
        feats = extract_patch_features(frame, model, device)
        feature_maps.append(feats)
        if (i + 1) % 10 == 0 or i == len(frames_rgb) - 1:
            print(f"{log_prefix} frame {i + 1}/{len(frames_rgb)}")
    return feature_maps


def blend_feature_overlay(
    frame_rgb: np.ndarray,
    mask_hw: np.ndarray,
    color_map: np.ndarray,
    alpha: float = 0.65,
    bg_dim: float = 0.2,
) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    bg_dim = float(np.clip(bg_dim, 0.0, 1.0))

    rgb = frame_rgb.astype(np.float32)
    color = color_map.astype(np.float32)
    bg = rgb * bg_dim
    fg = (1.0 - alpha) * rgb + alpha * color
    out = np.where(mask_hw[..., None], fg, bg)
    return np.clip(out, 0, 255).astype(np.uint8)


def frame_paths_for_camera(dataset_root: Path, camera_name: str) -> tuple[list[Path], list[Path]]:
    rgb_paths = sorted((dataset_root / "rgb" / camera_name).glob("frame_*.png"))
    mask_paths = sorted((dataset_root / "mask" / camera_name).glob("frame_*.png"))
    if not rgb_paths:
        raise FileNotFoundError(f"No RGB frames found for {camera_name}")
    if not mask_paths:
        raise FileNotFoundError(f"No masks found for {camera_name}")
    return rgb_paths, mask_paths


def draw_text_with_outline(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.6,
    thickness: int = 2,
) -> None:
    cv2.putText(
        image,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def make_labeled_tile(image_bgr: np.ndarray, label: str, header_h: int = 32) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    tile = np.full((h + header_h, w, 3), 18, dtype=np.uint8)
    tile[header_h:, :, :] = image_bgr
    draw_text_with_outline(tile, label, (10, 22), color=(200, 235, 255), scale=0.62, thickness=2)
    return tile


def make_grid_frame(
    labeled_tiles: list[np.ndarray],
    columns: int,
    frame_idx: int,
    num_frames: int,
    title: str,
) -> np.ndarray:
    if not labeled_tiles:
        raise ValueError("labeled_tiles cannot be empty")

    tile_h, tile_w = labeled_tiles[0].shape[:2]
    columns = max(1, columns)
    rows = math.ceil(len(labeled_tiles) / columns)
    title_h = 56
    gap = 10

    canvas_h = title_h + rows * tile_h + max(0, rows - 1) * gap
    canvas_w = columns * tile_w + max(0, columns - 1) * gap
    canvas = np.full((canvas_h, canvas_w, 3), 12, dtype=np.uint8)

    draw_text_with_outline(canvas, title, (14, 24), color=(255, 255, 255), scale=0.72, thickness=2)
    draw_text_with_outline(
        canvas,
        f"frame {frame_idx + 1}/{num_frames}",
        (14, 47),
        color=(210, 210, 210),
        scale=0.55,
        thickness=1,
    )

    for idx, tile in enumerate(labeled_tiles):
        row = idx // columns
        col = idx % columns
        y0 = title_h + row * (tile_h + gap)
        x0 = col * (tile_w + gap)
        canvas[y0:y0 + tile_h, x0:x0 + tile_w] = tile

    return canvas


def run_dino_feature_viewer(
    rgb_dir: Path,
    mask_dir: Path,
    output_video: Path,
    output_preview: Path,
    max_frames: Optional[int] = None,
    fps: int = 12,
    fit_frames: int = 8,
    alpha: float = 0.65,
    bg_dim: float = 0.2,
    dino_repo: str = "facebookresearch/dinov2",
    dino_model: str = "dinov2_vits14",
    device: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
) -> tuple[Path, Path]:
    if device is None:
        device = pick_device()
    print(f"[DINO Viewer] Using device: {device}")

    frames_rgb = load_frames(rgb_dir, max_frames)
    masks = load_fg_masks(mask_dir, frames_rgb.shape[0], (frames_rgb.shape[1], frames_rgb.shape[2]))

    if model is None:
        print(f"[DINO Viewer] Loading DINO from torch hub: {dino_repo} / {dino_model}")
        model = load_dino_model(dino_repo=dino_repo, dino_model=dino_model, device=device)

    print("[DINO Viewer] Extracting feature maps...")
    feature_maps = extract_feature_maps(frames_rgb, model, device, log_prefix="  [DINO Viewer]")

    print("[DINO Viewer] Fitting PCA projection...")
    mu, basis = fit_pca_basis(feature_maps, masks, fit_frames=fit_frames)

    print("[DINO Viewer] Colorizing DINO features...")
    color_maps = []
    for i, feats in enumerate(feature_maps):
        color = colorize_features(feats, masks[i], mu, basis, out_hw=(frames_rgb.shape[1], frames_rgb.shape[2]))
        color_maps.append(color)
        if (i + 1) % 10 == 0 or i == len(feature_maps) - 1:
            print(f"  frame {i + 1}/{len(feature_maps)}")

    print("[DINO Viewer] Rendering video...")
    render_video(
        frames_rgb=frames_rgb,
        masks=masks,
        color_maps=color_maps,
        out_video=output_video,
        out_preview=output_preview,
        fps=fps,
        alpha=alpha,
        bg_dim=bg_dim,
    )

    print(f"[DINO Viewer] Done. Video: {output_video}")
    print(f"[DINO Viewer] Done. Preview: {output_preview}")
    return output_video, output_preview


def run_multi_view_dino_feature_viewer(
    dataset_root: Path,
    output_root: Path,
    camera_names: Optional[list[str]] = None,
    max_frames: Optional[int] = None,
    fit_frames: int = 8,
    fps: int = 12,
    alpha: float = 0.65,
    bg_dim: float = 0.2,
    max_pca_samples: int = 60000,
    grid_columns: int = 4,
    dino_repo: str = "facebookresearch/dinov2",
    dino_model: str = "dinov2_vits14",
    device: Optional[str] = None,
) -> tuple[Path, Path]:
    resolved_camera_names = discover_cameras(dataset_root, camera_names)
    if device is None:
        device = pick_device()
    print(f"[MultiView DINO] Using device: {device}")
    print(f"[MultiView DINO] Cameras: {', '.join(resolved_camera_names)}")

    model = load_dino_model(dino_repo=dino_repo, dino_model=dino_model, device=device)

    print("[MultiView DINO] Pass 1/2: fitting shared PCA basis...")
    feature_groups = []
    mask_groups = []
    for camera_name in resolved_camera_names:
        rgb_dir = dataset_root / "rgb" / camera_name
        mask_dir = dataset_root / "mask" / camera_name
        fit_cap = fit_frames if max_frames is None else min(fit_frames, max_frames)
        frames_rgb = load_frames(rgb_dir, fit_cap)
        masks = load_fg_masks(mask_dir, frames_rgb.shape[0], (frames_rgb.shape[1], frames_rgb.shape[2]))
        feature_maps = extract_feature_maps(
            frames_rgb,
            model,
            device,
            log_prefix=f"  [MultiView DINO][fit][{camera_name}]",
        )
        feature_groups.append(feature_maps)
        mask_groups.append(masks)

    mu, basis, lo, hi = fit_pca_visualization_basis(
        feature_map_groups=feature_groups,
        mask_groups=mask_groups,
        fit_frames=fit_frames,
        max_samples=max_pca_samples,
    )

    mv_root = output_root / "multiview_dino"
    mv_root.mkdir(parents=True, exist_ok=True)
    camera_writers: dict[str, cv2.VideoWriter] = {}
    camera_video_paths: dict[str, Path] = {}
    camera_preview_paths: dict[str, Path] = {}
    last_frames: dict[str, np.ndarray] = {}

    per_camera_paths = {name: frame_paths_for_camera(dataset_root, name) for name in resolved_camera_names}
    num_frames = min(
        min(len(rgb_paths), len(mask_paths))
        for rgb_paths, mask_paths in per_camera_paths.values()
    )
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)
    if num_frames <= 0:
        raise RuntimeError("No frames available for multiview rendering.")

    sample_bgr = cv2.imread(str(per_camera_paths[resolved_camera_names[0]][0][0]), cv2.IMREAD_COLOR)
    if sample_bgr is None:
        raise RuntimeError(f"Failed reading sample frame for {resolved_camera_names[0]}")
    tile_h, tile_w = sample_bgr.shape[:2]

    montage_columns = max(1, int(grid_columns))
    sample_tile = make_labeled_tile(sample_bgr, resolved_camera_names[0])
    sample_grid = make_grid_frame(
        [sample_tile for _ in resolved_camera_names],
        columns=montage_columns,
        frame_idx=0,
        num_frames=max(1, num_frames),
        title="Shared-PCA DINO Features",
    )
    montage_video_path = mv_root / "all_cams_shared_pca.mp4"
    montage_preview_path = mv_root / "all_cams_shared_pca_last_frame.png"
    montage_writer = cv2.VideoWriter(
        str(montage_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1, int(fps)),
        (sample_grid.shape[1], sample_grid.shape[0]),
    )
    if not montage_writer.isOpened():
        raise RuntimeError(f"Could not open montage video writer for {montage_video_path}")

    for camera_name in resolved_camera_names:
        cam_dir = mv_root / camera_name
        cam_dir.mkdir(parents=True, exist_ok=True)
        video_path = cam_dir / "features_shared_pca.mp4"
        preview_path = cam_dir / "last_frame.png"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(1, int(fps)),
            (tile_w, tile_h),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {video_path}")
        camera_writers[camera_name] = writer
        camera_video_paths[camera_name] = video_path
        camera_preview_paths[camera_name] = preview_path

    print("[MultiView DINO] Pass 2/2: rendering shared-color videos...")
    montage_last = None
    for frame_idx in range(num_frames):
        tiles = []
        for camera_name in resolved_camera_names:
            rgb_path = per_camera_paths[camera_name][0][frame_idx]
            mask_path = per_camera_paths[camera_name][1][frame_idx]

            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if bgr is None:
                raise RuntimeError(f"Failed reading RGB frame: {rgb_path}")
            if mask is None:
                raise RuntimeError(f"Failed reading mask frame: {mask_path}")

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mask_bool = mask > 127
            feat_map = extract_patch_features(rgb, model, device)
            color_map = colorize_features(
                feat_map,
                mask_bool,
                mu,
                basis,
                out_hw=(rgb.shape[0], rgb.shape[1]),
                lo=lo,
                hi=hi,
            )
            overlay_rgb = blend_feature_overlay(
                rgb,
                mask_bool,
                color_map,
                alpha=alpha,
                bg_dim=bg_dim,
            )
            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
            camera_writers[camera_name].write(overlay_bgr)
            last_frames[camera_name] = overlay_bgr
            tiles.append(make_labeled_tile(overlay_bgr, camera_name))

        grid = make_grid_frame(
            tiles,
            columns=montage_columns,
            frame_idx=frame_idx,
            num_frames=num_frames,
            title="Shared-PCA DINO Features",
        )
        montage_writer.write(grid)
        montage_last = grid

        if (frame_idx + 1) % 10 == 0 or frame_idx == num_frames - 1:
            print(f"  [MultiView DINO] frame {frame_idx + 1}/{num_frames}")

    montage_writer.release()
    if montage_last is not None:
        cv2.imwrite(str(montage_preview_path), montage_last)

    for camera_name, writer in camera_writers.items():
        writer.release()
        if camera_name in last_frames:
            cv2.imwrite(str(camera_preview_paths[camera_name]), last_frames[camera_name])

    print(f"[MultiView DINO] Montage video: {montage_video_path}")
    print(f"[MultiView DINO] Montage preview: {montage_preview_path}")
    for camera_name in resolved_camera_names:
        print(
            f"[MultiView DINO] {camera_name}: "
            f"{camera_video_paths[camera_name]} | {camera_preview_paths[camera_name]}"
        )

    return montage_video_path, montage_preview_path


def sample_features_at_points(
    feat_map: torch.Tensor,    # (feat_h, feat_w, C)
    points_xy: torch.Tensor,   # (N, 2) float – pixel coords (x, y)
    frame_hw: tuple[int, int], # (H, W) of the original RGB frame
) -> torch.Tensor:             # (N, C) float – L2-normalised
    """
    Bilinear-sample a DINO feature map at given pixel positions.

    The feature map lives on a coarser patch grid (e.g., H/14 × W/14).
    Pixel coordinates are mapped to the normalised [-1, 1] range expected by
    ``torch.nn.functional.grid_sample``.

    Parameters
    ----------
    feat_map   : (feat_h, feat_w, C) feature tensor from extract_patch_features.
    points_xy  : (N, 2) float tensor with (x, y) pixel coordinates in [0, W] × [0, H].
    frame_hw   : (H, W) dimensions of the original RGB frame.

    Returns
    -------
    (N, C) L2-normalised feature vectors at the requested positions.
    """
    feat_h, feat_w, C = feat_map.shape
    H, W = frame_hw
    device = feat_map.device

    # Normalise pixel coords to [-1, 1] (align_corners=True convention)
    norm_x = (points_xy[:, 0].float() / max(W - 1, 1)) * 2.0 - 1.0
    norm_y = (points_xy[:, 1].float() / max(H - 1, 1)) * 2.0 - 1.0

    grid = torch.stack([norm_x, norm_y], dim=-1)    # (N, 2)
    grid = grid.unsqueeze(0).unsqueeze(0)            # (1, 1, N, 2)

    feat_t = feat_map.permute(2, 0, 1).unsqueeze(0) # (1, C, feat_h, feat_w)

    sampled = torch.nn.functional.grid_sample(
        feat_t.float(),
        grid.to(device).float(),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # (1, C, 1, N)

    sampled = sampled[0, :, 0, :].permute(1, 0)     # (N, C)

    # Re-normalise (bilinear interpolation breaks exact unit length)
    norms = sampled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return sampled / norms


def extract_dino_track_features(
    rgb_dir: Path,
    tracks: np.ndarray,        # (T, N, 2)  pixel coords (x, y) – CoTracker output
    visibilities: np.ndarray,  # (T, N)     bool / float
    dino_repo: str = "facebookresearch/dinov2",
    dino_model_name: str = "dinov2_vits14",
    device: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
    max_frames: Optional[int] = None,
) -> np.ndarray:               # (T, N, C) float32  L2-normalised
    """
    Extract DINO feature descriptors at CoTracker track positions.

    For each (frame, track) pair where the track is visible, the DINO patch
    feature map is bilinearly sampled at the track's pixel location.
    Invisible tracks receive zero vectors.

    Parameters
    ----------
    rgb_dir          : directory containing frame_*.png RGB images.
    tracks           : (T, N, 2) array of pixel coordinates from CoTracker.
    visibilities     : (T, N) boolean/float visibility flags.
    dino_repo        : torch.hub repo string.
    dino_model_name  : model name within the repo.
    device           : target device; auto-detected if None.
    model            : pre-loaded DINO model; loaded from hub if None.
    max_frames       : cap on the number of frames to process.

    Returns
    -------
    (T, N, C) float32 array of L2-normalised DINO descriptors.
    Invisible tracks have zero vectors.
    """
    if device is None:
        device = pick_device()

    print(f"[DINO Track Features] Device: {device}")

    frames_rgb = load_frames(rgb_dir, max_frames)
    T_data, N = tracks.shape[:2]
    T = min(len(frames_rgb), T_data)

    if model is None:
        print(f"[DINO Track Features] Loading model: {dino_repo}/{dino_model_name}")
        model = load_dino_model(dino_repo=dino_repo, dino_model=dino_model_name, device=device)

    # Infer feature dimensionality C from a single test forward pass
    test_feats = extract_patch_features(frames_rgb[0], model, device)
    C = test_feats.shape[-1]
    print(f"[DINO Track Features] Feature dim C={C}, T={T}, N={N}")

    out = np.zeros((T, N, C), dtype=np.float32)

    for t in range(T):
        vis_idx = np.where(np.asarray(visibilities[t]).astype(bool))[0]
        if len(vis_idx) == 0:
            continue

        feat_map = extract_patch_features(frames_rgb[t], model, device)  # (fH, fW, C)
        frame_hw = (frames_rgb[t].shape[0], frames_rgb[t].shape[1])

        pts = torch.from_numpy(tracks[t][vis_idx]).float().to(device)    # (Nv, 2)
        sampled = sample_features_at_points(feat_map.to(device), pts, frame_hw)  # (Nv, C)
        out[t][vis_idx] = sampled.detach().cpu().numpy()

        if (t + 1) % 10 == 0 or t == T - 1:
            print(f"  [DINO Track Features] frame {t + 1}/{T}")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize DINO features in single-view or multiview mode.")
    parser.add_argument(
        "--multiview",
        action="store_true",
        help="Run shared-PCA multiview visualization across cameras instead of the single-camera viewer.",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("blade_103706"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/blade_103706"),
        help="Output root used by multiview mode.",
    )
    parser.add_argument(
        "--cam",
        action="append",
        default=None,
        help="Camera to include in multiview mode; repeat to select multiple cameras. Default: all cameras.",
    )
    parser.add_argument("--rgb-dir", type=Path, default=Path("blade_103706/rgb/cam_000"))
    parser.add_argument("--mask-dir", type=Path, default=Path("blade_103706/mask/cam_000"))
    parser.add_argument("--output-video", type=Path, default=Path("outputs/blade_103706_cam_000_dino_features.mp4"))
    parser.add_argument(
        "--output-preview",
        type=Path,
        default=Path("outputs/blade_103706_cam_000_dino_features_last_frame.png"),
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--fit-frames", type=int, default=8, help="How many initial frames to use for PCA fit")
    parser.add_argument("--alpha", type=float, default=0.65, help="Overlay strength in foreground")
    parser.add_argument("--bg-dim", type=float, default=0.2, help="Background brightness factor")
    parser.add_argument("--max-pca-samples", type=int, default=60000)
    parser.add_argument("--grid-columns", type=int, default=4, help="Columns used for the multiview montage video")
    parser.add_argument("--dino-repo", type=str, default="facebookresearch/dinov2")
    parser.add_argument("--dino-model", type=str, default="dinov2_vits14")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu/cuda/mps")
    args = parser.parse_args()

    if args.multiview:
        run_multi_view_dino_feature_viewer(
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            camera_names=args.cam,
            max_frames=args.max_frames,
            fit_frames=args.fit_frames,
            fps=args.fps,
            alpha=args.alpha,
            bg_dim=args.bg_dim,
            max_pca_samples=args.max_pca_samples,
            grid_columns=args.grid_columns,
            dino_repo=args.dino_repo,
            dino_model=args.dino_model,
            device=args.device,
        )
        return

    run_dino_feature_viewer(
        rgb_dir=args.rgb_dir,
        mask_dir=args.mask_dir,
        output_video=args.output_video,
        output_preview=args.output_preview,
        max_frames=args.max_frames,
        fps=args.fps,
        fit_frames=args.fit_frames,
        alpha=args.alpha,
        bg_dim=args.bg_dim,
        dino_repo=args.dino_repo,
        dino_model=args.dino_model,
        device=args.device,
    )


if __name__ == "__main__":
    main()
