#!/usr/bin/env python3
import argparse
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
    parser = argparse.ArgumentParser(description="Visualize DINO features as RGB overlays on foreground.")
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
    parser.add_argument("--dino-repo", type=str, default="facebookresearch/dinov2")
    parser.add_argument("--dino-model", type=str, default="dinov2_vits14")
    args = parser.parse_args()

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
    )


if __name__ == "__main__":
    main()
