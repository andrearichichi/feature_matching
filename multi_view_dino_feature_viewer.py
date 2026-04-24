#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from dino_feature_viewer import (
    extract_patch_features,
    load_dino_model,
    load_fg_masks,
    load_frames,
    pick_device,
)


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


def resize_mask_to_feat(mask_hw: np.ndarray, feat_hw: tuple[int, int]) -> np.ndarray:
    feat_h, feat_w = feat_hw
    small = cv2.resize(mask_hw.astype(np.uint8), (feat_w, feat_h), interpolation=cv2.INTER_NEAREST)
    return small > 0


def extract_feature_maps(
    frames_rgb: np.ndarray,
    model: torch.nn.Module,
    device: str,
    log_prefix: str,
) -> list[torch.Tensor]:
    feature_maps = []
    for i, frame in enumerate(frames_rgb):
        feats = extract_patch_features(frame, model, device)
        feature_maps.append(feats)
        if (i + 1) % 10 == 0 or i == len(frames_rgb) - 1:
            print(f"{log_prefix} frame {i + 1}/{len(frames_rgb)}")
    return feature_maps


def collect_feature_samples(
    feature_maps: list[torch.Tensor],
    masks: np.ndarray,
    fit_frames: int,
    max_samples: int,
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


def fit_pca_visualization_basis(
    feature_map_groups: list[list[torch.Tensor]],
    mask_groups: list[np.ndarray],
    fit_frames: int,
    max_samples: int,
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


def colorize_features_shared(
    feat_map: torch.Tensor,
    mask_hw: np.ndarray,
    mu: torch.Tensor,
    basis: torch.Tensor,
    out_hw: tuple[int, int],
    lo: np.ndarray,
    hi: np.ndarray,
) -> np.ndarray:
    proj = (feat_map.detach().cpu() - mu) @ basis
    proj_np = proj.numpy()
    denom = np.maximum(hi - lo, 1e-6)
    norm = np.clip((proj_np - lo) / denom, 0.0, 1.0)
    rgb_small = (norm * 255.0).astype(np.uint8)

    h, w = out_hw
    rgb = cv2.resize(rgb_small, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb[~mask_hw] = 0
    return rgb


def blend_feature_overlay(
    frame_rgb: np.ndarray,
    mask_hw: np.ndarray,
    color_map: np.ndarray,
    alpha: float,
    bg_dim: float,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render DINO features with a shared PCA basis across multiple cameras."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("blade_103706"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/blade_103706"))
    parser.add_argument(
        "--cam",
        action="append",
        default=None,
        help="Camera to include; repeat the flag to select multiple cameras. Default: all cameras.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fit-frames", type=int, default=8, help="Frames per camera used to fit the shared PCA basis")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=0.65, help="Overlay strength in foreground")
    parser.add_argument("--bg-dim", type=float, default=0.2, help="Background brightness factor")
    parser.add_argument("--max-pca-samples", type=int, default=60000)
    parser.add_argument("--grid-columns", type=int, default=4, help="Columns used for the montage video")
    parser.add_argument("--dino-repo", type=str, default="facebookresearch/dinov2")
    parser.add_argument("--dino-model", type=str, default="dinov2_vits14")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu/cuda/mps")
    args = parser.parse_args()

    camera_names = discover_cameras(args.dataset_root, args.cam)
    device = args.device or pick_device()
    print(f"[MultiView DINO] Using device: {device}")
    print(f"[MultiView DINO] Cameras: {', '.join(camera_names)}")

    model = load_dino_model(dino_repo=args.dino_repo, dino_model=args.dino_model, device=device)

    print("[MultiView DINO] Pass 1/2: fitting shared PCA basis...")
    feature_groups = []
    mask_groups = []
    for camera_name in camera_names:
        rgb_dir = args.dataset_root / "rgb" / camera_name
        mask_dir = args.dataset_root / "mask" / camera_name
        fit_cap = args.fit_frames if args.max_frames is None else min(args.fit_frames, args.max_frames)
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
        fit_frames=args.fit_frames,
        max_samples=args.max_pca_samples,
    )

    mv_root = args.output_root / "multiview_dino"
    mv_root.mkdir(parents=True, exist_ok=True)
    camera_writers: dict[str, cv2.VideoWriter] = {}
    camera_video_paths: dict[str, Path] = {}
    camera_preview_paths: dict[str, Path] = {}
    last_frames: dict[str, np.ndarray] = {}

    per_camera_paths = {name: frame_paths_for_camera(args.dataset_root, name) for name in camera_names}
    num_frames = min(
        min(len(rgb_paths), len(mask_paths))
        for rgb_paths, mask_paths in per_camera_paths.values()
    )
    if args.max_frames is not None:
        num_frames = min(num_frames, args.max_frames)
    if num_frames <= 0:
        raise RuntimeError("No frames available for multiview rendering.")

    sample_bgr = cv2.imread(str(per_camera_paths[camera_names[0]][0][0]), cv2.IMREAD_COLOR)
    if sample_bgr is None:
        raise RuntimeError(f"Failed reading sample frame for {camera_names[0]}")
    tile_h, tile_w = sample_bgr.shape[:2]

    montage_columns = max(1, int(args.grid_columns))
    sample_tile = make_labeled_tile(sample_bgr, camera_names[0])
    sample_grid = make_grid_frame(
        [sample_tile for _ in camera_names],
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
        max(1, int(args.fps)),
        (sample_grid.shape[1], sample_grid.shape[0]),
    )
    if not montage_writer.isOpened():
        raise RuntimeError(f"Could not open montage video writer for {montage_video_path}")

    for camera_name in camera_names:
        cam_dir = mv_root / camera_name
        cam_dir.mkdir(parents=True, exist_ok=True)
        video_path = cam_dir / "features_shared_pca.mp4"
        preview_path = cam_dir / "last_frame.png"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(1, int(args.fps)),
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
        for camera_name in camera_names:
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
            color_map = colorize_features_shared(
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
                alpha=args.alpha,
                bg_dim=args.bg_dim,
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
    for camera_name in camera_names:
        print(
            f"[MultiView DINO] {camera_name}: "
            f"{camera_video_paths[camera_name]} | {camera_preview_paths[camera_name]}"
        )


if __name__ == "__main__":
    main()
