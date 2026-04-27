#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil
from typing import Optional

import cv2
import numpy as np
import torch

from dino_feature_viewer import (
    apply_track_feature_transform,
    extract_dino_track_features,
    fit_shared_track_feature_transform,
    load_dino_model,
    run_dino_feature_viewer,
)


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


def load_fg_masks(
    mask_dir: Path,
    max_frames: Optional[int] = None,
    target_hw: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    mask_paths = sorted(mask_dir.glob("frame_*.png"))
    if not mask_paths:
        raise FileNotFoundError(f"No masks found in {mask_dir}")

    if max_frames is not None:
        mask_paths = mask_paths[:max_frames]

    masks = []
    for p in mask_paths:
        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed reading mask: {p}")
        if target_hw is not None and (mask.shape[0], mask.shape[1]) != target_hw:
            mask = cv2.resize(mask, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
        masks.append(mask > 127)

    return np.stack(masks, axis=0)


def load_saved_tracks(track_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not track_path.exists():
        raise FileNotFoundError(f"Missing saved CoTracker tracks: {track_path}")

    with np.load(track_path) as data:
        if "tracks" not in data or "visibilities" not in data:
            raise RuntimeError(f"Invalid saved track file: {track_path}")
        tracks = np.asarray(data["tracks"], dtype=np.float32)
        visibilities = np.asarray(data["visibilities"], dtype=bool)

    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise RuntimeError(f"Unexpected saved track shape in {track_path}: {tracks.shape}")
    if visibilities.shape != tracks.shape[:2]:
        raise RuntimeError(
            f"Saved track/visibility mismatch in {track_path}: tracks={tracks.shape}, visibilities={visibilities.shape}"
        )

    return tracks, visibilities


def draw_tracks_green(
    frames_rgb: np.ndarray,
    fg_masks: np.ndarray,
    tracks: torch.Tensor,
    visibilities: torch.Tensor,
    out_video: Path,
    out_preview: Path,
    fps: int,
    point_radius: int,
) -> None:
    t, h, w, _ = frames_rgb.shape

    out_video.parent.mkdir(parents=True, exist_ok=True)
    out_preview.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {out_video}")

    tracks_np = tracks[0].detach().cpu().numpy()  # (T, N, 2)
    vis_np = visibilities[0].detach().cpu().numpy()  # (T, N)

    preview_bgr = None
    for i in range(t):
        bgr = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR)

        # Draw only visible points on each frame in green.
        visible_idx = np.where(vis_np[i] > 0)[0]
        for j in visible_idx:
            x, y = tracks_np[i, j]
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < w and 0 <= yi < h:
                if not fg_masks[i, yi, xi]:
                    continue
                cv2.circle(bgr, (xi, yi), radius=point_radius, color=(0, 255, 0), thickness=-1)

        writer.write(bgr)
        if i == t - 1:
            preview_bgr = bgr.copy()

    writer.release()

    if preview_bgr is not None:
        cv2.imwrite(str(out_preview), preview_bgr)


def fit_and_save_shared_dino_track_features(
    output_root: Path,
    camera_names: list[str],
    output_dim: Optional[int],
    max_samples: int,
    seed: int = 0,
) -> Optional[Path]:
    feature_groups = []
    loaded_features: dict[str, np.ndarray] = {}

    for camera_name in camera_names:
        feature_path = output_root / camera_name / "dino" / "track_features.npz"
        if not feature_path.exists():
            print(f"Skipping shared DINO alignment for {camera_name}: missing {feature_path}")
            continue

        with np.load(feature_path) as feature_data:
            features = np.asarray(feature_data["features"], dtype=np.float32)

        loaded_features[camera_name] = features
        feature_groups.append(features)

    if len(feature_groups) < 2:
        print("Skipping shared DINO alignment: need at least two cameras with extracted DINO features.")
        return None

    mu, basis, scales, explained = fit_shared_track_feature_transform(
        feature_groups=feature_groups,
        output_dim=output_dim,
        max_samples=max_samples,
        seed=seed,
    )

    transform_path = output_root / "multiview_dino_alignment.npz"
    np.savez_compressed(
        transform_path,
        mu=mu,
        basis=basis,
        scales=scales,
        explained_variance_ratio=explained,
        cameras=np.array(sorted(loaded_features.keys())),
    )

    total_explained = float(explained.sum())
    print(
        "Fitted shared DINO alignment "
        f"(raw_dim={mu.shape[0]}, aligned_dim={basis.shape[1]}, explained={total_explained:.4f})"
    )
    print(f"Saved shared DINO alignment → {transform_path}")

    for camera_name, features in loaded_features.items():
        aligned = apply_track_feature_transform(features, mu, basis, scales)
        aligned_path = output_root / camera_name / "dino" / "track_features_aligned.npz"
        np.savez_compressed(aligned_path, features=aligned)
        print(f"Saved aligned DINO track features → {aligned_path}  shape={aligned.shape}")

    return transform_path


def run_camera(
    cam_name: str,
    rgb_dir: Path,
    mask_dir: Optional[Path],
    output_root: Path,
    device: str,
    args: argparse.Namespace,
) -> None:
    cam_out = output_root / cam_name
    cotracker_out = cam_out / "cotracker"
    dino_out = cam_out / "dino"

    if args.clean_output and cam_out.exists():
        shutil.rmtree(cam_out)

    cotracker_video = cotracker_out / "tracks_green.mp4"
    cotracker_preview = cotracker_out / "last_frame.png"
    dino_video = dino_out / "features.mp4"
    dino_preview = dino_out / "last_frame.png"

    print(f"\n=== Camera {cam_name} ===")
    print(f"RGB dir: {rgb_dir}")
    need_dino_viewer = not args.skip_dino_viewer
    need_dino_features = not args.skip_dino_features
    need_tracking = not args.reuse_existing_tracks
    need_masks = need_tracking or need_dino_viewer

    if mask_dir is not None:
        print(f"Mask dir: {mask_dir}")

    frames_rgb = load_frames(rgb_dir, args.max_frames)
    fg_masks = None
    if need_masks:
        if mask_dir is None:
            raise FileNotFoundError(f"Mask directory required for {cam_name} but not provided.")
        fg_masks = load_fg_masks(mask_dir, args.max_frames, target_hw=(frames_rgb.shape[1], frames_rgb.shape[2]))
        if fg_masks.shape[0] != frames_rgb.shape[0]:
            raise RuntimeError(
                f"RGB/mask frame count mismatch for {cam_name}: rgb={frames_rgb.shape[0]}, mask={fg_masks.shape[0]}"
            )

    if need_tracking:
        assert fg_masks is not None
        video_t = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)
        segm_mask_t = torch.from_numpy(fg_masks[0].astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

        print("Loading CoTracker model from torch hub...")
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        model = model.to(device).eval()

        print("Running tracking...")
        with torch.no_grad():
            tracks, visibilities = model(video_t, grid_size=args.grid_size, segm_mask=segm_mask_t)

        visible_first = int((visibilities[0, 0] > 0).sum().item())
        print(f"Tracked points total: {tracks.shape[2]} | visible on first frame: {visible_first}")
        if tracks.shape[2] < 50:
            print("Warning: too few points. Increase --grid-size (e.g. 224 or 256).")

        cotracker_out.mkdir(parents=True, exist_ok=True)
        tracks_np = tracks[0].detach().cpu().numpy()  # (T, N, 2)
        vis_np = visibilities[0].detach().cpu().numpy()  # (T, N)
        np.savez_compressed(
            cotracker_out / "tracks.npz",
            tracks=tracks_np,
            visibilities=vis_np.astype(bool),
            frame_hw=np.array([frames_rgb.shape[1], frames_rgb.shape[2]], dtype=np.int32),
        )
        print(f"Saved CoTracker tracks → {cotracker_out / 'tracks.npz'}  shape={tracks_np.shape}")

        print("Rendering CoTracker output...")
        draw_tracks_green(
            frames_rgb=frames_rgb,
            fg_masks=fg_masks,
            tracks=tracks,
            visibilities=visibilities,
            out_video=cotracker_video,
            out_preview=cotracker_preview,
            fps=args.fps,
            point_radius=args.point_radius,
        )
    else:
        track_path = cotracker_out / "tracks.npz"
        tracks_np, vis_np = load_saved_tracks(track_path)
        print(f"Using existing CoTracker tracks → {track_path}  shape={tracks_np.shape}")

    dino_model = None
    if need_dino_viewer or need_dino_features:
        print(f"Loading DINO model from torch hub: {args.dino_repo} / {args.dino_model}")
        dino_model = load_dino_model(
            dino_repo=args.dino_repo,
            dino_model=args.dino_model,
            device=device,
        )

    if not args.skip_dino_viewer:
        assert fg_masks is not None
        print("Running external DINO feature viewer...")
        run_dino_feature_viewer(
            rgb_dir=rgb_dir,
            mask_dir=mask_dir,
            output_video=dino_video,
            output_preview=dino_preview,
            max_frames=args.max_frames,
            fps=args.fps,
            fit_frames=args.dino_fit_frames,
            alpha=args.dino_alpha,
            bg_dim=args.dino_bg_dim,
            dino_repo=args.dino_repo,
            dino_model=args.dino_model,
            device=device,
            model=dino_model,
        )

    if need_dino_features:
        print("Extracting DINO features at CoTracker track positions...")
        track_feats = extract_dino_track_features(
            rgb_dir=rgb_dir,
            tracks=tracks_np,
            visibilities=vis_np.astype(bool),
            dino_repo=args.dino_repo,
            dino_model_name=args.dino_model,
            device=device,
            model=dino_model,
            max_frames=args.max_frames,
        )
        dino_out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dino_out / "track_features.npz", features=track_feats)
        print(f"Saved DINO track features → {dino_out / 'track_features.npz'}  shape={track_feats.shape}")

    if need_tracking:
        print(f"Done CoTracker video: {cotracker_video}")
        print(f"Done CoTracker preview: {cotracker_preview}")
    if not args.skip_dino_viewer:
        print(f"Done DINO video: {dino_video}")
        print(f"Done DINO preview: {dino_preview}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run per-camera CoTracker + DINO extraction with structured outputs."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("blade_103706"),
        help="Dataset root containing rgb/ and mask/ folders",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/blade_103706"),
        help="Root folder where structured outputs are written",
    )
    parser.add_argument("--cam", type=str, default="cam_000", help="Single camera name (ignored with --all-cams)")
    parser.add_argument("--all-cams", action="store_true", help="Process all cameras found in dataset_root/rgb")
    parser.add_argument(
        "--clean-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete previous outputs before writing new results",
    )
    parser.add_argument("--dino-fit-frames", type=int, default=8, help="Initial frames used to fit DINO PCA colors")
    parser.add_argument("--dino-alpha", type=float, default=0.65, help="DINO overlay strength on foreground")
    parser.add_argument("--dino-bg-dim", type=float, default=0.2, help="Background brightness in DINO visualization")
    parser.add_argument("--skip-dino-viewer", action="store_true", help="Skip DINO feature visualization export")
    parser.add_argument(
        "--skip-dino-features",
        action="store_true",
        help="Skip DINO descriptor extraction at CoTracker track positions",
    )
    parser.add_argument(
        "--reuse-existing-tracks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse existing output-root/cam_XXX/cotracker/tracks.npz instead of rerunning CoTracker",
    )
    parser.add_argument("--dino-repo", type=str, default="facebookresearch/dinov2",
                        help="torch.hub repository for DINO")
    parser.add_argument("--dino-model", type=str, default="dinov2_vits14",
                        help="DINO model name within the hub repo")
    parser.add_argument("--fps", type=int, default=12, help="Output video FPS")
    parser.add_argument("--grid-size", type=int, default=224, help="Grid size for dense tracking (larger = more points)")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on number of frames")
    parser.add_argument("--point-radius", type=int, default=2, help="Radius of rendered green points")
    parser.add_argument(
        "--shared-dino-align",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit one shared DINO transform across all processed cameras and save aligned track descriptors",
    )
    parser.add_argument(
        "--shared-dino-dim",
        type=int,
        default=None,
        help="Optional output dimension for the shared DINO-aligned descriptors (defaults to raw DINO dim)",
    )
    parser.add_argument(
        "--shared-dino-max-samples",
        type=int,
        default=60000,
        help="Maximum number of pooled DINO descriptors used to fit the shared multiview alignment",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    if args.reuse_existing_tracks and args.clean_output:
        print("Disabling --clean-output because --reuse-existing-tracks needs the saved tracks to remain in place.")
        args.clean_output = False

    rgb_root = args.dataset_root / "rgb"
    mask_root = args.dataset_root / "mask"
    if not rgb_root.exists():
        raise FileNotFoundError(f"Expected RGB folder not found under: {args.dataset_root}")
    if (not args.reuse_existing_tracks or not args.skip_dino_viewer) and not mask_root.exists():
        raise FileNotFoundError(f"Expected mask folder not found under: {args.dataset_root}")

    if args.clean_output and args.output_root.exists():
        print(f"Cleaning previous outputs in: {args.output_root}")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.all_cams:
        cams = sorted([p.name for p in rgb_root.iterdir() if p.is_dir() and p.name.startswith("cam_")])
        if not cams:
            raise RuntimeError(f"No camera folders found in: {rgb_root}")
    else:
        cams = [args.cam]

    print(f"Cameras to process: {', '.join(cams)}")
    for cam_name in cams:
        rgb_dir = rgb_root / cam_name
        mask_dir = mask_root / cam_name
        if not rgb_dir.exists():
            print(f"Skipping {cam_name}: missing RGB dir {rgb_dir}")
            continue
        if (not args.reuse_existing_tracks or not args.skip_dino_viewer) and not mask_dir.exists():
            print(f"Skipping {cam_name}: missing mask dir {mask_dir}")
            continue

        run_camera(
            cam_name=cam_name,
            rgb_dir=rgb_dir,
            mask_dir=mask_dir if mask_dir.exists() else None,
            output_root=args.output_root,
            device=device,
            args=args,
        )

    if args.shared_dino_align and not args.skip_dino_features:
        fit_and_save_shared_dino_track_features(
            output_root=args.output_root,
            camera_names=cams,
            output_dim=args.shared_dino_dim,
            max_samples=args.shared_dino_max_samples,
        )

    print("\nAll requested cameras processed.")
    print(f"Structured outputs root: {args.output_root}")
    print("Run pair_camera_matching.py separately to match tracks across camera pairs.")


if __name__ == "__main__":
    main()
