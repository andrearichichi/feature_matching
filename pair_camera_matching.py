#!/usr/bin/env python3
import argparse
import csv
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class CameraModel:
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    position: np.ndarray
    world_to_camera: np.ndarray
    camera_to_world: np.ndarray


def list_frame_paths(rgb_dir: Path) -> list[Path]:
    frame_paths = sorted(rgb_dir.glob("frame_*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No RGB frames found in {rgb_dir}")
    return frame_paths


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vec / norm


def pair_output_stem(camera_a: CameraModel, camera_b: CameraModel) -> str:
    return f"{camera_a.name}__{camera_b.name}"


def make_match_color(index: int) -> tuple[int, int, int]:
    hue = int((index * 29) % 180)
    hsv = np.array([[[hue, 220, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_text_with_outline(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.65,
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


def draw_match_marker(
    image: np.ndarray,
    point_xy: tuple[int, int],
    color: tuple[int, int, int],
    radius: int,
    label: Optional[str] = None,
) -> None:
    cv2.circle(image, point_xy, radius + 2, (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(image, point_xy, radius, color, thickness=-1, lineType=cv2.LINE_AA)
    if label is not None:
        draw_text_with_outline(
            image,
            label,
            (point_xy[0] + radius + 4, point_xy[1] - radius - 2),
            color=color,
            scale=0.45,
            thickness=1,
        )


def draw_match_line(
    image: np.ndarray,
    pt_a: tuple[int, int],
    pt_b: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.line(image, pt_a, pt_b, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.line(image, pt_a, pt_b, color, thickness, cv2.LINE_AA)


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


def build_camera_model(camera_dict: dict) -> CameraModel:
    width = int(camera_dict["width"])
    height = int(camera_dict["height"])
    fx = camera_dict.get("fx")
    fy = camera_dict.get("fy")
    cx = camera_dict.get("cx")
    cy = camera_dict.get("cy")

    if fy is None:
        fov_y = float(camera_dict["fov_y_deg"])
        fy = 0.5 * height / np.tan(np.deg2rad(fov_y) * 0.5)
    if fx is None:
        fx = float(fy)
    if cx is None:
        cx = 0.5 * (width - 1)
    if cy is None:
        cy = 0.5 * (height - 1)

    position = np.asarray(camera_dict["position"], dtype=np.float32)
    target = np.asarray(camera_dict["target"], dtype=np.float32)
    up = np.asarray(camera_dict["up"], dtype=np.float32)

    forward = normalize_vector(target - position)
    right = normalize_vector(np.cross(up, forward))
    camera_up = normalize_vector(np.cross(forward, right))
    down = -camera_up

    camera_to_world = np.stack([right, down, forward], axis=1).astype(np.float32)
    world_to_camera = camera_to_world.T

    return CameraModel(
        name=str(camera_dict["name"]),
        width=width,
        height=height,
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        position=position,
        world_to_camera=world_to_camera,
        camera_to_world=camera_to_world,
    )


def load_camera_models(dataset_root: Path) -> dict[str, CameraModel]:
    metadata_path = dataset_root / "metadata" / "cameras.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing camera metadata: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    cameras = payload.get("cameras", [])
    if not cameras:
        raise RuntimeError(f"No cameras found in metadata: {metadata_path}")

    return {cam["name"]: build_camera_model(cam) for cam in cameras}


def discover_processed_cameras(output_root: Path) -> list[str]:
    if not output_root.exists():
        raise FileNotFoundError(f"Output root does not exist: {output_root}")

    camera_names = []
    for cam_dir in sorted(output_root.iterdir()):
        if not cam_dir.is_dir():
            continue
        tracks_path = cam_dir / "cotracker" / "tracks.npz"
        raw_features_path = cam_dir / "dino" / "track_features.npz"
        if tracks_path.exists() and raw_features_path.exists():
            camera_names.append(cam_dir.name)

    if not camera_names:
        raise RuntimeError(
            f"No processed cameras found under {output_root}. "
            "Expected cotracker/tracks.npz and dino/track_features.npz for each camera."
        )

    return camera_names


def resolve_camera_pairs(
    camera_names: list[str],
    cam_a: Optional[str],
    cam_b: Optional[str],
) -> list[tuple[str, str]]:
    if (cam_a is None) != (cam_b is None):
        raise ValueError("Use --cam-a and --cam-b together, or omit both to process all pairs.")

    if cam_a is not None and cam_b is not None:
        if cam_a == cam_b:
            raise ValueError("Camera pair must contain two distinct camera names.")
        missing = [name for name in (cam_a, cam_b) if name not in camera_names]
        if missing:
            raise ValueError(f"Requested camera(s) not found in processed outputs: {', '.join(missing)}")
        return [(cam_a, cam_b)]

    return list(combinations(camera_names, 2))


def load_pair_inputs(
    output_root: Path,
    cam_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    track_path = output_root / cam_name / "cotracker" / "tracks.npz"
    feature_path = output_root / cam_name / "dino" / "track_features.npz"

    if not track_path.exists():
        raise FileNotFoundError(f"Missing CoTracker output for {cam_name}: {track_path}")
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Missing DINO track features for {cam_name}: {feature_path}. "
            "Run main.py without --skip-dino-features."
        )

    print(f"Using DINO features for {cam_name}: {feature_path}")

    with np.load(track_path) as track_data:
        tracks = np.asarray(track_data["tracks"], dtype=np.float32)
        visibilities = np.asarray(track_data["visibilities"]).astype(bool)

    with np.load(feature_path) as feature_data:
        features = np.asarray(feature_data["features"], dtype=np.float32)

    if tracks.shape[:2] != visibilities.shape:
        raise RuntimeError(
            f"Track/visibility shape mismatch for {cam_name}: "
            f"tracks={tracks.shape}, visibilities={visibilities.shape}"
        )
    if features.shape[:2] != tracks.shape[:2]:
        raise RuntimeError(
            f"Track/feature shape mismatch for {cam_name}: "
            f"tracks={tracks.shape}, features={features.shape}"
        )

    return tracks, visibilities, features


def aggregate_track_descriptors(features: np.ndarray, visibilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feature_norms = np.linalg.norm(features, axis=-1)
    valid = visibilities & np.isfinite(feature_norms) & (feature_norms > 1e-8)
    sums = np.where(valid[..., None], features, 0.0).sum(axis=0)
    counts = valid.sum(axis=0).astype(np.int32)

    desc_norms = np.linalg.norm(sums, axis=-1, keepdims=True)
    descriptors = np.zeros_like(sums, dtype=np.float32)
    mask = desc_norms[:, 0] > 1e-8
    descriptors[mask] = sums[mask] / desc_norms[mask]
    return descriptors.astype(np.float32), counts


def compute_common_feature_similarity(
    cache_a: dict[str, np.ndarray],
    cache_b: dict[str, np.ndarray],
    track_idx_a: int,
    track_idx_b: int,
    common_mask: np.ndarray,
) -> tuple[float, float]:
    feats_a = cache_a["features"][common_mask, track_idx_a]
    feats_b = cache_b["features"][common_mask, track_idx_b]
    if feats_a.shape[0] == 0:
        return float("-inf"), float("-inf")

    sims = np.sum(feats_a * feats_b, axis=-1)
    sims = sims[np.isfinite(sims)]
    if sims.size == 0:
        return float("-inf"), float("-inf")

    return float(np.median(sims)), float(np.mean(sims))


def sample_depth_at_points(depth_hw: np.ndarray, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if points_xy.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=bool)

    h, w = depth_hw.shape
    map_x = points_xy[:, 0].astype(np.float32).reshape(-1, 1)
    map_y = points_xy[:, 1].astype(np.float32).reshape(-1, 1)
    sampled = cv2.remap(
        depth_hw.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    ).reshape(-1)

    in_bounds = (
        (points_xy[:, 0] >= 0.0)
        & (points_xy[:, 0] <= float(w - 1))
        & (points_xy[:, 1] >= 0.0)
        & (points_xy[:, 1] <= float(h - 1))
    )
    valid = in_bounds & np.isfinite(sampled) & (sampled > 1e-6)
    return sampled.astype(np.float32), valid


def backproject_to_world(camera: CameraModel, points_xy: np.ndarray, depths: np.ndarray) -> np.ndarray:
    x = (points_xy[:, 0] - camera.cx) / camera.fx * depths
    y = (points_xy[:, 1] - camera.cy) / camera.fy * depths
    points_camera = np.stack([x, y, depths], axis=-1).astype(np.float32)
    return (points_camera @ camera.camera_to_world.T) + camera.position[None, :]


def project_world_to_image(camera: CameraModel, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = points_world - camera.position[None, :]
    points_camera = centered @ camera.world_to_camera.T
    z = points_camera[:, 2]
    valid = np.isfinite(z) & (z > 1e-6)

    pixels = np.full((points_world.shape[0], 2), np.nan, dtype=np.float32)
    if np.any(valid):
        pixels[valid, 0] = camera.fx * (points_camera[valid, 0] / z[valid]) + camera.cx
        pixels[valid, 1] = camera.fy * (points_camera[valid, 1] / z[valid]) + camera.cy
    return pixels, valid


def reconstruct_world_tracks(
    dataset_root: Path,
    camera: CameraModel,
    tracks: np.ndarray,
    visibilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    depth_dir = dataset_root / "depth_npy" / camera.name
    depth_paths = sorted(depth_dir.glob("frame_*.npy"))
    if len(depth_paths) < tracks.shape[0]:
        raise RuntimeError(
            f"Not enough depth frames for {camera.name}: need {tracks.shape[0]}, found {len(depth_paths)} in {depth_dir}"
        )

    num_frames, num_tracks = visibilities.shape
    world_tracks = np.full((num_frames, num_tracks, 3), np.nan, dtype=np.float32)
    world_valid = np.zeros((num_frames, num_tracks), dtype=bool)

    for frame_idx in range(num_frames):
        visible_idx = np.flatnonzero(visibilities[frame_idx])
        if visible_idx.size == 0:
            continue

        depth_hw = np.load(depth_paths[frame_idx]).astype(np.float32)
        points_xy = tracks[frame_idx, visible_idx]
        sampled_depth, depth_valid = sample_depth_at_points(depth_hw, points_xy)
        if not np.any(depth_valid):
            continue

        world_points = backproject_to_world(camera, points_xy[depth_valid], sampled_depth[depth_valid])
        valid_idx = visible_idx[depth_valid]
        world_tracks[frame_idx, valid_idx] = world_points
        world_valid[frame_idx, valid_idx] = True

    return world_tracks, world_valid


def median_reprojection_error(camera: CameraModel, points_world: np.ndarray, target_tracks: np.ndarray) -> float:
    projected_xy, valid = project_world_to_image(camera, points_world)
    if not np.any(valid):
        return float("inf")

    inside = (
        valid
        & np.isfinite(projected_xy[:, 0])
        & np.isfinite(projected_xy[:, 1])
        & (projected_xy[:, 0] >= 0.0)
        & (projected_xy[:, 0] <= float(camera.width - 1))
        & (projected_xy[:, 1] >= 0.0)
        & (projected_xy[:, 1] <= float(camera.height - 1))
    )
    if not np.any(inside):
        return float("inf")

    errors = np.linalg.norm(projected_xy[inside] - target_tracks[inside], axis=-1)
    return float(np.median(errors))


def select_topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros((0,), dtype=np.int32)
    k = max(1, min(int(k), values.size))
    if k == values.size:
        return np.argsort(-values)
    topk = np.argpartition(-values, k - 1)[:k]
    return topk[np.argsort(-values[topk])]


def build_camera_cache(
    dataset_root: Path,
    output_root: Path,
    camera: CameraModel,
) -> dict[str, np.ndarray]:
    tracks, visibilities, features = load_pair_inputs(
        output_root,
        camera.name,
    )
    descriptors, descriptor_counts = aggregate_track_descriptors(features, visibilities)
    world_tracks, world_valid = reconstruct_world_tracks(dataset_root, camera, tracks, visibilities)
    world_counts = world_valid.sum(axis=0).astype(np.int32)
    return {
        "tracks": tracks,
        "visibilities": visibilities,
        "features": features,
        "descriptors": descriptors,
        "descriptor_counts": descriptor_counts,
        "world_tracks": world_tracks,
        "world_valid": world_valid,
        "world_counts": world_counts,
    }


def evaluate_candidate(
    track_idx_a: int,
    track_idx_b: int,
    dino_similarity: float,
    cache_a: dict[str, np.ndarray],
    cache_b: dict[str, np.ndarray],
    camera_a: CameraModel,
    camera_b: CameraModel,
    args: argparse.Namespace,
) -> Optional[dict]:
    common = (
        cache_a["visibilities"][:, track_idx_a]
        & cache_b["visibilities"][:, track_idx_b]
        & cache_a["world_valid"][:, track_idx_a]
        & cache_b["world_valid"][:, track_idx_b]
    )
    common_frames = int(common.sum())
    if common_frames < args.min_common_frames:
        return None

    common_dino_similarity, mean_common_dino_similarity = compute_common_feature_similarity(
        cache_a=cache_a,
        cache_b=cache_b,
        track_idx_a=track_idx_a,
        track_idx_b=track_idx_b,
        common_mask=common,
    )
    if common_dino_similarity < args.min_common_dino_similarity:
        return None

    world_a = cache_a["world_tracks"][common, track_idx_a]
    world_b = cache_b["world_tracks"][common, track_idx_b]
    world_distances = np.linalg.norm(world_a - world_b, axis=-1)
    median_world_distance = float(np.median(world_distances))
    if median_world_distance > args.max_world_distance:
        return None

    reproj_ab = median_reprojection_error(camera_b, world_a, cache_b["tracks"][common, track_idx_b])
    reproj_ba = median_reprojection_error(camera_a, world_b, cache_a["tracks"][common, track_idx_a])
    if not np.isfinite(reproj_ab) or not np.isfinite(reproj_ba):
        return None

    median_reprojection = float(np.median(np.array([reproj_ab, reproj_ba], dtype=np.float32)))
    if median_reprojection > args.max_reprojection_error:
        return None

    score = (
        float(args.aggregate_dino_weight) * float(dino_similarity)
        + float(args.common_dino_weight) * common_dino_similarity
        - float(args.world_distance_weight) * median_world_distance
        - float(args.reprojection_weight) * (median_reprojection / max(args.max_reprojection_error, 1e-6))
    )

    return {
        "track_index_a": int(track_idx_a),
        "track_index_b": int(track_idx_b),
        "score": float(score),
        "dino_similarity": float(dino_similarity),
        "common_dino_similarity": common_dino_similarity,
        "mean_common_dino_similarity": mean_common_dino_similarity,
        "median_world_distance": median_world_distance,
        "median_reprojection_error": median_reprojection,
        "common_frames": common_frames,
        "descriptor_count_a": int(cache_a["descriptor_counts"][track_idx_a]),
        "descriptor_count_b": int(cache_b["descriptor_counts"][track_idx_b]),
    }


def match_camera_pair(
    camera_a: CameraModel,
    camera_b: CameraModel,
    cache_a: dict[str, np.ndarray],
    cache_b: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> list[dict]:
    eligible_a = np.flatnonzero(
        (cache_a["descriptor_counts"] >= args.min_track_visible_frames)
        & (cache_a["world_counts"] >= args.min_common_frames)
    )
    eligible_b = np.flatnonzero(
        (cache_b["descriptor_counts"] >= args.min_track_visible_frames)
        & (cache_b["world_counts"] >= args.min_common_frames)
    )

    if eligible_a.size == 0 or eligible_b.size == 0:
        return []

    desc_b = cache_b["descriptors"][eligible_b]
    desc_b_t = desc_b.T
    candidates: list[dict] = []

    for track_idx_a in eligible_a:
        descriptor_a = cache_a["descriptors"][track_idx_a]
        if not np.isfinite(descriptor_a).all():
            continue

        similarities = descriptor_a @ desc_b_t
        candidate_local_idx = np.flatnonzero(similarities >= args.min_dino_similarity)
        if candidate_local_idx.size == 0:
            continue

        ordered_local_idx = select_topk_indices(similarities[candidate_local_idx], args.top_k)
        for local_pos in ordered_local_idx:
            local_idx_b = candidate_local_idx[local_pos]
            track_idx_b = int(eligible_b[local_idx_b])
            candidate = evaluate_candidate(
                track_idx_a=int(track_idx_a),
                track_idx_b=track_idx_b,
                dino_similarity=float(similarities[local_idx_b]),
                cache_a=cache_a,
                cache_b=cache_b,
                camera_a=camera_a,
                camera_b=camera_b,
                args=args,
            )
            if candidate is not None:
                candidates.append(candidate)

    candidates.sort(key=lambda item: item["score"], reverse=True)

    used_a: set[int] = set()
    used_b: set[int] = set()
    accepted: list[dict] = []
    for candidate in candidates:
        track_idx_a = candidate["track_index_a"]
        track_idx_b = candidate["track_index_b"]
        if track_idx_a in used_a or track_idx_b in used_b:
            continue
        accepted.append(candidate)
        used_a.add(track_idx_a)
        used_b.add(track_idx_b)

    return accepted


def build_global_match_clusters(
    pair_match_records: list[dict],
    caches: dict[str, dict[str, np.ndarray]],
    min_cameras: int,
) -> list[dict]:
    if not pair_match_records:
        return []

    ordered_edges = sorted(pair_match_records, key=lambda item: item["score"], reverse=True)
    clusters: dict[int, dict] = {}
    node_to_cluster: dict[tuple[str, int], int] = {}
    next_cluster_id = 0

    def add_edge(cluster: dict, edge: dict) -> None:
        cluster["pair_edges"].append(edge)

    def create_cluster(edge: dict) -> int:
        nonlocal next_cluster_id
        cid = next_cluster_id
        next_cluster_id += 1
        node_a = (edge["camera_a"], edge["track_index_a"])
        node_b = (edge["camera_b"], edge["track_index_b"])
        members_by_camera = {
            edge["camera_a"]: int(edge["track_index_a"]),
            edge["camera_b"]: int(edge["track_index_b"]),
        }
        cluster = {
            "cluster_id": cid,
            "members_by_camera": members_by_camera,
            "pair_edges": [],
        }
        add_edge(cluster, edge)
        clusters[cid] = cluster
        node_to_cluster[node_a] = cid
        node_to_cluster[node_b] = cid
        return cid

    def attach_node(cluster: dict, node: tuple[str, int]) -> bool:
        camera_name, track_idx = node
        existing = cluster["members_by_camera"].get(camera_name)
        if existing is not None and existing != int(track_idx):
            return False
        cluster["members_by_camera"][camera_name] = int(track_idx)
        node_to_cluster[node] = int(cluster["cluster_id"])
        return True

    def merge_clusters(target_id: int, source_id: int, extra_edge: dict) -> bool:
        if target_id == source_id:
            add_edge(clusters[target_id], extra_edge)
            return True

        target = clusters[target_id]
        source = clusters[source_id]

        overlap = set(target["members_by_camera"]).intersection(source["members_by_camera"])
        for camera_name in overlap:
            if target["members_by_camera"][camera_name] != source["members_by_camera"][camera_name]:
                return False

        for camera_name, track_idx in source["members_by_camera"].items():
            target["members_by_camera"][camera_name] = int(track_idx)
            node_to_cluster[(camera_name, int(track_idx))] = target_id

        target["pair_edges"].extend(source["pair_edges"])
        add_edge(target, extra_edge)
        del clusters[source_id]
        return True

    for edge in ordered_edges:
        node_a = (edge["camera_a"], int(edge["track_index_a"]))
        node_b = (edge["camera_b"], int(edge["track_index_b"]))
        cluster_id_a = node_to_cluster.get(node_a)
        cluster_id_b = node_to_cluster.get(node_b)

        if cluster_id_a is None and cluster_id_b is None:
            create_cluster(edge)
            continue

        if cluster_id_a is not None and cluster_id_b is None:
            cluster = clusters[cluster_id_a]
            if attach_node(cluster, node_b):
                add_edge(cluster, edge)
            continue

        if cluster_id_a is None and cluster_id_b is not None:
            cluster = clusters[cluster_id_b]
            if attach_node(cluster, node_a):
                add_edge(cluster, edge)
            continue

        assert cluster_id_a is not None and cluster_id_b is not None
        if cluster_id_a == cluster_id_b:
            add_edge(clusters[cluster_id_a], edge)
            continue

        size_a = len(clusters[cluster_id_a]["members_by_camera"])
        size_b = len(clusters[cluster_id_b]["members_by_camera"])
        target_id, source_id = (cluster_id_a, cluster_id_b) if size_a >= size_b else (cluster_id_b, cluster_id_a)
        merge_clusters(target_id, source_id, edge)

    global_clusters = []
    for cluster in clusters.values():
        members_by_camera = {camera_name: int(track_idx) for camera_name, track_idx in cluster["members_by_camera"].items()}
        if len(members_by_camera) < min_cameras:
            continue

        pair_edges = list(cluster["pair_edges"])
        edge_scores = np.array([edge["score"] for edge in pair_edges], dtype=np.float32)
        dino_scores = np.array([edge["dino_similarity"] for edge in pair_edges], dtype=np.float32)
        common_dino_scores = np.array([edge["common_dino_similarity"] for edge in pair_edges], dtype=np.float32)

        members = []
        for camera_name, track_idx in sorted(members_by_camera.items()):
            members.append(
                {
                    "camera": camera_name,
                    "track_index": int(track_idx),
                    "descriptor_count": int(caches[camera_name]["descriptor_counts"][track_idx]),
                    "world_count": int(caches[camera_name]["world_counts"][track_idx]),
                }
            )

        global_clusters.append(
            {
                "cluster_id": int(cluster["cluster_id"]),
                "num_cameras": len(members_by_camera),
                "num_pair_edges": len(pair_edges),
                "mean_pair_score": float(edge_scores.mean()) if edge_scores.size else 0.0,
                "max_pair_score": float(edge_scores.max()) if edge_scores.size else 0.0,
                "mean_pair_dino_similarity": float(dino_scores.mean()) if dino_scores.size else 0.0,
                "mean_common_dino_similarity": float(common_dino_scores.mean()) if common_dino_scores.size else 0.0,
                "members_by_camera": members_by_camera,
                "members": members,
                "pair_edges": pair_edges,
            }
        )

    global_clusters.sort(
        key=lambda item: (
            int(item["num_cameras"]),
            float(item["mean_pair_score"]),
            float(item["max_pair_score"]),
        ),
        reverse=True,
    )

    for new_cluster_id, cluster in enumerate(global_clusters):
        cluster["cluster_id"] = int(new_cluster_id)

    return global_clusters


def save_pair_matches(
    output_root: Path,
    camera_a: CameraModel,
    camera_b: CameraModel,
    matches: list[dict],
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    pair_dir = output_root / "pair_matches"
    pair_dir.mkdir(parents=True, exist_ok=True)

    stem = pair_output_stem(camera_a, camera_b)
    json_path = pair_dir / f"{stem}.json"
    npz_path = pair_dir / f"{stem}.npz"

    payload = {
        "camera_a": camera_a.name,
        "camera_b": camera_b.name,
        "num_matches": len(matches),
        "config": {
            "min_track_visible_frames": int(args.min_track_visible_frames),
            "min_common_frames": int(args.min_common_frames),
            "min_dino_similarity": float(args.min_dino_similarity),
            "min_common_dino_similarity": float(args.min_common_dino_similarity),
            "top_k": int(args.top_k),
            "max_world_distance": float(args.max_world_distance),
            "max_reprojection_error": float(args.max_reprojection_error),
            "aggregate_dino_weight": float(args.aggregate_dino_weight),
            "common_dino_weight": float(args.common_dino_weight),
            "world_distance_weight": float(args.world_distance_weight),
            "reprojection_weight": float(args.reprojection_weight),
        },
        "matches": matches,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if matches:
        np.savez_compressed(
            npz_path,
            track_indices_a=np.array([m["track_index_a"] for m in matches], dtype=np.int32),
            track_indices_b=np.array([m["track_index_b"] for m in matches], dtype=np.int32),
            scores=np.array([m["score"] for m in matches], dtype=np.float32),
            dino_similarities=np.array([m["dino_similarity"] for m in matches], dtype=np.float32),
            common_dino_similarities=np.array([m["common_dino_similarity"] for m in matches], dtype=np.float32),
            mean_common_dino_similarities=np.array([m["mean_common_dino_similarity"] for m in matches], dtype=np.float32),
            median_world_distances=np.array([m["median_world_distance"] for m in matches], dtype=np.float32),
            median_reprojection_errors=np.array([m["median_reprojection_error"] for m in matches], dtype=np.float32),
            common_frames=np.array([m["common_frames"] for m in matches], dtype=np.int32),
        )
    else:
        np.savez_compressed(
            npz_path,
            track_indices_a=np.zeros((0,), dtype=np.int32),
            track_indices_b=np.zeros((0,), dtype=np.int32),
            scores=np.zeros((0,), dtype=np.float32),
            dino_similarities=np.zeros((0,), dtype=np.float32),
            common_dino_similarities=np.zeros((0,), dtype=np.float32),
            mean_common_dino_similarities=np.zeros((0,), dtype=np.float32),
            median_world_distances=np.zeros((0,), dtype=np.float32),
            median_reprojection_errors=np.zeros((0,), dtype=np.float32),
            common_frames=np.zeros((0,), dtype=np.int32),
        )

    return json_path, npz_path


def save_global_matches(
    output_root: Path,
    clusters: list[dict],
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    global_dir = output_root / "global_matches"
    global_dir.mkdir(parents=True, exist_ok=True)

    json_path = global_dir / "global_matches.json"
    npz_path = global_dir / "global_matches.npz"

    payload = {
        "num_clusters": len(clusters),
        "config": {
            "global_min_cameras": int(args.global_min_cameras),
            "min_track_visible_frames": int(args.min_track_visible_frames),
            "min_common_frames": int(args.min_common_frames),
            "min_dino_similarity": float(args.min_dino_similarity),
            "min_common_dino_similarity": float(args.min_common_dino_similarity),
            "top_k": int(args.top_k),
            "max_world_distance": float(args.max_world_distance),
            "max_reprojection_error": float(args.max_reprojection_error),
        },
        "clusters": clusters,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    cluster_ids: list[int] = []
    cluster_sizes: list[int] = []
    cluster_mean_scores: list[float] = []
    member_cluster_ids: list[int] = []
    member_camera_names: list[str] = []
    member_track_indices: list[int] = []

    for cluster in clusters:
        cluster_ids.append(int(cluster["cluster_id"]))
        cluster_sizes.append(int(cluster["num_cameras"]))
        cluster_mean_scores.append(float(cluster["mean_pair_score"]))
        for member in cluster["members"]:
            member_cluster_ids.append(int(cluster["cluster_id"]))
            member_camera_names.append(str(member["camera"]))
            member_track_indices.append(int(member["track_index"]))

    np.savez_compressed(
        npz_path,
        cluster_ids=np.asarray(cluster_ids, dtype=np.int32),
        cluster_sizes=np.asarray(cluster_sizes, dtype=np.int32),
        cluster_mean_scores=np.asarray(cluster_mean_scores, dtype=np.float32),
        member_cluster_ids=np.asarray(member_cluster_ids, dtype=np.int32),
        member_camera_names=np.asarray(member_camera_names),
        member_track_indices=np.asarray(member_track_indices, dtype=np.int32),
    )

    return json_path, npz_path


def save_global_match_table(
    output_root: Path,
    camera_names: list[str],
    clusters: list[dict],
) -> tuple[Path, Path]:
    global_dir = output_root / "global_matches"
    global_dir.mkdir(parents=True, exist_ok=True)

    csv_path = global_dir / "global_match_table.csv"
    npz_path = global_dir / "global_match_table.npz"

    ordered_cameras = list(camera_names)
    num_clusters = len(clusters)
    num_cameras = len(ordered_cameras)
    track_index_matrix = np.full((num_clusters, num_cameras), -1, dtype=np.int32)
    cluster_ids = np.zeros((num_clusters,), dtype=np.int32)
    cluster_sizes = np.zeros((num_clusters,), dtype=np.int32)
    pair_edge_counts = np.zeros((num_clusters,), dtype=np.int32)
    mean_pair_scores = np.zeros((num_clusters,), dtype=np.float32)
    max_pair_scores = np.zeros((num_clusters,), dtype=np.float32)
    mean_pair_dino = np.zeros((num_clusters,), dtype=np.float32)
    mean_common_dino = np.zeros((num_clusters,), dtype=np.float32)

    for row_idx, cluster in enumerate(clusters):
        cluster_ids[row_idx] = int(cluster["cluster_id"])
        cluster_sizes[row_idx] = int(cluster["num_cameras"])
        pair_edge_counts[row_idx] = int(cluster["num_pair_edges"])
        mean_pair_scores[row_idx] = float(cluster["mean_pair_score"])
        max_pair_scores[row_idx] = float(cluster["max_pair_score"])
        mean_pair_dino[row_idx] = float(cluster["mean_pair_dino_similarity"])
        mean_common_dino[row_idx] = float(cluster["mean_common_dino_similarity"])
        members_by_camera = cluster["members_by_camera"]
        for col_idx, camera_name in enumerate(ordered_cameras):
            track_idx = members_by_camera.get(camera_name)
            if track_idx is not None:
                track_index_matrix[row_idx, col_idx] = int(track_idx)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "global_track_id",
                "num_cameras",
                "num_pair_edges",
                "mean_pair_score",
                "max_pair_score",
                "mean_pair_dino_similarity",
                "mean_common_dino_similarity",
                *ordered_cameras,
            ]
        )
        for row_idx in range(num_clusters):
            track_cells = []
            for col_idx in range(num_cameras):
                value = int(track_index_matrix[row_idx, col_idx])
                track_cells.append("" if value < 0 else str(value))
            writer.writerow(
                [
                    int(cluster_ids[row_idx]),
                    int(cluster_sizes[row_idx]),
                    int(pair_edge_counts[row_idx]),
                    float(mean_pair_scores[row_idx]),
                    float(max_pair_scores[row_idx]),
                    float(mean_pair_dino[row_idx]),
                    float(mean_common_dino[row_idx]),
                    *track_cells,
                ]
            )

    np.savez_compressed(
        npz_path,
        camera_names=np.asarray(ordered_cameras),
        cluster_ids=cluster_ids,
        cluster_sizes=cluster_sizes,
        pair_edge_counts=pair_edge_counts,
        mean_pair_scores=mean_pair_scores,
        max_pair_scores=max_pair_scores,
        mean_pair_dino_similarities=mean_pair_dino,
        mean_common_dino_similarities=mean_common_dino,
        track_index_matrix=track_index_matrix,
    )

    return csv_path, npz_path


def render_pair_match_video(
    dataset_root: Path,
    output_root: Path,
    camera_a: CameraModel,
    camera_b: CameraModel,
    cache_a: dict[str, np.ndarray],
    cache_b: dict[str, np.ndarray],
    matches: list[dict],
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    pair_dir = output_root / "pair_matches"
    pair_dir.mkdir(parents=True, exist_ok=True)

    stem = pair_output_stem(camera_a, camera_b)
    video_path = pair_dir / f"{stem}.mp4"
    preview_path = pair_dir / f"{stem}_last_frame.png"

    frame_paths_a = list_frame_paths(dataset_root / "rgb" / camera_a.name)
    frame_paths_b = list_frame_paths(dataset_root / "rgb" / camera_b.name)
    num_frames = min(
        len(frame_paths_a),
        len(frame_paths_b),
        cache_a["tracks"].shape[0],
        cache_b["tracks"].shape[0],
    )
    if num_frames <= 0:
        raise RuntimeError(f"No shared frames available for {camera_a.name} and {camera_b.name}")

    sample_a = cv2.imread(str(frame_paths_a[0]), cv2.IMREAD_COLOR)
    sample_b = cv2.imread(str(frame_paths_b[0]), cv2.IMREAD_COLOR)
    if sample_a is None or sample_b is None:
        raise RuntimeError(f"Failed reading sample RGB frames for {camera_a.name}/{camera_b.name}")

    h_a, w_a = sample_a.shape[:2]
    h_b, w_b = sample_b.shape[:2]
    title_h = 74
    gap = 36
    canvas_h = title_h + max(h_a, h_b)
    canvas_w = w_a + gap + w_b
    fps = max(1, int(args.video_fps))

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (canvas_w, canvas_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {video_path}")

    rendered_matches = matches[: max(0, int(args.max_visualized_matches))]
    label_count = max(0, int(args.label_top_matches))
    point_radius = max(1, int(args.point_radius))
    line_thickness = max(1, int(args.line_thickness))
    xoff_b = w_a + gap

    last_canvas = None
    for frame_idx in range(num_frames):
        frame_a = cv2.imread(str(frame_paths_a[frame_idx]), cv2.IMREAD_COLOR)
        frame_b = cv2.imread(str(frame_paths_b[frame_idx]), cv2.IMREAD_COLOR)
        if frame_a is None or frame_b is None:
            raise RuntimeError(
                f"Failed reading RGB frame {frame_idx} for {camera_a.name}/{camera_b.name}"
            )

        canvas = np.full((canvas_h, canvas_w, 3), 18, dtype=np.uint8)
        canvas[title_h:title_h + h_a, 0:w_a] = frame_a
        canvas[title_h:title_h + h_b, xoff_b:xoff_b + w_b] = frame_b
        canvas[:, w_a:w_a + gap] = 30

        draw_text_with_outline(
            canvas,
            f"{camera_a.name}  <->  {camera_b.name}",
            (16, 28),
            color=(255, 255, 255),
            scale=0.8,
            thickness=2,
        )
        draw_text_with_outline(
            canvas,
            f"matches: {len(matches)} | showing: {len(rendered_matches)} | frame {frame_idx + 1}/{num_frames}",
            (16, 56),
            color=(210, 210, 210),
            scale=0.58,
            thickness=1,
        )
        draw_text_with_outline(canvas, camera_a.name, (16, title_h - 10), color=(180, 220, 255), scale=0.65, thickness=2)
        draw_text_with_outline(
            canvas,
            camera_b.name,
            (xoff_b + 16, title_h - 10),
            color=(180, 220, 255),
            scale=0.65,
            thickness=2,
        )

        for rank, match in enumerate(rendered_matches):
            track_idx_a = match["track_index_a"]
            track_idx_b = match["track_index_b"]
            color = make_match_color(rank)
            label = str(rank + 1) if rank < label_count else None

            visible_a = bool(cache_a["visibilities"][frame_idx, track_idx_a])
            visible_b = bool(cache_b["visibilities"][frame_idx, track_idx_b])

            pt_a = None
            pt_b = None
            if visible_a:
                xy_a = cache_a["tracks"][frame_idx, track_idx_a]
                if np.isfinite(xy_a).all():
                    pt_a = (int(round(float(xy_a[0]))), title_h + int(round(float(xy_a[1]))))
            if visible_b:
                xy_b = cache_b["tracks"][frame_idx, track_idx_b]
                if np.isfinite(xy_b).all():
                    pt_b = (xoff_b + int(round(float(xy_b[0]))), title_h + int(round(float(xy_b[1]))))

            if pt_a is not None and pt_b is not None:
                draw_match_line(canvas, pt_a, pt_b, color, line_thickness)
            if pt_a is not None:
                draw_match_marker(canvas, pt_a, color, point_radius, label=label)
            if pt_b is not None:
                draw_match_marker(canvas, pt_b, color, point_radius, label=label)

        writer.write(canvas)
        last_canvas = canvas

    writer.release()
    if last_canvas is not None:
        cv2.imwrite(str(preview_path), last_canvas)

    return video_path, preview_path


def render_global_match_video(
    dataset_root: Path,
    output_root: Path,
    camera_names: list[str],
    caches: dict[str, dict[str, np.ndarray]],
    clusters: list[dict],
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    global_dir = output_root / "global_matches"
    global_dir.mkdir(parents=True, exist_ok=True)

    video_path = global_dir / "all_cams_global_matches.mp4"
    preview_path = global_dir / "all_cams_global_matches_last_frame.png"

    if not camera_names:
        raise RuntimeError("camera_names cannot be empty for global match rendering.")

    frame_paths_by_camera = {camera_name: list_frame_paths(dataset_root / "rgb" / camera_name) for camera_name in camera_names}
    num_frames = min(
        min(len(frame_paths_by_camera[camera_name]), caches[camera_name]["tracks"].shape[0])
        for camera_name in camera_names
    )
    if num_frames <= 0:
        raise RuntimeError("No shared frames available for global match rendering.")

    sample_bgr = cv2.imread(str(frame_paths_by_camera[camera_names[0]][0]), cv2.IMREAD_COLOR)
    if sample_bgr is None:
        raise RuntimeError(f"Failed reading sample RGB frame for {camera_names[0]}")
    target_h, target_w = sample_bgr.shape[:2]

    sample_tile = make_labeled_tile(sample_bgr, camera_names[0])
    sample_grid = make_grid_frame(
        [sample_tile for _ in camera_names],
        columns=max(1, int(args.grid_columns)),
        frame_idx=0,
        num_frames=max(1, num_frames),
        title="Global Matches",
    )
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1, int(args.video_fps)),
        (sample_grid.shape[1], sample_grid.shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {video_path}")

    shown_clusters = clusters[: max(0, int(args.max_visualized_matches))]
    label_count = max(0, int(args.label_top_matches))
    point_radius = max(1, int(args.point_radius))
    last_canvas = None

    for frame_idx in range(num_frames):
        tiles = []
        for camera_name in camera_names:
            frame_bgr = cv2.imread(str(frame_paths_by_camera[camera_name][frame_idx]), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise RuntimeError(f"Failed reading RGB frame {frame_idx} for {camera_name}")

            frame_h, frame_w = frame_bgr.shape[:2]
            scale_x = float(target_w) / max(1, frame_w)
            scale_y = float(target_h) / max(1, frame_h)
            if frame_h != target_h or frame_w != target_w:
                frame_bgr = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            for rank, cluster in enumerate(shown_clusters):
                track_idx = cluster["members_by_camera"].get(camera_name)
                if track_idx is None:
                    continue

                if not bool(caches[camera_name]["visibilities"][frame_idx, track_idx]):
                    continue

                xy = caches[camera_name]["tracks"][frame_idx, track_idx]
                if not np.isfinite(xy).all():
                    continue

                px = int(round(float(xy[0]) * scale_x))
                py = int(round(float(xy[1]) * scale_y))
                color = make_match_color(rank)
                label = str(rank + 1) if rank < label_count else None
                draw_match_marker(frame_bgr, (px, py), color, point_radius, label=label)

            tiles.append(make_labeled_tile(frame_bgr, camera_name))

        canvas = make_grid_frame(
            tiles,
            columns=max(1, int(args.grid_columns)),
            frame_idx=frame_idx,
            num_frames=num_frames,
            title=f"Global Matches | clusters: {len(clusters)} | showing: {len(shown_clusters)}",
        )
        writer.write(canvas)
        last_canvas = canvas

    writer.release()
    if last_canvas is not None:
        cv2.imwrite(str(preview_path), last_canvas)

    return video_path, preview_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Match CoTracker tracks between camera pairs using DINO descriptors and depth-based geometry."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("blade_103706"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/blade_103706"))
    parser.add_argument("--cam-a", type=str, default=None, help="First camera of the pair to match")
    parser.add_argument("--cam-b", type=str, default=None, help="Second camera of the pair to match")
    parser.add_argument("--min-track-visible-frames", type=int, default=6)
    parser.add_argument("--min-common-frames", type=int, default=6)
    parser.add_argument("--min-dino-similarity", type=float, default=0.15)
    parser.add_argument(
        "--min-common-dino-similarity",
        type=float,
        default=0.10,
        help="Minimum median DINO cosine similarity across frames where both tracks are visible",
    )
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--max-world-distance", type=float, default=0.05)
    parser.add_argument("--max-reprojection-error", type=float, default=12.0)
    parser.add_argument("--aggregate-dino-weight", type=float, default=0.35)
    parser.add_argument("--common-dino-weight", type=float, default=1.0)
    parser.add_argument("--world-distance-weight", type=float, default=2.0)
    parser.add_argument("--reprojection-weight", type=float, default=0.25)
    parser.add_argument(
        "--skip-global-matching",
        action="store_true",
        help="Skip the final multiview aggregation step that merges pairwise matches into global camera clusters",
    )
    parser.add_argument(
        "--global-min-cameras",
        type=int,
        default=2,
        help="Keep only global match clusters that contain at least this many distinct cameras",
    )
    parser.add_argument("--skip-video", action="store_true", help="Skip MP4 visualization export")
    parser.add_argument("--video-fps", type=int, default=12, help="FPS of the pair-match visualization video")
    parser.add_argument("--grid-columns", type=int, default=4, help="Columns used for the global multiview montage")
    parser.add_argument(
        "--max-visualized-matches",
        type=int,
        default=40,
        help="Render only the top-scoring matches/clusters to keep the videos readable",
    )
    parser.add_argument(
        "--label-top-matches",
        type=int,
        default=12,
        help="Draw numeric labels only for the first N rendered matches",
    )
    parser.add_argument("--point-radius", type=int, default=4, help="Radius of rendered match points")
    parser.add_argument("--line-thickness", type=int, default=2, help="Thickness of correspondence lines")
    args = parser.parse_args()

    processed_cameras = discover_processed_cameras(args.output_root)
    camera_models = load_camera_models(args.dataset_root)
    missing_metadata = [name for name in processed_cameras if name not in camera_models]
    if missing_metadata:
        raise RuntimeError(
            f"Cameras found in outputs but missing in metadata: {', '.join(sorted(missing_metadata))}"
        )

    camera_pairs = resolve_camera_pairs(processed_cameras, args.cam_a, args.cam_b)
    if not camera_pairs:
        raise RuntimeError("No camera pairs available for matching.")

    caches: dict[str, dict[str, np.ndarray]] = {}
    for camera_name in sorted({name for pair in camera_pairs for name in pair}):
        print(f"Loading cached inputs for {camera_name}...")
        caches[camera_name] = build_camera_cache(
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            camera=camera_models[camera_name],
        )

    print(f"Matching {len(camera_pairs)} camera pair(s)...")
    pair_match_records: list[dict] = []
    for cam_a_name, cam_b_name in camera_pairs:
        camera_a = camera_models[cam_a_name]
        camera_b = camera_models[cam_b_name]
        matches = match_camera_pair(
            camera_a=camera_a,
            camera_b=camera_b,
            cache_a=caches[cam_a_name],
            cache_b=caches[cam_b_name],
            args=args,
        )
        for match in matches:
            pair_match_records.append(
                {
                    "camera_a": cam_a_name,
                    "camera_b": cam_b_name,
                    "track_index_a": int(match["track_index_a"]),
                    "track_index_b": int(match["track_index_b"]),
                    "score": float(match["score"]),
                    "dino_similarity": float(match["dino_similarity"]),
                    "common_dino_similarity": float(match["common_dino_similarity"]),
                    "mean_common_dino_similarity": float(match["mean_common_dino_similarity"]),
                    "median_world_distance": float(match["median_world_distance"]),
                    "median_reprojection_error": float(match["median_reprojection_error"]),
                    "common_frames": int(match["common_frames"]),
                }
            )
        json_path, npz_path = save_pair_matches(args.output_root, camera_a, camera_b, matches, args)
        message = (
            f"{cam_a_name} <-> {cam_b_name}: {len(matches)} matches "
            f"saved to {json_path} and {npz_path}"
        )
        if not args.skip_video:
            video_path, preview_path = render_pair_match_video(
                dataset_root=args.dataset_root,
                output_root=args.output_root,
                camera_a=camera_a,
                camera_b=camera_b,
                cache_a=caches[cam_a_name],
                cache_b=caches[cam_b_name],
                matches=matches,
                args=args,
            )
            message += f" | video: {video_path} | preview: {preview_path}"
        print(message)

    if not args.skip_global_matching:
        global_clusters = build_global_match_clusters(
            pair_match_records=pair_match_records,
            caches=caches,
            min_cameras=args.global_min_cameras,
        )
        global_json_path, global_npz_path = save_global_matches(args.output_root, global_clusters, args)
        global_table_csv_path, global_table_npz_path = save_global_match_table(
            output_root=args.output_root,
            camera_names=sorted(caches.keys()),
            clusters=global_clusters,
        )
        global_message = (
            f"Global multiview matches: {len(global_clusters)} clusters "
            f"saved to {global_json_path}, {global_npz_path}, "
            f"{global_table_csv_path} and {global_table_npz_path}"
        )
        if not args.skip_video and global_clusters:
            global_video_path, global_preview_path = render_global_match_video(
                dataset_root=args.dataset_root,
                output_root=args.output_root,
                camera_names=sorted(caches.keys()),
                caches=caches,
                clusters=global_clusters,
                args=args,
            )
            global_message += f" | video: {global_video_path} | preview: {global_preview_path}"
        print(global_message)


if __name__ == "__main__":
    main()
