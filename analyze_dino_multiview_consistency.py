#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


EPS = 1e-8

RAW_FEATURE_FILE_CANDIDATES = (
    "track_features.npz",
    "refined_feats.npy",
    "features.npy",
    "dino_features.npy",
    "feats.npy",
)
ALIGNED_FEATURE_FILE_CANDIDATES = ("track_features_aligned.npz",)
NUMERIC_NPZ_KEYS = (
    "features",
    "feats",
    "dino_features",
    "descriptors",
    "embeddings",
    "track_features",
    "aligned_features",
    "track_features_aligned",
)
COMPARISON_COLUMNS = (
    "camera_pair",
    "raw_random",
    "raw_nn",
    "raw_true_match",
    "raw_true_minus_random",
    "aligned_random",
    "aligned_nn",
    "aligned_true_match",
    "aligned_true_minus_random",
    "improvement_true_match",
    "improvement_true_minus_random",
)


@dataclass(frozen=True)
class FeatureSetSpec:
    key: str
    label: str
    file_candidates: tuple[str, ...]


@dataclass
class CameraData:
    name: str
    features: np.ndarray  # (T, N, C), float32, L2-normalized
    visible: np.ndarray  # (T, N), bool
    source_path: Path
    source_key: Optional[str]
    used_cotracker_visibility: bool


@dataclass
class PairMatches:
    track_indices_a: np.ndarray
    track_indices_b: np.ndarray
    sources: tuple[str, ...]


@dataclass
class PairMetrics:
    cam_a: str
    cam_b: str
    mean_random: float
    std_random: float
    mean_nn: float
    std_nn: float
    mean_true_match: float
    std_true_match: float
    separation_nn_random: float
    separation_true_random: float
    num_samples: int
    has_true_matches: bool
    true_match_sources: tuple[str, ...]


FEATURE_SET_SPECS = (
    FeatureSetSpec(key="raw", label="RAW", file_candidates=RAW_FEATURE_FILE_CANDIDATES),
    FeatureSetSpec(key="aligned", label="ALIGNED", file_candidates=ALIGNED_FEATURE_FILE_CANDIDATES),
)


def default_object_dir() -> Path:
    preferred = Path("outputs/blade_103706")
    if preferred.exists():
        return preferred
    return Path("outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare RAW and ALIGNED DINO feature consistency across multiple camera views."
    )
    parser.add_argument(
        "--object-dir",
        type=Path,
        default=default_object_dir(),
        help="Object output directory containing cam_XXX/, pair_matches/, global_matches/, etc.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of synchronized frames per pair to evaluate. Default: all available.",
    )
    parser.add_argument(
        "--samples-per-frame",
        type=int,
        default=1000,
        help="Visible feature samples drawn per frame for random and NN comparisons.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--nn-chunk-size",
        type=int,
        default=4096,
        help="Target feature chunk size used for nearest-neighbor similarity.",
    )
    args = parser.parse_args()

    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive.")
    if args.samples_per_frame <= 0:
        raise ValueError("--samples-per-frame must be positive.")
    if args.nn_chunk_size <= 0:
        raise ValueError("--nn-chunk-size must be positive.")
    return args


def discover_camera_dirs(object_dir: Path) -> list[Path]:
    if not object_dir.exists():
        raise FileNotFoundError(f"Object directory does not exist: {object_dir}")

    camera_dirs = sorted(
        path
        for path in object_dir.iterdir()
        if path.is_dir() and path.name.startswith("cam_") and (path / "dino").is_dir()
    )
    if len(camera_dirs) < 2:
        raise RuntimeError(
            f"Need at least two camera folders with a dino/ subdirectory under {object_dir}"
        )
    return camera_dirs


def choose_feature_file(dino_dir: Path, feature_spec: FeatureSetSpec) -> Path:
    for name in feature_spec.file_candidates:
        candidate = dino_dir / name
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Missing {feature_spec.label} DINO feature file in {dino_dir}. "
        f"Tried: {', '.join(feature_spec.file_candidates)}"
    )


def load_track_visibilities(cam_dir: Path) -> Optional[np.ndarray]:
    track_path = cam_dir / "cotracker" / "tracks.npz"
    if not track_path.is_file():
        return None

    with np.load(track_path) as data:
        if "visibilities" not in data:
            return None
        vis = np.asarray(data["visibilities"], dtype=bool)
    if vis.ndim != 2:
        return None
    return vis


def load_feature_array(path: Path) -> tuple[np.ndarray, Optional[str]]:
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32), None

    if path.suffix != ".npz":
        raise RuntimeError(f"Unsupported feature file format: {path}")

    best_key = None
    best_array = None
    best_size = -1

    with np.load(path) as data:
        for key in NUMERIC_NPZ_KEYS:
            if key in data:
                return np.asarray(data[key], dtype=np.float32), key

        for key in data.files:
            array = np.asarray(data[key])
            if array.ndim < 2 or not np.issubdtype(array.dtype, np.number):
                continue
            if array.size > best_size:
                best_key = key
                best_array = np.asarray(array, dtype=np.float32)
                best_size = int(array.size)

    if best_array is None:
        raise RuntimeError(f"Could not find a numeric feature array inside {path}")
    return best_array, best_key


def standardize_feature_array(
    raw: np.ndarray,
    expected_track_shape: Optional[tuple[int, int]],
) -> np.ndarray:
    array = np.asarray(raw, dtype=np.float32)

    if array.ndim == 2:
        return array[None, ...]

    if array.ndim == 3:
        if expected_track_shape is not None and array.shape[:2] == expected_track_shape:
            return array
        return array

    if array.ndim == 4:
        if array.shape[-1] >= 8:
            t, h, w, c = array.shape
            return array.reshape(t, h * w, c)
        if array.shape[1] >= 8:
            moved = np.moveaxis(array, 1, -1)
            t, h, w, c = moved.shape
            return moved.reshape(t, h * w, c)

    raise RuntimeError(
        f"Unsupported feature array shape {array.shape}. Expected (T, N, C), (N, C), or flattenable dense maps."
    )


def normalize_features(
    features: np.ndarray,
    visibility: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    feats = np.asarray(features, dtype=np.float32)
    if feats.ndim != 3:
        raise ValueError(f"Expected 3D features after standardization, got {feats.shape}")

    finite = np.isfinite(feats).all(axis=-1)
    norms = np.linalg.norm(np.where(np.isfinite(feats), feats, 0.0), axis=-1)
    inferred_visible = finite & (norms > EPS)

    if visibility is None:
        visible = inferred_visible
    else:
        if visibility.shape != feats.shape[:2]:
            raise RuntimeError(
                f"Visibility shape mismatch: features={feats.shape[:2]} visibility={visibility.shape}"
            )
        visible = np.asarray(visibility, dtype=bool) & inferred_visible

    out = np.zeros_like(feats, dtype=np.float32)
    if np.any(visible):
        out[visible] = feats[visible] / np.maximum(norms[visible, None], EPS)
    return out, visible


def load_camera_data(cam_dir: Path, feature_spec: FeatureSetSpec) -> CameraData:
    dino_dir = cam_dir / "dino"
    source_path = choose_feature_file(dino_dir, feature_spec)
    raw_features, source_key = load_feature_array(source_path)

    track_vis = load_track_visibilities(cam_dir)
    expected_track_shape = None if track_vis is None else tuple(track_vis.shape)
    features = standardize_feature_array(raw_features, expected_track_shape=expected_track_shape)

    used_track_vis = False
    visibility = None
    if track_vis is not None and track_vis.shape == features.shape[:2]:
        visibility = track_vis
        used_track_vis = True

    features, visible = normalize_features(features, visibility)
    return CameraData(
        name=cam_dir.name,
        features=features,
        visible=visible,
        source_path=source_path,
        source_key=source_key,
        used_cotracker_visibility=used_track_vis,
    )


def load_camera_set(camera_dirs: list[Path], feature_spec: FeatureSetSpec) -> list[CameraData]:
    return [load_camera_data(cam_dir, feature_spec) for cam_dir in camera_dirs]


def pair_key(cam_a: str, cam_b: str) -> tuple[str, str]:
    if cam_a == cam_b:
        raise ValueError("Camera pair requires two distinct cameras.")
    return (cam_a, cam_b) if cam_a < cam_b else (cam_b, cam_a)


def pair_identifier(cam_a: str, cam_b: str) -> str:
    return f"{cam_a}__{cam_b}"


def pair_label(cam_a: str, cam_b: str) -> str:
    return f"{cam_a} vs {cam_b}"


def add_pair_matches(
    store: dict[tuple[str, str], dict[str, object]],
    cam_a: str,
    cam_b: str,
    track_indices_a: np.ndarray,
    track_indices_b: np.ndarray,
    source: str,
    known_cameras: set[str],
) -> None:
    if cam_a not in known_cameras or cam_b not in known_cameras:
        return

    a = np.asarray(track_indices_a, dtype=np.int64).reshape(-1)
    b = np.asarray(track_indices_b, dtype=np.int64).reshape(-1)
    count = min(a.size, b.size)
    if count == 0:
        return

    a = a[:count]
    b = b[:count]
    valid = (a >= 0) & (b >= 0)
    if not np.any(valid):
        return

    a = a[valid].astype(np.int32, copy=False)
    b = b[valid].astype(np.int32, copy=False)

    key = pair_key(cam_a, cam_b)
    if key[0] != cam_a:
        a, b = b, a

    entry = store.setdefault(key, {"pairs": [], "sources": set()})
    pairs = np.stack([a, b], axis=1)
    entry["pairs"].append(pairs)
    entry["sources"].add(source)


def finalize_pair_matches(
    raw_store: dict[tuple[str, str], dict[str, object]]
) -> dict[tuple[str, str], PairMatches]:
    finalized: dict[tuple[str, str], PairMatches] = {}
    for key, entry in raw_store.items():
        chunks = entry["pairs"]
        if not chunks:
            continue
        merged = np.concatenate(chunks, axis=0).astype(np.int32, copy=False)
        unique_pairs = np.unique(merged, axis=0)
        finalized[key] = PairMatches(
            track_indices_a=unique_pairs[:, 0],
            track_indices_b=unique_pairs[:, 1],
            sources=tuple(sorted(str(source) for source in entry["sources"])),
        )
    return finalized


def load_pair_matches_from_pair_dir(
    object_dir: Path,
    known_cameras: set[str],
    store: dict[tuple[str, str], dict[str, object]],
) -> None:
    pair_dir = object_dir / "pair_matches"
    if not pair_dir.is_dir():
        return

    for path in sorted(pair_dir.glob("*.npz")):
        stem_parts = path.stem.split("__")
        if len(stem_parts) != 2:
            continue
        cam_a, cam_b = stem_parts
        with np.load(path) as data:
            if "track_indices_a" not in data or "track_indices_b" not in data:
                continue
            add_pair_matches(
                store=store,
                cam_a=cam_a,
                cam_b=cam_b,
                track_indices_a=data["track_indices_a"],
                track_indices_b=data["track_indices_b"],
                source=str(path.relative_to(object_dir)),
                known_cameras=known_cameras,
            )

    for path in sorted(pair_dir.glob("*.json")):
        stem_parts = path.stem.split("__")
        if len(stem_parts) != 2:
            continue
        cam_a, cam_b = stem_parts
        payload = json.loads(path.read_text(encoding="utf-8"))
        matches = payload.get("matches", [])
        if not matches:
            continue
        add_pair_matches(
            store=store,
            cam_a=cam_a,
            cam_b=cam_b,
            track_indices_a=np.array([m.get("track_index_a", -1) for m in matches], dtype=np.int32),
            track_indices_b=np.array([m.get("track_index_b", -1) for m in matches], dtype=np.int32),
            source=str(path.relative_to(object_dir)),
            known_cameras=known_cameras,
        )


def add_cluster_members_as_pairs(
    store: dict[tuple[str, str], dict[str, object]],
    members: list[tuple[str, int]],
    source: str,
    known_cameras: set[str],
) -> None:
    deduped = sorted(
        set((cam, int(track_idx)) for cam, track_idx in members if cam in known_cameras and track_idx >= 0)
    )
    for (cam_a, idx_a), (cam_b, idx_b) in combinations(deduped, 2):
        add_pair_matches(
            store=store,
            cam_a=cam_a,
            cam_b=cam_b,
            track_indices_a=np.array([idx_a], dtype=np.int32),
            track_indices_b=np.array([idx_b], dtype=np.int32),
            source=source,
            known_cameras=known_cameras,
        )


def load_global_match_table_npz(
    path: Path,
    object_dir: Path,
    known_cameras: set[str],
    store: dict[tuple[str, str], dict[str, object]],
) -> bool:
    with np.load(path) as data:
        if "camera_names" not in data or "track_index_matrix" not in data:
            return False
        camera_names = [str(name) for name in np.asarray(data["camera_names"]).tolist()]
        matrix = np.asarray(data["track_index_matrix"], dtype=np.int32)

    if matrix.ndim != 2 or len(camera_names) != matrix.shape[1]:
        return False

    source = str(path.relative_to(object_dir))
    for row in matrix:
        members = [
            (camera_name, int(track_idx))
            for camera_name, track_idx in zip(camera_names, row)
            if int(track_idx) >= 0
        ]
        add_cluster_members_as_pairs(store, members, source, known_cameras)
    return True


def load_global_match_table_csv(
    path: Path,
    object_dir: Path,
    known_cameras: set[str],
    store: dict[tuple[str, str], dict[str, object]],
) -> bool:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        camera_fields = [field for field in fieldnames if field in known_cameras]
        if not camera_fields:
            return False

        source = str(path.relative_to(object_dir))
        for row in reader:
            members = []
            for camera_name in camera_fields:
                value = (row.get(camera_name) or "").strip()
                if not value:
                    continue
                try:
                    members.append((camera_name, int(value)))
                except ValueError:
                    continue
            add_cluster_members_as_pairs(store, members, source, known_cameras)
    return True


def load_global_matches_npz(
    path: Path,
    object_dir: Path,
    known_cameras: set[str],
    store: dict[tuple[str, str], dict[str, object]],
) -> bool:
    with np.load(path) as data:
        required = {"member_cluster_ids", "member_camera_names", "member_track_indices"}
        if not required.issubset(data.files):
            return False
        cluster_ids = np.asarray(data["member_cluster_ids"], dtype=np.int32)
        camera_names = [str(name) for name in np.asarray(data["member_camera_names"]).tolist()]
        track_indices = np.asarray(data["member_track_indices"], dtype=np.int32)

    if not (cluster_ids.shape == track_indices.shape == (len(camera_names),)):
        return False

    cluster_to_members: dict[int, list[tuple[str, int]]] = {}
    for cluster_id, camera_name, track_idx in zip(
        cluster_ids.tolist(),
        camera_names,
        track_indices.tolist(),
    ):
        cluster_to_members.setdefault(int(cluster_id), []).append((camera_name, int(track_idx)))

    source = str(path.relative_to(object_dir))
    for members in cluster_to_members.values():
        add_cluster_members_as_pairs(store, members, source, known_cameras)
    return True


def load_global_matches_json(
    path: Path,
    object_dir: Path,
    known_cameras: set[str],
    store: dict[tuple[str, str], dict[str, object]],
) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    clusters = payload.get("clusters")
    if not isinstance(clusters, list):
        return False

    source = str(path.relative_to(object_dir))
    for cluster in clusters:
        members = []
        for member in cluster.get("members", []):
            camera_name = str(member.get("camera", ""))
            track_idx = member.get("track_index")
            if camera_name and track_idx is not None:
                try:
                    members.append((camera_name, int(track_idx)))
                except (TypeError, ValueError):
                    continue
        add_cluster_members_as_pairs(store, members, source, known_cameras)
    return True


def load_alignment_matches_npz(
    path: Path,
    object_dir: Path,
    known_cameras: set[str],
    store: dict[tuple[str, str], dict[str, object]],
) -> bool:
    with np.load(path) as data:
        keys = set(data.files)
        if {"camera_names", "track_index_matrix"}.issubset(keys):
            camera_names = [str(name) for name in np.asarray(data["camera_names"]).tolist()]
            matrix = np.asarray(data["track_index_matrix"], dtype=np.int32)
            if matrix.ndim == 2 and matrix.shape[1] == len(camera_names):
                source = str(path.relative_to(object_dir))
                for row in matrix:
                    members = [
                        (camera_name, int(track_idx))
                        for camera_name, track_idx in zip(camera_names, row)
                        if int(track_idx) >= 0
                    ]
                    add_cluster_members_as_pairs(store, members, source, known_cameras)
                return True
        if {"member_cluster_ids", "member_camera_names", "member_track_indices"}.issubset(keys):
            cluster_ids = np.asarray(data["member_cluster_ids"], dtype=np.int32)
            camera_names = [str(name) for name in np.asarray(data["member_camera_names"]).tolist()]
            track_indices = np.asarray(data["member_track_indices"], dtype=np.int32)
            if cluster_ids.shape == track_indices.shape == (len(camera_names),):
                source = str(path.relative_to(object_dir))
                cluster_to_members: dict[int, list[tuple[str, int]]] = {}
                for cluster_id, camera_name, track_idx in zip(
                    cluster_ids.tolist(),
                    camera_names,
                    track_indices.tolist(),
                ):
                    cluster_to_members.setdefault(int(cluster_id), []).append((camera_name, int(track_idx)))
                for members in cluster_to_members.values():
                    add_cluster_members_as_pairs(store, members, source, known_cameras)
                return True
    return False


def load_all_true_matches(
    object_dir: Path,
    camera_names: list[str],
) -> dict[tuple[str, str], PairMatches]:
    known_cameras = set(camera_names)
    store: dict[tuple[str, str], dict[str, object]] = {}

    load_pair_matches_from_pair_dir(object_dir, known_cameras, store)

    global_dir = object_dir / "global_matches"
    if global_dir.is_dir():
        table_npz = global_dir / "global_match_table.npz"
        if table_npz.is_file():
            load_global_match_table_npz(table_npz, object_dir, known_cameras, store)

        table_csv = global_dir / "global_match_table.csv"
        if table_csv.is_file():
            load_global_match_table_csv(table_csv, object_dir, known_cameras, store)

        global_npz = global_dir / "global_matches.npz"
        if global_npz.is_file():
            load_global_matches_npz(global_npz, object_dir, known_cameras, store)

        global_json = global_dir / "global_matches.json"
        if global_json.is_file():
            load_global_matches_json(global_json, object_dir, known_cameras, store)

    alignment_npz = object_dir / "multiview_dino_alignment.npz"
    if alignment_npz.is_file():
        load_alignment_matches_npz(alignment_npz, object_dir, known_cameras, store)

    return finalize_pair_matches(store)


def sample_without_replacement(
    rng: np.random.Generator,
    values: np.ndarray,
    count: int,
) -> np.ndarray:
    if count <= 0 or values.size == 0:
        return np.zeros((0,), dtype=np.int32)
    if values.size <= count:
        return values.astype(np.int32, copy=False)
    return rng.choice(values, size=count, replace=False).astype(np.int32, copy=False)


def best_match_similarities(
    query: np.ndarray,
    target: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    if query.size == 0 or target.size == 0:
        return np.zeros((0,), dtype=np.float32)

    query = np.asarray(query, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    best = np.full((query.shape[0],), -np.inf, dtype=np.float32)

    for start in range(0, target.shape[0], chunk_size):
        stop = min(start + chunk_size, target.shape[0])
        sims = query @ target[start:stop].T
        best = np.maximum(best, sims.max(axis=1))
    return best


def concat_or_empty(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def summarize_distribution(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values))


def safe_subtract(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return float(a - b)


def compute_pair_metrics(
    cam_a: CameraData,
    cam_b: CameraData,
    pair_matches: Optional[PairMatches],
    max_frames: Optional[int],
    samples_per_frame: int,
    nn_chunk_size: int,
    rng: np.random.Generator,
) -> PairMetrics:
    num_frames = min(cam_a.features.shape[0], cam_b.features.shape[0])
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    random_chunks: list[np.ndarray] = []
    nn_chunks: list[np.ndarray] = []
    true_chunks: list[np.ndarray] = []
    total_samples = 0

    match_indices_a = None
    match_indices_b = None
    has_track_matches = False
    true_match_sources: tuple[str, ...] = ()

    if pair_matches is not None:
        valid_pair_mask = (
            (pair_matches.track_indices_a >= 0)
            & (pair_matches.track_indices_a < cam_a.features.shape[1])
            & (pair_matches.track_indices_b >= 0)
            & (pair_matches.track_indices_b < cam_b.features.shape[1])
        )
        if np.any(valid_pair_mask):
            match_indices_a = pair_matches.track_indices_a[valid_pair_mask]
            match_indices_b = pair_matches.track_indices_b[valid_pair_mask]
            has_track_matches = True
            true_match_sources = pair_matches.sources

    for frame_idx in range(num_frames):
        visible_a = np.flatnonzero(cam_a.visible[frame_idx])
        visible_b = np.flatnonzero(cam_b.visible[frame_idx])
        sample_count = min(samples_per_frame, visible_a.size, visible_b.size)

        if sample_count > 0:
            sampled_a = sample_without_replacement(rng, visible_a, sample_count)
            sampled_b = sample_without_replacement(rng, visible_b, sample_count)

            feats_a = cam_a.features[frame_idx, sampled_a]
            feats_b = cam_b.features[frame_idx, sampled_b]
            random_chunks.append(np.sum(feats_a * feats_b, axis=-1))

            target_b = cam_b.features[frame_idx, visible_b]
            nn_chunks.append(best_match_similarities(feats_a, target_b, chunk_size=nn_chunk_size))
            total_samples += int(sample_count)

        if has_track_matches:
            common = cam_a.visible[frame_idx, match_indices_a] & cam_b.visible[frame_idx, match_indices_b]
            if np.any(common):
                idx = np.flatnonzero(common)
                if idx.size > samples_per_frame:
                    idx = sample_without_replacement(rng, idx.astype(np.int32, copy=False), samples_per_frame)
                feats_true_a = cam_a.features[frame_idx, match_indices_a[idx]]
                feats_true_b = cam_b.features[frame_idx, match_indices_b[idx]]
                true_chunks.append(np.sum(feats_true_a * feats_true_b, axis=-1))

    random_values = concat_or_empty(random_chunks)
    nn_values = concat_or_empty(nn_chunks)
    true_values = concat_or_empty(true_chunks)

    mean_random, std_random = summarize_distribution(random_values)
    mean_nn, std_nn = summarize_distribution(nn_values)
    mean_true, std_true = summarize_distribution(true_values)

    return PairMetrics(
        cam_a=cam_a.name,
        cam_b=cam_b.name,
        mean_random=mean_random,
        std_random=std_random,
        mean_nn=mean_nn,
        std_nn=std_nn,
        mean_true_match=mean_true,
        std_true_match=std_true,
        separation_nn_random=safe_subtract(mean_nn, mean_random),
        separation_true_random=safe_subtract(mean_true, mean_random),
        num_samples=int(total_samples),
        has_true_matches=bool(true_values.size > 0),
        true_match_sources=true_match_sources,
    )


def pair_seed(base_seed: int, pair_index: int) -> int:
    return int(np.uint64(base_seed) + np.uint64(pair_index) * np.uint64(10007))


def evaluate_feature_set(
    camera_names: list[str],
    camera_by_name: dict[str, CameraData],
    true_match_lookup: dict[tuple[str, str], PairMatches],
    args: argparse.Namespace,
) -> dict[tuple[str, str], PairMetrics]:
    results: dict[tuple[str, str], PairMetrics] = {}

    for pair_index, (cam_a_name, cam_b_name) in enumerate(combinations(camera_names, 2)):
        rng = np.random.default_rng(pair_seed(args.seed, pair_index))
        key = pair_key(cam_a_name, cam_b_name)
        results[key] = compute_pair_metrics(
            cam_a=camera_by_name[cam_a_name],
            cam_b=camera_by_name[cam_b_name],
            pair_matches=true_match_lookup.get(key),
            max_frames=args.max_frames,
            samples_per_frame=args.samples_per_frame,
            nn_chunk_size=args.nn_chunk_size,
            rng=rng,
        )

    return results


def build_comparison_rows(
    camera_names: list[str],
    raw_results: dict[tuple[str, str], PairMetrics],
    aligned_results: dict[tuple[str, str], PairMetrics],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for cam_a, cam_b in combinations(camera_names, 2):
        key = pair_key(cam_a, cam_b)
        raw = raw_results[key]
        aligned = aligned_results[key]
        rows.append(
            {
                "camera_pair": pair_identifier(cam_a, cam_b),
                "raw_random": raw.mean_random,
                "raw_nn": raw.mean_nn,
                "raw_true_match": raw.mean_true_match,
                "raw_true_minus_random": raw.separation_true_random,
                "aligned_random": aligned.mean_random,
                "aligned_nn": aligned.mean_nn,
                "aligned_true_match": aligned.mean_true_match,
                "aligned_true_minus_random": aligned.separation_true_random,
                "improvement_true_match": safe_subtract(aligned.mean_true_match, raw.mean_true_match),
                "improvement_true_minus_random": safe_subtract(
                    aligned.separation_true_random,
                    raw.separation_true_random,
                ),
            }
        )

    return rows


def write_comparison_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in COMPARISON_COLUMNS})


def format_value(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.4f}"


def save_raw_vs_aligned_plot(
    camera_names: list[str],
    raw_results: dict[tuple[str, str], PairMetrics],
    aligned_results: dict[tuple[str, str], PairMetrics],
    output_path: Path,
) -> None:
    ordered_pairs = [pair_key(cam_a, cam_b) for cam_a, cam_b in combinations(camera_names, 2)]
    labels = [pair_label(cam_a, cam_b) for cam_a, cam_b in ordered_pairs]
    x = np.arange(len(ordered_pairs), dtype=np.float32)

    raw_true = np.array([raw_results[key].mean_true_match for key in ordered_pairs], dtype=np.float32)
    raw_true_std = np.array([raw_results[key].std_true_match for key in ordered_pairs], dtype=np.float32)
    aligned_true = np.array([aligned_results[key].mean_true_match for key in ordered_pairs], dtype=np.float32)
    aligned_true_std = np.array([aligned_results[key].std_true_match for key in ordered_pairs], dtype=np.float32)
    improvement = aligned_true - raw_true

    fig_w = max(12.0, 0.75 * len(ordered_pairs) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_w, 6.8), constrained_layout=True)
    ax.set_facecolor("#fbfbfd")
    ax.grid(axis="y", color="#d9dee7", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    width = 0.34
    raw_valid = np.isfinite(raw_true)
    aligned_valid = np.isfinite(aligned_true)
    improvement_valid = np.isfinite(improvement)

    raw_handle = None
    aligned_handle = None
    if np.any(raw_valid):
        raw_handle = ax.bar(
            x[raw_valid] - width / 2.0,
            raw_true[raw_valid],
            width=width,
            yerr=raw_true_std[raw_valid],
            capsize=3,
            color="#9aa3ad",
            edgecolor="#58616b",
            linewidth=0.8,
            label="RAW True Match",
        )
    if np.any(aligned_valid):
        aligned_handle = ax.bar(
            x[aligned_valid] + width / 2.0,
            aligned_true[aligned_valid],
            width=width,
            yerr=aligned_true_std[aligned_valid],
            capsize=3,
            color="#2a9d8f",
            edgecolor="#1a665d",
            linewidth=0.8,
            label="ALIGNED True Match",
        )

    ax.axhline(0.0, color="#888888", linewidth=1.0, alpha=0.7)
    ax.set_title("RAW vs ALIGNED DINO True-Match Consistency")
    ax.set_xlabel("Camera Pair")
    ax.set_ylabel("True-Match Cosine Similarity")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    finite_true = np.concatenate([raw_true[np.isfinite(raw_true)], aligned_true[np.isfinite(aligned_true)]])
    if finite_true.size > 0:
        lower = min(-0.05, float(finite_true.min()) - 0.05)
        upper = min(1.05, max(0.15, float(finite_true.max()) + 0.10))
        ax.set_ylim(lower, upper)
    else:
        ax.set_ylim(-0.05, 1.05)
        ax.text(
            0.5,
            0.5,
            "No valid true-match correspondences found.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="#555555",
        )

    ax2 = ax.twinx()
    improvement_handle = None
    if np.any(improvement_valid):
        improvement_handle = ax2.bar(
            x[improvement_valid],
            improvement[improvement_valid],
            width=0.14,
            color="#e9c46a",
            edgecolor="#b78e2c",
            linewidth=0.8,
            alpha=0.45,
            label="Improvement",
        )
        imp_abs = float(np.max(np.abs(improvement[improvement_valid])))
        imp_lim = max(0.05, imp_abs + 0.02)
        ax2.set_ylim(-imp_lim, imp_lim)
    else:
        ax2.set_ylim(-0.05, 0.05)
    ax2.axhline(0.0, color="#b78e2c", linewidth=1.0, alpha=0.5)
    ax2.set_ylabel("Aligned - RAW True-Match Similarity")

    handles = []
    labels_out = []
    for handle in (raw_handle, aligned_handle, improvement_handle):
        if handle is not None:
            handles.append(handle)
            labels_out.append(handle.get_label())
    if handles:
        ax.legend(handles, labels_out, loc="upper right")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def nanmean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).any():
        return float("nan")
    return float(np.nanmean(array))


def print_camera_set_overview(feature_spec: FeatureSetSpec, camera_data: list[CameraData]) -> None:
    print(f"\n{feature_spec.label} feature sources:")
    for cam in camera_data:
        source = cam.source_path.name if cam.source_key is None else f"{cam.source_path.name}:{cam.source_key}"
        print(
            f"- {cam.name}: {source}  shape={cam.features.shape}  "
            f"track_visibility={'yes' if cam.used_cotracker_visibility else 'inferred'}"
        )


def print_pairwise_comparison(
    camera_names: list[str],
    raw_results: dict[tuple[str, str], PairMetrics],
    aligned_results: dict[tuple[str, str], PairMetrics],
) -> None:
    print("\nPer-pair comparison:")
    for cam_a, cam_b in combinations(camera_names, 2):
        key = pair_key(cam_a, cam_b)
        raw = raw_results[key]
        aligned = aligned_results[key]
        improvement = safe_subtract(aligned.mean_true_match, raw.mean_true_match)
        source_text = "none"
        if raw.has_true_matches:
            source_text = ", ".join(raw.true_match_sources)
        print(
            f"[{pair_label(cam_a, cam_b)}] "
            f"raw_true={format_value(raw.mean_true_match)}  "
            f"aligned_true={format_value(aligned.mean_true_match)}  "
            f"delta_true={format_value(improvement)}  "
            f"raw_nn={format_value(raw.mean_nn)}  "
            f"aligned_nn={format_value(aligned.mean_nn)}  "
            f"true_sources={source_text}"
        )


def print_alignment_interpretation(
    rows: list[dict[str, object]],
    raw_results: dict[tuple[str, str], PairMetrics],
    aligned_results: dict[tuple[str, str], PairMetrics],
) -> None:
    true_improvements = [
        float(row["improvement_true_match"])
        for row in rows
        if np.isfinite(float(row["improvement_true_match"]))
    ]
    true_gap_improvements = [
        float(row["improvement_true_minus_random"])
        for row in rows
        if np.isfinite(float(row["improvement_true_minus_random"]))
    ]
    ordered_keys = sorted(set(raw_results).intersection(aligned_results))
    nn_improvements = [
        safe_subtract(aligned_results[key].mean_nn, raw_results[key].mean_nn)
        for key in ordered_keys
        if np.isfinite(raw_results[key].mean_nn) and np.isfinite(aligned_results[key].mean_nn)
    ]

    mean_raw_true = nanmean_or_nan([raw.mean_true_match for raw in raw_results.values() if raw.has_true_matches])
    mean_aligned_true = nanmean_or_nan(
        [aligned.mean_true_match for aligned in aligned_results.values() if aligned.has_true_matches]
    )
    mean_true_improvement = nanmean_or_nan(true_improvements)
    mean_gap_improvement = nanmean_or_nan(true_gap_improvements)
    mean_nn_improvement = nanmean_or_nan(nn_improvements)

    print("\nInterpretation:")
    if not true_improvements:
        print("- No usable true correspondences were found, so RAW vs ALIGNED true-match consistency could not be measured directly.")
        if np.isfinite(mean_nn_improvement):
            print(f"- Mean NN change (ALIGNED - RAW): {mean_nn_improvement:.4f}")
        return

    print(
        f"- Mean RAW true-match similarity: {format_value(mean_raw_true)} | "
        f"Mean ALIGNED true-match similarity: {format_value(mean_aligned_true)}"
    )
    print(
        f"- Mean true-match improvement (ALIGNED - RAW): {format_value(mean_true_improvement)}"
    )
    print(
        f"- Mean improvement in true-minus-random separation: {format_value(mean_gap_improvement)}"
    )
    print(f"- Mean NN change (ALIGNED - RAW): {format_value(mean_nn_improvement)}")

    if np.isfinite(mean_true_improvement) and np.isfinite(mean_gap_improvement):
        if mean_true_improvement > 0.01 and mean_gap_improvement > 0.01:
            print("- Aligned features improve true-match consistency on average.")
        elif mean_true_improvement > 0.0 and mean_gap_improvement > 0.0:
            print("- Aligned features provide a small but positive improvement in true-match consistency.")
        else:
            print("- Aligned features do not improve true-match consistency on average.")

    if np.isfinite(mean_nn_improvement):
        if mean_nn_improvement > 0.01 and (not np.isfinite(mean_gap_improvement) or mean_gap_improvement <= 0.01):
            print("- Alignment seems to help nearest-neighbor search more than true physical correspondence.")
        elif mean_nn_improvement > 0.01 and np.isfinite(mean_gap_improvement) and mean_gap_improvement > 0.01:
            print("- Alignment helps both NN retrieval and true physical correspondence.")
        elif mean_nn_improvement <= 0.01 and np.isfinite(mean_gap_improvement) and mean_gap_improvement > 0.01:
            print("- Alignment mainly helps true physical correspondence rather than generic NN retrieval.")

    ranked = sorted(
        (
            (
                str(row["camera_pair"]),
                float(row["improvement_true_match"]),
                float(row["improvement_true_minus_random"]),
            )
            for row in rows
            if np.isfinite(float(row["improvement_true_match"]))
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    if ranked:
        if ranked[0][1] > 0.0:
            print("- Largest true-match improvements:")
            preview = ranked[:3]
        else:
            print("- No camera pair improved true-match similarity; least severe degradations:")
            preview = ranked[:3]
        for camera_pair, delta_true, delta_gap in preview:
            print(
                f"  {camera_pair}: "
                f"delta_true={delta_true:.4f}, "
                f"delta_true_minus_random={delta_gap:.4f}"
            )


def main() -> None:
    args = parse_args()
    object_dir = args.object_dir

    camera_dirs = discover_camera_dirs(object_dir)
    camera_names = [cam_dir.name for cam_dir in camera_dirs]
    print(f"Found {len(camera_names)} cameras under {object_dir}")

    raw_camera_data = load_camera_set(camera_dirs, FEATURE_SET_SPECS[0])
    aligned_camera_data = load_camera_set(camera_dirs, FEATURE_SET_SPECS[1])

    print_camera_set_overview(FEATURE_SET_SPECS[0], raw_camera_data)
    print_camera_set_overview(FEATURE_SET_SPECS[1], aligned_camera_data)

    true_match_lookup = load_all_true_matches(object_dir, camera_names)
    print(f"\nLoaded true-match candidates for {len(true_match_lookup)} camera pairs")

    raw_results = evaluate_feature_set(
        camera_names=camera_names,
        camera_by_name={cam.name: cam for cam in raw_camera_data},
        true_match_lookup=true_match_lookup,
        args=args,
    )
    aligned_results = evaluate_feature_set(
        camera_names=camera_names,
        camera_by_name={cam.name: cam for cam in aligned_camera_data},
        true_match_lookup=true_match_lookup,
        args=args,
    )

    comparison_rows = build_comparison_rows(camera_names, raw_results, aligned_results)
    print_pairwise_comparison(camera_names, raw_results, aligned_results)

    summary_path = object_dir / "dino_raw_vs_aligned_summary.csv"
    plot_path = object_dir / "dino_raw_vs_aligned_comparison.png"
    write_comparison_csv(comparison_rows, summary_path)
    save_raw_vs_aligned_plot(camera_names, raw_results, aligned_results, plot_path)

    print(f"\nSaved summary CSV: {summary_path}")
    print(f"Saved comparison plot: {plot_path}")
    print_alignment_interpretation(comparison_rows, raw_results, aligned_results)


if __name__ == "__main__":
    main()
