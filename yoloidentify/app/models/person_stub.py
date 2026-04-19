"""
Face recognition utilities based on RetinaFace detection and ArcFace embeddings.

The runtime entrypoint for the user requirement is `PersonStubJudge.predict`,
which returns only `list[str]`.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
from uniface import ArcFace, RetinaFace, set_cache_dir

from app.models.base import CoarsePrediction


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "celebrity_face_db" / "imdb-wiki-cropped-face-data"
DEFAULT_MODEL_CACHE_DIR = PROJECT_ROOT / "weights" / "uniface"
DEFAULT_FACEBANK_DIR = PROJECT_ROOT / "data" / "celebrity_face_db" / "facebank"
DEFAULT_FACEBANK_PATH = DEFAULT_FACEBANK_DIR / "person_facebank_full.npz"
DEFAULT_FACEBANK_MANIFEST_PATH = DEFAULT_FACEBANK_DIR / "person_facebank_full_manifest.json"
DEFAULT_MATCH_THRESHOLD = 0.45
DEFAULT_DETECTOR_INPUT_SIZE = (320, 320)
DEFAULT_PROVIDERS = ["CPUExecutionProvider"]


@dataclass(frozen=True)
class MetadataSource:
    source_name: str
    mat_path: Path
    record_key: str
    image_root: Path


@dataclass(frozen=True)
class CandidateImage:
    person_name: str
    image_path: Path
    source_name: str
    face_score: float
    second_face_score: float
    target_bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class FaceEmbedding:
    bbox: tuple[int, int, int, int]
    confidence: float
    embedding: np.ndarray


@dataclass(frozen=True)
class FaceBank:
    label_names: list[str]
    label_ids: np.ndarray
    embeddings: np.ndarray
    sample_paths: list[str]
    sample_sources: list[str]
    match_threshold: float

    @property
    def names(self) -> list[str]:
        return self.label_names

    @property
    def sample_count(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def identity_count(self) -> int:
        return len(self.label_names)

    def identity_counts(self) -> list[int]:
        if self.label_ids.size == 0:
            return [0 for _ in self.label_names]
        counts = np.bincount(self.label_ids, minlength=len(self.label_names))
        return [int(value) for value in counts.tolist()]

    def existing_path_set(self) -> set[str]:
        return {str(Path(path).resolve()) for path in self.sample_paths}

    def save(self, output_path: str | os.PathLike[str]) -> Path:
        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            version=np.int32(2),
            label_names=np.asarray(self.label_names, dtype=str),
            label_ids=np.asarray(self.label_ids, dtype=np.int32),
            embeddings=np.asarray(self.embeddings, dtype=np.float32),
            sample_paths=np.asarray(self.sample_paths, dtype=str),
            sample_sources=np.asarray(self.sample_sources, dtype=str),
            match_threshold=np.float32(self.match_threshold),
            names=np.asarray(self.label_names, dtype=str),  # compatibility helper for quick inspection
        )
        return target

    @classmethod
    def load(cls, facebank_path: str | os.PathLike[str]) -> "FaceBank":
        target = Path(facebank_path).resolve()
        payload = np.load(target, allow_pickle=False)

        if "label_names" in payload and "label_ids" in payload:
            label_names = [str(item) for item in payload["label_names"].tolist()]
            label_ids = np.asarray(payload["label_ids"], dtype=np.int32)
            embeddings = _normalize_matrix(np.asarray(payload["embeddings"], dtype=np.float32))
            sample_paths = [str(item) for item in payload.get("sample_paths", np.asarray([], dtype=str)).tolist()]
            sample_sources = [str(item) for item in payload.get("sample_sources", np.asarray([], dtype=str)).tolist()]
            threshold_array = payload["match_threshold"]
            match_threshold = float(threshold_array.reshape(-1)[0]) if threshold_array.size else DEFAULT_MATCH_THRESHOLD
            return cls(
                label_names=label_names,
                label_ids=label_ids,
                embeddings=embeddings,
                sample_paths=sample_paths,
                sample_sources=sample_sources,
                match_threshold=match_threshold,
            )

        # Backward-compatible path for the earlier prototype-only npz layout.
        legacy_names = [str(item) for item in payload["names"].tolist()]
        legacy_embeddings = _normalize_matrix(np.asarray(payload["embeddings"], dtype=np.float32))
        name_to_id: dict[str, int] = {}
        label_names: list[str] = []
        label_ids: list[int] = []
        for name in legacy_names:
            if name not in name_to_id:
                name_to_id[name] = len(label_names)
                label_names.append(name)
            label_ids.append(name_to_id[name])

        threshold_array = payload["match_threshold"]
        match_threshold = float(threshold_array.reshape(-1)[0]) if threshold_array.size else DEFAULT_MATCH_THRESHOLD
        sample_paths = [f"legacy://sample/{index}" for index in range(len(legacy_names))]
        sample_sources = ["legacy"] * len(legacy_names)
        return cls(
            label_names=label_names,
            label_ids=np.asarray(label_ids, dtype=np.int32),
            embeddings=legacy_embeddings,
            sample_paths=sample_paths,
            sample_sources=sample_sources,
            match_threshold=match_threshold,
        )

    def match(self, embedding: np.ndarray, threshold: float | None = None) -> tuple[str | None, float]:
        if self.embeddings.size == 0:
            return None, float("-inf")

        similarities = self.embeddings @ _normalize_vector(embedding)
        identity_scores = np.full((len(self.label_names),), -1.0, dtype=np.float32)
        np.maximum.at(identity_scores, self.label_ids, similarities)
        best_index = int(np.argmax(identity_scores))
        best_score = float(identity_scores[best_index])
        required = self.match_threshold if threshold is None else float(threshold)
        if best_score < required:
            return None, best_score
        return self.label_names[best_index], best_score


DEFAULT_METADATA_SOURCES: tuple[MetadataSource, ...] = (
    MetadataSource(
        source_name="imdb",
        mat_path=DATASET_ROOT / "imdb_crop_clean" / "imdb_crop" / "imdb.mat",
        record_key="imdb",
        image_root=DATASET_ROOT / "imdb_crop_clean" / "imdb_crop",
    ),
    MetadataSource(
        source_name="wiki",
        mat_path=DATASET_ROOT / "wiki_crop_clean" / "wiki_crop" / "wiki.mat",
        record_key="wiki",
        image_root=DATASET_ROOT / "wiki_crop_clean" / "wiki_crop",
    ),
)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return np.asarray(vector, dtype=np.float32)
    return (np.asarray(vector, dtype=np.float32) / norm).astype(np.float32, copy=False)


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    return (matrix / norms).astype(np.float32, copy=False)


def _unwrap_mat_value(value: Any) -> Any:
    current = value
    while isinstance(current, np.ndarray):
        if current.size == 0:
            return None
        current = current.flat[0]
    return current


def _extract_string(value: Any) -> str | None:
    current = _unwrap_mat_value(value)
    if current is None:
        return None
    text = str(current).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _extract_float(value: Any) -> float:
    current = _unwrap_mat_value(value)
    if current is None:
        return float("nan")
    try:
        return float(current)
    except (TypeError, ValueError):
        return float("nan")


def _extract_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.size < 4:
        return None
    flattened = array.reshape(-1)
    coords = tuple(float(item) for item in flattened[:4])
    if any(math.isnan(coord) for coord in coords):
        return None
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        return None
    return coords


def _metadata_sources_from_names(source_names: Sequence[str] | None = None) -> list[MetadataSource]:
    source_map = {source.source_name: source for source in DEFAULT_METADATA_SOURCES}
    if source_names is None:
        return list(DEFAULT_METADATA_SOURCES)
    if len(source_names) == 0:
        return []

    resolved: list[MetadataSource] = []
    for name in source_names:
        key = str(name).strip().lower()
        if key not in source_map:
            raise KeyError(f"Unknown metadata source: {name}")
        resolved.append(source_map[key])
    return resolved


def iter_metadata_candidates(
    source_names: Sequence[str] | None = None,
    *,
    allowed_names: Sequence[str] | None = None,
) -> Iterable[CandidateImage]:
    allowed = {str(name) for name in allowed_names} if allowed_names else None
    for source in _metadata_sources_from_names(source_names):
        if not source.mat_path.exists():
            continue

        record = loadmat(source.mat_path)[source.record_key][0, 0]
        total = record["full_path"].shape[1]
        for index in range(total):
            person_name = _extract_string(record["name"][0, index])
            relative_path = _extract_string(record["full_path"][0, index])
            if not person_name or not relative_path:
                continue
            if allowed is not None and person_name not in allowed:
                continue

            image_path = source.image_root / relative_path
            target_bbox = _extract_bbox(record["face_location"][0, index]) if "face_location" in record.dtype.names else None
            yield CandidateImage(
                person_name=person_name,
                image_path=image_path,
                source_name=source.source_name,
                face_score=_extract_float(record["face_score"][0, index]),
                second_face_score=_extract_float(record["second_face_score"][0, index]),
                target_bbox=target_bbox,
            )


def iter_labeled_directory_candidates(
    labeled_dirs: Sequence[str | os.PathLike[str]] | None,
    *,
    allowed_names: Sequence[str] | None = None,
) -> Iterable[CandidateImage]:
    if not labeled_dirs:
        return
    allowed = {str(name) for name in allowed_names} if allowed_names else None

    valid_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for root_dir in labeled_dirs:
        root = Path(root_dir).expanduser().resolve()
        if not root.exists():
            continue

        for person_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            person_name = person_dir.name.strip()
            if not person_name:
                continue
            if allowed is not None and person_name not in allowed:
                continue
            for image_path in sorted(person_dir.rglob("*")):
                if not image_path.is_file():
                    continue
                if image_path.suffix.lower() not in valid_suffixes:
                    continue
                yield CandidateImage(
                    person_name=person_name,
                    image_path=image_path.resolve(),
                    source_name=f"labeled_dir:{root}",
                    face_score=float("nan"),
                    second_face_score=float("nan"),
                    target_bbox=None,
                )


def _candidate_sort_key(candidate: CandidateImage) -> tuple[int, float, str]:
    single_face_bonus = 1 if math.isnan(candidate.second_face_score) else 0
    face_score = candidate.face_score if not math.isnan(candidate.face_score) else float("-inf")
    return (single_face_bonus, face_score, str(candidate.image_path))


def _group_candidates_by_name(
    *,
    source_names: Sequence[str] | None = None,
    labeled_dirs: Sequence[str | os.PathLike[str]] | None = None,
    allowed_names: Sequence[str] | None = None,
) -> dict[str, list[CandidateImage]]:
    grouped: dict[str, list[CandidateImage]] = defaultdict(list)
    for candidate in iter_metadata_candidates(source_names=source_names, allowed_names=allowed_names):
        grouped[candidate.person_name].append(candidate)
    for candidate in iter_labeled_directory_candidates(labeled_dirs=labeled_dirs, allowed_names=allowed_names):
        grouped[candidate.person_name].append(candidate)
    return grouped


def _load_bgr_image(image: Any) -> np.ndarray:
    if isinstance(image, (str, os.PathLike, Path)):
        image_path = Path(image).expanduser().resolve()
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        return frame

    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        Image = None

    if Image is not None and isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if isinstance(image, np.ndarray):
        frame = np.asarray(image).copy()
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if frame.ndim == 3 and frame.shape[2] == 3:
            return frame
        raise ValueError(f"Unsupported ndarray image shape: {frame.shape}")

    raise TypeError(f"Unsupported image input type: {type(image)!r}")


def _crop_face(image_bgr: np.ndarray, bbox: Sequence[float], landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]] | None:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_bgr.shape[1], x2)
    y2 = min(image_bgr.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = image_bgr[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None

    relative_landmarks = np.asarray(landmarks, dtype=np.float32).copy()
    relative_landmarks[:, 0] -= x1
    relative_landmarks[:, 1] -= y1
    return crop, relative_landmarks, (x1, y1, x2, y2)


def _bbox_area(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = [float(value) for value in left]
    rx1, ry1, rx2, ry2 = [float(value) for value in right]

    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0.0:
        return 0.0

    union = _bbox_area(left) + _bbox_area(right) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _extract_face_embeddings(
    image_bgr: np.ndarray,
    detector: RetinaFace,
    recognizer: ArcFace,
    *,
    max_faces: int = 0,
) -> list[FaceEmbedding]:
    faces = detector.detect(image_bgr, max_num=max_faces)
    detections: list[FaceEmbedding] = []
    for face in faces:
        cropped = _crop_face(image_bgr, face.bbox, face.landmarks)
        if cropped is None:
            continue
        crop_bgr, relative_landmarks, bbox = cropped
        embedding = recognizer.get_normalized_embedding(crop_bgr, relative_landmarks).astype(np.float32)
        detections.append(
            FaceEmbedding(
                bbox=bbox,
                confidence=float(face.confidence),
                embedding=embedding,
            )
        )
    detections.sort(key=lambda item: (item.bbox[0], item.bbox[1]))
    return detections


def _extract_candidate_embedding(
    candidate: CandidateImage,
    detector: RetinaFace,
    recognizer: ArcFace,
) -> FaceEmbedding | None:
    image_bgr = cv2.imread(str(candidate.image_path))
    if image_bgr is None:
        return None

    faces = detector.detect(image_bgr, max_num=0)
    if not faces:
        return None

    selected_face = None
    selected_score = None
    for face in faces:
        if candidate.target_bbox is not None:
            iou = _bbox_iou(face.bbox, candidate.target_bbox)
            score = (iou, float(face.confidence), _bbox_area(face.bbox))
        else:
            score = (_bbox_area(face.bbox), float(face.confidence))

        if selected_face is None or score > selected_score:
            selected_face = face
            selected_score = score

    if selected_face is None:
        return None

    cropped = _crop_face(image_bgr, selected_face.bbox, selected_face.landmarks)
    if cropped is None:
        return None

    crop_bgr, relative_landmarks, bbox = cropped
    embedding = recognizer.get_normalized_embedding(crop_bgr, relative_landmarks).astype(np.float32)
    return FaceEmbedding(
        bbox=bbox,
        confidence=float(selected_face.confidence),
        embedding=embedding,
    )


def _build_facebank_from_samples(
    *,
    label_names: list[str],
    sample_label_ids: list[int],
    sample_embeddings: list[np.ndarray],
    sample_paths: list[str],
    sample_sources: list[str],
    match_threshold: float,
) -> FaceBank:
    embeddings = np.stack(sample_embeddings, axis=0) if sample_embeddings else np.empty((0, 512), dtype=np.float32)
    return FaceBank(
        label_names=label_names,
        label_ids=np.asarray(sample_label_ids, dtype=np.int32),
        embeddings=_normalize_matrix(embeddings),
        sample_paths=[str(Path(path).resolve()) for path in sample_paths],
        sample_sources=list(sample_sources),
        match_threshold=float(match_threshold),
    )


def build_facebank(
    *,
    output_path: str | os.PathLike[str] = DEFAULT_FACEBANK_PATH,
    manifest_path: str | os.PathLike[str] = DEFAULT_FACEBANK_MANIFEST_PATH,
    source_names: Sequence[str] | None = ("imdb", "wiki"),
    labeled_dirs: Sequence[str | os.PathLike[str]] | None = None,
    preferred_names: Sequence[str] | None = None,
    limit_identities: int | None = None,
    max_samples_per_identity: int | None = None,
    holdout_per_identity: int = 0,
    min_samples_per_identity: int = 1,
    include_paths_in_manifest: bool = False,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    detector_input_size: tuple[int, int] = DEFAULT_DETECTOR_INPUT_SIZE,
    detector_confidence_threshold: float = 0.6,
    detector_nms_threshold: float = 0.4,
    providers: Sequence[str] | None = None,
    model_cache_dir: str | os.PathLike[str] = DEFAULT_MODEL_CACHE_DIR,
    verbose: bool = True,
) -> tuple[FaceBank, dict[str, Any]]:
    if holdout_per_identity < 0:
        raise ValueError("holdout_per_identity must be >= 0")
    if min_samples_per_identity <= 0:
        raise ValueError("min_samples_per_identity must be > 0")

    cache_dir = Path(model_cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    set_cache_dir(str(cache_dir))

    detector = RetinaFace(
        input_size=detector_input_size,
        confidence_threshold=detector_confidence_threshold,
        nms_threshold=detector_nms_threshold,
        providers=list(providers) if providers else list(DEFAULT_PROVIDERS),
    )
    recognizer = ArcFace(providers=list(providers) if providers else list(DEFAULT_PROVIDERS))

    grouped = _group_candidates_by_name(
        source_names=source_names,
        labeled_dirs=labeled_dirs,
        allowed_names=preferred_names,
    )
    ordered_candidates = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    if preferred_names:
        preferred_set = [str(name) for name in preferred_names]
        ordered_candidates = [(name, grouped[name]) for name in preferred_set if name in grouped]

    label_names: list[str] = []
    sample_label_ids: list[int] = []
    sample_embeddings: list[np.ndarray] = []
    sample_paths: list[str] = []
    sample_sources: list[str] = []
    manifest = {
        "facebank_path": str(Path(output_path).resolve()),
        "match_threshold": float(match_threshold),
        "holdout_per_identity": int(holdout_per_identity),
        "min_samples_per_identity": int(min_samples_per_identity),
        "sources": [source.source_name for source in _metadata_sources_from_names(source_names)]
        if source_names
        else [],
        "labeled_dirs": [str(Path(path).expanduser().resolve()) for path in (labeled_dirs or [])],
        "identities": [],
        "sample_count": 0,
        "identity_count": 0,
    }

    iterator = tqdm(
        ordered_candidates,
        desc="Building full facebank",
        unit="identity",
        disable=not verbose,
    )
    for person_name, candidates in iterator:
        ranked_candidates = sorted(candidates, key=_candidate_sort_key, reverse=True)
        if max_samples_per_identity is not None:
            ranked_candidates = ranked_candidates[: int(max_samples_per_identity)]

        successful: list[tuple[CandidateImage, FaceEmbedding]] = []
        for candidate in ranked_candidates:
            extracted = _extract_candidate_embedding(candidate, detector, recognizer)
            if extracted is None:
                continue
            successful.append((candidate, extracted))

        if len(successful) <= holdout_per_identity:
            continue

        train_samples = successful[: len(successful) - holdout_per_identity]
        holdout_samples = successful[len(successful) - holdout_per_identity :] if holdout_per_identity else []
        if len(train_samples) < min_samples_per_identity:
            continue

        label_id = len(label_names)
        label_names.append(person_name)
        for candidate, embedding in train_samples:
            sample_label_ids.append(label_id)
            sample_embeddings.append(embedding.embedding)
            sample_paths.append(str(candidate.image_path.resolve()))
            sample_sources.append(candidate.source_name)

        identity_entry: dict[str, Any] = {
            "name": person_name,
            "train_sample_count": len(train_samples),
            "holdout_sample_count": len(holdout_samples),
            "sources": sorted({candidate.source_name for candidate, _ in successful}),
        }
        if include_paths_in_manifest:
            identity_entry["train_paths"] = [str(candidate.image_path.resolve()) for candidate, _ in train_samples]
            identity_entry["holdout_paths"] = [str(candidate.image_path.resolve()) for candidate, _ in holdout_samples]
        manifest["identities"].append(identity_entry)

        if limit_identities is not None and len(label_names) >= int(limit_identities):
            break

    if not sample_embeddings:
        raise RuntimeError("No valid face embeddings were collected for the facebank.")

    facebank = _build_facebank_from_samples(
        label_names=label_names,
        sample_label_ids=sample_label_ids,
        sample_embeddings=sample_embeddings,
        sample_paths=sample_paths,
        sample_sources=sample_sources,
        match_threshold=match_threshold,
    )
    saved_facebank_path = facebank.save(output_path)
    manifest["facebank_path"] = str(saved_facebank_path)
    manifest["sample_count"] = facebank.sample_count
    manifest["identity_count"] = facebank.identity_count

    manifest_target = Path(manifest_path).resolve()
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    manifest_target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return facebank, manifest


def update_facebank(
    *,
    facebank_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
    manifest_path: str | os.PathLike[str] | None = None,
    source_names: Sequence[str] | None = None,
    labeled_dirs: Sequence[str | os.PathLike[str]] | None = None,
    limit_identities: int | None = None,
    max_samples_per_identity: int | None = None,
    min_new_samples_per_identity: int = 1,
    match_threshold: float | None = None,
    detector_input_size: tuple[int, int] = DEFAULT_DETECTOR_INPUT_SIZE,
    detector_confidence_threshold: float = 0.6,
    detector_nms_threshold: float = 0.4,
    providers: Sequence[str] | None = None,
    model_cache_dir: str | os.PathLike[str] = DEFAULT_MODEL_CACHE_DIR,
    verbose: bool = True,
) -> tuple[FaceBank, dict[str, Any]]:
    existing_bank = FaceBank.load(facebank_path)
    cache_dir = Path(model_cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    set_cache_dir(str(cache_dir))

    detector = RetinaFace(
        input_size=detector_input_size,
        confidence_threshold=detector_confidence_threshold,
        nms_threshold=detector_nms_threshold,
        providers=list(providers) if providers else list(DEFAULT_PROVIDERS),
    )
    recognizer = ArcFace(providers=list(providers) if providers else list(DEFAULT_PROVIDERS))

    grouped = _group_candidates_by_name(source_names=source_names if source_names is not None else (), labeled_dirs=labeled_dirs)
    existing_paths = existing_bank.existing_path_set()
    name_to_id = {name: index for index, name in enumerate(existing_bank.label_names)}

    label_names = list(existing_bank.label_names)
    sample_label_ids = existing_bank.label_ids.astype(np.int32).tolist()
    sample_embeddings = [embedding.astype(np.float32) for embedding in existing_bank.embeddings]
    sample_paths = list(existing_bank.sample_paths)
    sample_sources = list(existing_bank.sample_sources)

    added_identities: list[dict[str, Any]] = []
    iterator = tqdm(
        sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])),
        desc="Updating facebank",
        unit="identity",
        disable=not verbose,
    )
    for person_name, candidates in iterator:
        ranked_candidates = sorted(candidates, key=_candidate_sort_key, reverse=True)
        if max_samples_per_identity is not None:
            ranked_candidates = ranked_candidates[: int(max_samples_per_identity)]

        new_samples: list[tuple[CandidateImage, FaceEmbedding]] = []
        for candidate in ranked_candidates:
            resolved_path = str(candidate.image_path.resolve())
            if resolved_path in existing_paths:
                continue
            extracted = _extract_candidate_embedding(candidate, detector, recognizer)
            if extracted is None:
                continue
            new_samples.append((candidate, extracted))

        if len(new_samples) < min_new_samples_per_identity:
            continue

        if person_name not in name_to_id:
            name_to_id[person_name] = len(label_names)
            label_names.append(person_name)
        label_id = name_to_id[person_name]

        for candidate, embedding in new_samples:
            resolved_path = str(candidate.image_path.resolve())
            sample_label_ids.append(label_id)
            sample_embeddings.append(embedding.embedding)
            sample_paths.append(resolved_path)
            sample_sources.append(candidate.source_name)
            existing_paths.add(resolved_path)

        added_identities.append(
            {
                "name": person_name,
                "added_sample_count": len(new_samples),
                "sources": sorted({candidate.source_name for candidate, _ in new_samples}),
                "added_paths": [str(candidate.image_path.resolve()) for candidate, _ in new_samples],
            }
        )

        if limit_identities is not None and len(added_identities) >= int(limit_identities):
            break

    updated_bank = _build_facebank_from_samples(
        label_names=label_names,
        sample_label_ids=sample_label_ids,
        sample_embeddings=sample_embeddings,
        sample_paths=sample_paths,
        sample_sources=sample_sources,
        match_threshold=existing_bank.match_threshold if match_threshold is None else float(match_threshold),
    )

    output_target = Path(output_path).resolve() if output_path is not None else Path(facebank_path).resolve()
    updated_bank.save(output_target)

    summary = {
        "facebank_path": str(output_target),
        "sample_count": updated_bank.sample_count,
        "identity_count": updated_bank.identity_count,
        "added_identity_count": len(added_identities),
        "added_sample_count": int(sum(item["added_sample_count"] for item in added_identities)),
        "added_identities": added_identities,
        "sources": list(source_names or []),
        "labeled_dirs": [str(Path(path).expanduser().resolve()) for path in (labeled_dirs or [])],
    }
    if manifest_path is not None:
        manifest_target = Path(manifest_path).resolve()
        manifest_target.parent.mkdir(parents=True, exist_ok=True)
        manifest_target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return updated_bank, summary


def find_unknown_image(
    excluded_names: Sequence[str],
    *,
    source_names: Sequence[str] | None = ("imdb",),
    preferred_names: Sequence[str] = ("Fred Astaire", "Gregory Peck", "Lauren Bacall"),
) -> Path:
    excluded = {str(name) for name in excluded_names}
    ranked_candidates: dict[str, list[CandidateImage]] = defaultdict(list)
    for candidate in iter_metadata_candidates(source_names=source_names, allowed_names=preferred_names):
        if candidate.person_name in excluded:
            continue
        ranked_candidates[candidate.person_name].append(candidate)

    for person_name in preferred_names:
        if person_name in excluded or person_name not in ranked_candidates:
            continue
        chosen = max(ranked_candidates[person_name], key=_candidate_sort_key)
        return chosen.image_path.resolve()

    raise RuntimeError("Unable to find an unknown identity image outside the current facebank.")


class PersonStubJudge:
    def __init__(
        self,
        *,
        facebank_path: str | os.PathLike[str] = DEFAULT_FACEBANK_PATH,
        match_threshold: float | None = None,
        detector_input_size: tuple[int, int] = DEFAULT_DETECTOR_INPUT_SIZE,
        detector_confidence_threshold: float = 0.6,
        detector_nms_threshold: float = 0.4,
        providers: Sequence[str] | None = None,
        model_cache_dir: str | os.PathLike[str] = DEFAULT_MODEL_CACHE_DIR,
    ) -> None:
        self.facebank_path = Path(facebank_path).resolve()
        self.match_threshold = match_threshold
        self.detector_input_size = detector_input_size
        self.detector_confidence_threshold = detector_confidence_threshold
        self.detector_nms_threshold = detector_nms_threshold
        self.providers = list(providers) if providers else list(DEFAULT_PROVIDERS)
        self.model_cache_dir = Path(model_cache_dir).resolve()

        self._facebank: FaceBank | None = None
        self._detector: RetinaFace | None = None
        self._recognizer: ArcFace | None = None

    def _ensure_models(self) -> tuple[RetinaFace, ArcFace]:
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        set_cache_dir(str(self.model_cache_dir))

        if self._detector is None:
            self._detector = RetinaFace(
                input_size=self.detector_input_size,
                confidence_threshold=self.detector_confidence_threshold,
                nms_threshold=self.detector_nms_threshold,
                providers=self.providers,
            )
        if self._recognizer is None:
            self._recognizer = ArcFace(providers=self.providers)
        return self._detector, self._recognizer

    def _ensure_facebank(self) -> FaceBank:
        if self._facebank is None:
            if not self.facebank_path.exists():
                raise FileNotFoundError(
                    f"Facebank not found: {self.facebank_path}. "
                    "Build it first with build_facebank() or tests/build_person_facebank.py."
                )
            self._facebank = FaceBank.load(self.facebank_path)
        return self._facebank

    def recognize_faces(self, image: Any) -> list[str]:
        image_bgr = _load_bgr_image(image)
        facebank = self._ensure_facebank()
        detector, recognizer = self._ensure_models()
        threshold = self.match_threshold if self.match_threshold is not None else facebank.match_threshold

        detections = _extract_face_embeddings(image_bgr, detector, recognizer, max_faces=0)
        if not detections or facebank.embeddings.size == 0:
            return []

        recognized_names: list[str] = []
        for detected in detections:
            matched_name, _ = facebank.match(detected.embedding, threshold=threshold)
            if matched_name is not None:
                recognized_names.append(matched_name)
        return recognized_names

    def predict(self, image: Any, coarse: CoarsePrediction | None = None) -> list[str]:
        _ = coarse
        return self.recognize_faces(image)
