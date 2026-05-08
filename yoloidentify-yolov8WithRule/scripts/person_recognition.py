"""
Module: person_recognition
Purpose: Run RetinaFace + ArcFace recognition against the local facebank.
Usage: PersonStubJudge().predict(image_path) returns face names, using "普通人" for unknown faces.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from uniface import ArcFace, RetinaFace, set_cache_dir


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FACEBANK_PATH = ROOT / "data" / "celebrity_face_db" / "facebank" / "person_facebank_full.npz"
DEFAULT_MODEL_CACHE_DIR = ROOT / "weights" / "uniface"
DEFAULT_MATCH_THRESHOLD = 0.45
DEFAULT_DETECTOR_INPUT_SIZE = (320, 320)
DEFAULT_PROVIDERS = ["CPUExecutionProvider"]
UNKNOWN_PERSON_NAME = "普通人"


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

    @classmethod
    def load(cls, facebank_path: str | os.PathLike[str]) -> "FaceBank":
        target = Path(facebank_path).expanduser().resolve()
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
        return cls(
            label_names=label_names,
            label_ids=np.asarray(label_ids, dtype=np.int32),
            embeddings=legacy_embeddings,
            sample_paths=[],
            sample_sources=[],
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


def _load_bgr_image(image: Any) -> np.ndarray:
    if isinstance(image, (str, os.PathLike, Path)):
        image_path = Path(image).expanduser().resolve()
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        return frame

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


def _crop_face(
    image_bgr: np.ndarray,
    bbox: Sequence[float],
    landmarks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]] | None:
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
        detections.append(FaceEmbedding(bbox=bbox, confidence=float(face.confidence), embedding=embedding))
    detections.sort(key=lambda item: (item.bbox[0], item.bbox[1]))
    return detections


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
        self.facebank_path = Path(facebank_path).expanduser().resolve()
        self.match_threshold = match_threshold
        self.detector_input_size = detector_input_size
        self.detector_confidence_threshold = detector_confidence_threshold
        self.detector_nms_threshold = detector_nms_threshold
        self.providers = list(providers) if providers else list(DEFAULT_PROVIDERS)
        self.model_cache_dir = Path(model_cache_dir).expanduser().resolve()

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
                raise FileNotFoundError(f"Facebank not found: {self.facebank_path}")
            self._facebank = FaceBank.load(self.facebank_path)
        return self._facebank

    def recognize_faces(self, image: Any) -> list[dict]:
        image_bgr = _load_bgr_image(image)
        facebank = self._ensure_facebank()
        detector, recognizer = self._ensure_models()
        threshold = self.match_threshold if self.match_threshold is not None else facebank.match_threshold

        detections = _extract_face_embeddings(image_bgr, detector, recognizer, max_faces=0)
        results = []
        for detected in detections:
            matched_name, score = facebank.match(detected.embedding, threshold=threshold)
            results.append(
                {
                    "name": matched_name or UNKNOWN_PERSON_NAME,
                    "matched": matched_name is not None,
                    "score": float(score),
                    "bbox": list(detected.bbox),
                    "confidence": detected.confidence,
                }
            )
        return results

    def predict(self, image: Any) -> list[str]:
        return [item["name"] for item in self.recognize_faces(image)]


class FaceRecognitionService:
    provider = "RetinaFace + ArcFace"

    def __init__(self, judge: PersonStubJudge | None = None) -> None:
        self.judge = judge or PersonStubJudge()

    def recognize(self, image_path: Path, face_detections: list[dict]) -> dict:
        identities = self.judge.recognize_faces(image_path)
        status = "ok"
        if not identities and face_detections:
            status = "no_retinaface_detection"
            identities = [{"name": UNKNOWN_PERSON_NAME, "matched": False} for _ in face_detections]

        return {
            "provider": self.provider,
            "status": status,
            "image": str(image_path.resolve()),
            "face_count": len(identities),
            "identities": identities,
        }
