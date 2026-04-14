"""
YOLOWorld-backed multi-label coarse classifier.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from app.models.base import CoarseClassifier, CoarsePrediction, CoarseTopKItem


COARSE_LABEL_ORDER = ("human", "flag", "violence", "normal")
COARSE_LABEL_IDS = {name: index for index, name in enumerate(COARSE_LABEL_ORDER)}
WORLD_PROMPTS = ("person", "flag", "gun", "grenade", "knife")
WORLD_TO_COARSE = {
    "person": "human",
    "human": "human",
    "flag": "flag",
    "gun": "violence",
    "grenade": "violence",
    "knife": "violence",
}


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("-", " ").replace("_", " ")


def _resolve_name(names: Any, class_idx: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_idx, class_idx))
    if isinstance(names, (list, tuple)) and 0 <= class_idx < len(names):
        return str(names[class_idx])
    return str(class_idx)


def _to_coarse_label(raw_name: str, aliases: dict[str, str]) -> str | None:
    normalized = _normalize_name(aliases.get(raw_name, raw_name))
    if normalized in WORLD_TO_COARSE:
        return WORLD_TO_COARSE[normalized]
    if "person" in normalized or "human" in normalized:
        return "human"
    if "flag" in normalized:
        return "flag"
    if any(token in normalized for token in ("gun", "grenade", "knife")):
        return "violence"
    return None


@dataclass
class YoloV8CoarseClassifier(CoarseClassifier):
    weights_path: str
    device: str = "auto"
    topk: int = 4
    category_aliases: dict[str, str] | None = None
    label_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_det: int = 64
    imgsz: int = 640
    warmup: bool = True

    def __post_init__(self) -> None:
        try:
            from ultralytics import YOLOWorld  # type: ignore
        except ImportError:
            from ultralytics import YOLO as YOLOWorld  # type: ignore
        import torch

        weights = self.weights_path if self.weights_path and os.path.exists(self.weights_path) else "yolov8s-world.pt"
        self._model = YOLOWorld(weights)
        if not hasattr(self._model, "set_classes"):
            raise RuntimeError(
                "Loaded model does not expose set_classes(). Please use a YOLOWorld model, "
                "for example yolov8s-world.pt."
            )

        self._aliases = {str(k): str(v) for k, v in (self.category_aliases or {}).items()}
        self._world_prompts = list(WORLD_PROMPTS)
        self._model.set_classes(self._world_prompts)
        self._topk_limit = max(1, min(int(self.topk), len(COARSE_LABEL_ORDER)))
        self._torch = torch
        self._device = self._resolve_device(self.device)
        self._use_half = self._device != "cpu" and bool(torch.cuda.is_available())
        self._predict_kwargs = {
            "verbose": False,
            "device": self._device,
            "conf": float(self.label_threshold),
            "iou": float(self.iou_threshold),
            "max_det": int(self.max_det),
            "imgsz": int(self.imgsz),
            "half": self._use_half,
        }

        if self.warmup:
            self._warmup()

    def predict(self, image: Any) -> CoarsePrediction:
        results = self._model.predict(source=image, **self._predict_kwargs)
        if not results:
            raise RuntimeError("YOLOWorld returned empty results.")

        result = results[0]
        boxes = getattr(result, "boxes", None)
        coarse_scores = {label: 0.0 for label in COARSE_LABEL_ORDER}

        if boxes is not None and len(boxes) > 0:
            classes = _to_list(getattr(boxes, "cls", None))
            confidences = _to_list(getattr(boxes, "conf", None))
            names = getattr(result, "names", None) or getattr(self._model, "names", {}) or {}

            for index, class_idx in enumerate(classes):
                raw_name = _resolve_name(names, int(class_idx))
                coarse_label = _to_coarse_label(raw_name, self._aliases)
                if coarse_label is None:
                    continue
                confidence = float(confidences[index]) if index < len(confidences) else 0.0
                coarse_scores[coarse_label] = max(coarse_scores[coarse_label], confidence)

        labels = [label for label in COARSE_LABEL_ORDER[:-1] if coarse_scores[label] > 0.0]
        if labels:
            coarse_scores["normal"] = 0.0
            category_ids = [COARSE_LABEL_IDS[label] for label in labels]
            category_names = labels
            scores = [coarse_scores[label] for label in labels]
        else:
            coarse_scores["normal"] = 1.0
            category_ids = [COARSE_LABEL_IDS["normal"]]
            category_names = ["normal"]
            scores = [1.0]

        ranked_labels = sorted(
            COARSE_LABEL_ORDER,
            key=lambda label: (-coarse_scores[label], COARSE_LABEL_ORDER.index(label)),
        )[: self._topk_limit]
        topk_items = [
            CoarseTopKItem(
                category_id=COARSE_LABEL_IDS[label],
                category_name=label,
                score=float(coarse_scores[label]),
            )
            for label in ranked_labels
        ]

        return CoarsePrediction(
            category_id=category_ids,
            category_name=category_names,
            score=scores,
            topk=topk_items,
            labels=list(category_names),
        )

    def _resolve_device(self, requested_device: Any) -> str:
        requested = str(requested_device or "auto").strip().lower()
        if requested in {"", "auto"}:
            return "0" if self._torch.cuda.is_available() else "cpu"
        if requested != "cpu" and not self._torch.cuda.is_available():
            return "cpu"
        return str(requested_device)

    def _warmup(self) -> None:
        from PIL import Image

        dummy = Image.new("RGB", (int(self.imgsz), int(self.imgsz)), color=(0, 0, 0))
        self._model.predict(source=dummy, **self._predict_kwargs)
        if self._use_half:
            self._torch.cuda.synchronize()
