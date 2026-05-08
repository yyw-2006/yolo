"""
模块名称：coarse_yolo_cls
作用：YOLOWorld 主干识别适配器，根据独立类别映射配置输出 10 类业务 tag。
使用方法：
  - classifier = YoloWorldClassifier(...)
  - prediction = classifier.predict(pil_image)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from app.models.base import CoarseClassifier, CoarsePrediction, CoarseTopKItem, TagDetection
from app.services.category_config import CategoryMapping


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


def _resolve_name(names: Any, class_idx: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_idx, class_idx))
    if isinstance(names, (list, tuple)) and 0 <= class_idx < len(names):
        return str(names[class_idx])
    return str(class_idx)


def _resolve_box(boxes: Any, index: int) -> tuple[float, float, float, float] | None:
    xyxy = getattr(boxes, "xyxy", None)
    if xyxy is None:
        return None
    values = _to_list(xyxy)
    if index >= len(values):
        return None
    box = values[index]
    if hasattr(box, "tolist"):
        box = box.tolist()
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))


def _build_prediction(
    mapping: CategoryMapping,
    category_scores: dict[str, float],
    detections: list[TagDetection],
    topk_limit: int,
) -> CoarsePrediction:
    normal_rule = mapping.get(mapping.normal_category)
    ordered = mapping.ordered_categories
    labels = [item.name for item in ordered if item.name != mapping.normal_category and category_scores.get(item.name, 0.0) > 0.0]

    if labels:
        category_ids = [mapping.get(label).id for label in labels]
        scores = [category_scores[label] for label in labels]
    else:
        labels = [mapping.normal_category]
        category_ids = [normal_rule.id]
        category_scores[mapping.normal_category] = 1.0
        scores = [1.0]

    ranked = sorted(
        ordered,
        key=lambda item: (-category_scores.get(item.name, 0.0), item.priority, item.id),
    )[:topk_limit]
    topk_items = [
        CoarseTopKItem(
            category_id=item.id,
            category_name=item.name,
            score=float(category_scores.get(item.name, 0.0)),
            display_name=item.display_name,
            chain=item.chain,
        )
        for item in ranked
    ]

    return CoarsePrediction(
        category_id=category_ids,
        category_name=labels,
        score=scores,
        topk=topk_items,
        labels=labels,
        detections=detections,
    )


@dataclass
class YoloWorldClassifier(CoarseClassifier):
    weights_path: str
    mapping: CategoryMapping
    device: str = "auto"
    topk: int = 4
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

        self._world_prompts = self.mapping.yolo_prompts
        if not self._world_prompts:
            raise RuntimeError("category mapping produced empty YOLOWorld prompts.")
        self._model.set_classes(self._world_prompts)
        self._topk_limit = max(1, min(int(self.topk), len(self.mapping.categories)))
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
        category_scores = {rule.name: 0.0 for rule in self.mapping.categories}
        detections: list[TagDetection] = []

        if boxes is not None and len(boxes) > 0:
            classes = _to_list(getattr(boxes, "cls", None))
            confidences = _to_list(getattr(boxes, "conf", None))
            names = getattr(result, "names", None) or getattr(self._model, "names", {}) or {}

            for index, class_idx in enumerate(classes):
                raw_name = _resolve_name(names, int(class_idx))
                confidence = float(confidences[index]) if index < len(confidences) else 0.0
                for match in self.mapping.match_tag(raw_name, confidence):
                    category_scores[match.category.name] = max(category_scores[match.category.name], confidence)
                    detections.append(
                        TagDetection(
                            raw_tag=raw_name,
                            category_name=match.category.name,
                            score=confidence,
                            chain=match.category.chain,
                            box=_resolve_box(boxes, index),
                        )
                    )

        return _build_prediction(
            mapping=self.mapping,
            category_scores=category_scores,
            detections=detections,
            topk_limit=self._topk_limit,
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


@dataclass
class RknnYoloWorldClassifier(CoarseClassifier):
    rknn_model_path: str
    text_model_path: str
    classes_file: str
    mapping: CategoryMapping
    topk: int = 5

    def __post_init__(self) -> None:
        missing = [
            path
            for path in (self.rknn_model_path, self.text_model_path, self.classes_file)
            if path and not os.path.exists(path)
        ]
        if missing:
            raise RuntimeError(
                "RKNN YOLOWorld backend requires converted model files. Missing: "
                + ", ".join(missing)
                + ". See docs/ocr_face_integration.md and weights/README.md."
            )
        try:
            import rknnlite.api  # noqa: F401
        except Exception as exc:
            raise RuntimeError("RKNN backend requires rknn-toolkit-lite2 on RK3588.") from exc

        raise RuntimeError(
            "RKNN YOLOWorld backend entry is configured, but project-specific RKNN postprocess is not bundled yet. "
            "Use AUDIT_INFERENCE_BACKEND=ultralytics for development or add the RKNN postprocess adapter here."
        )

    def predict(self, image: Any) -> CoarsePrediction:
        raise NotImplementedError("RKNN YOLOWorld backend is not initialized.")
