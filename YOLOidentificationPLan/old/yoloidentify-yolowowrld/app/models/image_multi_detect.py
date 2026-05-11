"""
模块名称：image_multi_detect
作用：封装 Hugging Face `abhi099k/image-multi-detect` ONNX 安全图片分类器。
使用方法：
  - classifier = ImageMultiDetectClassifier(model_path="weights/image-multi-detect/model.onnx", mapping=mapping)
  - prediction = classifier.predict(pil_image)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.models.base import CoarseClassifier, CoarsePrediction, CoarseTopKItem, TagDetection
from app.services.category_config import CategoryMapping


LABELS = ("nsfw", "violence", "weapon", "smoking", "alcohol", "drugs", "sensitive", "hate")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


@dataclass
class ImageMultiDetectClassifier(CoarseClassifier):
    model_path: str
    mapping: CategoryMapping
    topk: int = 5
    weapon_threshold: float = 0.5
    violence_threshold: float = 0.3
    hate_threshold: float = 0.5
    drugs_threshold: float = 0.3
    sensitive_as_drugs_threshold: float = 0.3
    nsfw_as_drugs_threshold: float = 0.4

    def __post_init__(self) -> None:
        import onnxruntime as ort

        model_path = Path(self.model_path).expanduser()
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parents[2] / model_path
        if not model_path.exists():
            raise FileNotFoundError(f"image-multi-detect model not found: {model_path}")

        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._topk_limit = max(1, min(int(self.topk), len(self.mapping.categories)))

    def predict(self, image: Any) -> CoarsePrediction:
        scores = self._predict_scores(image)
        category_scores = {rule.name: 0.0 for rule in self.mapping.categories}
        detections: list[TagDetection] = []

        self._add_if_hit(
            category_scores=category_scores,
            detections=detections,
            category_name="terrorism",
            raw_tag="image_multi_detect:weapon",
            score=scores["weapon"],
            threshold=self.weapon_threshold,
        )
        self._add_if_hit(
            category_scores=category_scores,
            detections=detections,
            category_name="terrorism",
            raw_tag="image_multi_detect:violence",
            score=scores["violence"],
            threshold=self.violence_threshold,
        )
        self._add_if_hit(
            category_scores=category_scores,
            detections=detections,
            category_name="nazi_symbol",
            raw_tag="image_multi_detect:hate",
            score=scores["hate"],
            threshold=self.hate_threshold,
        )
        self._add_if_hit(
            category_scores=category_scores,
            detections=detections,
            category_name="drugs",
            raw_tag="image_multi_detect:drugs",
            score=scores["drugs"],
            threshold=self.drugs_threshold,
        )
        self._add_if_hit(
            category_scores=category_scores,
            detections=detections,
            category_name="drugs",
            raw_tag="image_multi_detect:sensitive",
            score=scores["sensitive"],
            threshold=self.sensitive_as_drugs_threshold,
        )
        self._add_if_hit(
            category_scores=category_scores,
            detections=detections,
            category_name="drugs",
            raw_tag="image_multi_detect:nsfw",
            score=scores["nsfw"],
            threshold=self.nsfw_as_drugs_threshold,
        )

        return self._build_prediction(category_scores=category_scores, detections=detections)

    def _predict_scores(self, image: Any) -> dict[str, float]:
        tensor = self._preprocess(image)
        logits = self._session.run(None, {self._input_name: tensor})[0][0]
        probabilities = _sigmoid(logits)
        return {label: float(score) for label, score in zip(LABELS, probabilities)}

    def _preprocess(self, image: Any) -> np.ndarray:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB").resize((224, 224))
        arr = np.asarray(image).astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return arr.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)

    def _add_if_hit(
        self,
        category_scores: dict[str, float],
        detections: list[TagDetection],
        category_name: str,
        raw_tag: str,
        score: float,
        threshold: float,
    ) -> None:
        if category_name not in category_scores or score < threshold:
            return
        category = self.mapping.get(category_name)
        category_scores[category_name] = max(category_scores[category_name], float(score))
        detections.append(
            TagDetection(
                raw_tag=raw_tag,
                category_name=category_name,
                score=float(score),
                chain=category.chain,
                box=None,
            )
        )

    def _build_prediction(
        self,
        category_scores: dict[str, float],
        detections: list[TagDetection],
    ) -> CoarsePrediction:
        normal_rule = self.mapping.get(self.mapping.normal_category)
        ordered = self.mapping.ordered_categories
        labels = [
            item.name
            for item in ordered
            if item.name != self.mapping.normal_category and category_scores.get(item.name, 0.0) > 0.0
        ]

        if labels:
            category_ids = [self.mapping.get(label).id for label in labels]
            scores = [category_scores[label] for label in labels]
        else:
            labels = [self.mapping.normal_category]
            category_ids = [normal_rule.id]
            category_scores[self.mapping.normal_category] = 1.0
            scores = [1.0]

        ranked = sorted(
            ordered,
            key=lambda item: (-category_scores.get(item.name, 0.0), item.priority, item.id),
        )[: self._topk_limit]
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
