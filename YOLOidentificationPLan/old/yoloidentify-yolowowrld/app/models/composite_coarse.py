"""
模块名称：composite_coarse
作用：组合多个粗筛分类器，将 YOLOWorld 与安全图片分类器结果合并为统一粗筛输出。
使用方法：
  - classifier = CompositeCoarseClassifier(mapping=mapping, classifiers=[yolo, safety])
  - prediction = classifier.predict(pil_image)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.models.base import CoarseClassifier, CoarsePrediction, CoarseTopKItem, TagDetection
from app.services.category_config import CategoryMapping


@dataclass
class CompositeCoarseClassifier(CoarseClassifier):
    mapping: CategoryMapping
    classifiers: list[CoarseClassifier]
    topk: int = 5

    def __post_init__(self) -> None:
        if not self.classifiers:
            raise ValueError("CompositeCoarseClassifier requires at least one classifier.")
        self._topk_limit = max(1, min(int(self.topk), len(self.mapping.categories)))

    def predict(self, image: Any) -> CoarsePrediction:
        category_scores = {rule.name: 0.0 for rule in self.mapping.categories}
        detections: list[TagDetection] = []

        for classifier in self.classifiers:
            prediction = classifier.predict(image)
            for item in prediction.topk:
                if item.category_name == self.mapping.normal_category:
                    continue
                if item.category_name in category_scores:
                    category_scores[item.category_name] = max(
                        category_scores[item.category_name],
                        float(item.score),
                    )
            for detection in prediction.detections:
                if detection.category_name in category_scores:
                    detections.append(detection)
                    category_scores[detection.category_name] = max(
                        category_scores[detection.category_name],
                        float(detection.score),
                    )

        return self._build_prediction(category_scores=category_scores, detections=detections)

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
