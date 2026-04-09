"""
模块名称：coarse_yolo_cls
作用：封装 YOLOv8 分类模型作为粗分类器输出统一结构（category_id/category_name/score/topk）。
使用方法：
  - classifier = YoloV8CoarseClassifier(weights_path=..., device=..., topk=..., category_aliases=...)
  - pred = classifier.predict(pil_image)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ultralytics import YOLO

from app.models.base import CoarsePrediction, CoarseTopKItem, CoarseClassifier


@dataclass
class YoloV8CoarseClassifier(CoarseClassifier):
    weights_path: str
    device: str = "cpu"
    topk: int = 5
    category_aliases: dict[str, str] | None = None
    label_threshold: float = 0.3

    def __post_init__(self) -> None:
        weights = self.weights_path
        if not os.path.exists(weights):
            # 允许在没有自训权重的情况下用官方权重跑通链路（需要联网下载一次）
            weights = "yolov8n-cls.pt"

        self._model = YOLO(weights)
        self._aliases = self.category_aliases or {}

    def predict(self, image) -> CoarsePrediction:
        # Ultralytics 支持直接输入 PIL.Image
        results = self._model.predict(source=image, verbose=False, device=self.device)
        if not results:
            raise RuntimeError("Coarse classification returned empty results.")

        r0 = results[0]
        probs = getattr(r0, "probs", None)
        if probs is None:
            raise RuntimeError("Coarse classification result has no probs.")

        names: dict[int, str] = getattr(r0, "names", None) or getattr(self._model, "names", {}) or {}

        top1_idx = int(getattr(probs, "top1"))
        top1_conf = float(getattr(probs, "top1conf"))
        top5_idx = list(getattr(probs, "top5", []))[: self.topk]
        top5_conf = list(getattr(probs, "top5conf", []))[: self.topk]

        def to_category_name(class_idx: int) -> str:
            raw = str(names.get(int(class_idx), str(class_idx)))
            return self._aliases.get(raw, raw)

        category_name = to_category_name(top1_idx)
        topk_items: list[CoarseTopKItem] = []
        for i, idx in enumerate(top5_idx):
            score = float(top5_conf[i]) if i < len(top5_conf) else 0.0
            topk_items.append(
                CoarseTopKItem(category_id=int(idx), category_name=to_category_name(int(idx)), score=score)
            )

        labels = sorted(
            {item.category_name for item in topk_items if float(item.score) >= float(self.label_threshold)}
            | {category_name}
        )

        return CoarsePrediction(
            category_id=top1_idx,
            category_name=category_name,
            score=top1_conf,
            topk=topk_items,
            labels=labels,
        )

