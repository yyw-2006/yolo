"""
模块名称：base
作用：定义粗分类与细分判断的统一接口与通用数据结构，降低后续扩展（政治/游行等）的改动面。
使用方法：
  - 粗分类器实现 `CoarseClassifier.predict(image)`
  - 细分判断实现 `FineJudge.predict(image, coarse_result)`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class CoarseTopKItem:
    category_id: int | None
    category_name: str
    score: float


@dataclass(frozen=True)
class CoarsePrediction:
    category_id: int | None
    category_name: str
    score: float
    topk: list[CoarseTopKItem] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FinePrediction:
    implemented: bool
    category_name: str
    hit: bool
    label: str
    score: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)


class CoarseClassifier(Protocol):
    def predict(self, image: Any) -> CoarsePrediction:  # image: PIL.Image.Image
        raise NotImplementedError


class FineJudge(Protocol):
    def predict(self, image: Any, coarse: CoarsePrediction) -> FinePrediction:
        raise NotImplementedError

