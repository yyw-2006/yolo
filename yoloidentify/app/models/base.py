"""
Shared model-layer contracts for coarse and fine prediction stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


@dataclass(frozen=True)
class CoarseTopKItem:
    category_id: int | None
    category_name: str
    score: float


@dataclass(frozen=True)
class CoarsePrediction:
    category_id: int | None | list[int | None]
    category_name: str | list[str]
    score: float | list[float]
    topk: list[CoarseTopKItem] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        labels = [str(item) for item in (self.labels or _ensure_list(self.category_name))]

        if isinstance(self.category_id, tuple):
            object.__setattr__(self, "category_id", list(self.category_id))
        if isinstance(self.category_name, tuple):
            object.__setattr__(self, "category_name", [str(item) for item in self.category_name])
        elif isinstance(self.category_name, list):
            object.__setattr__(self, "category_name", [str(item) for item in self.category_name])
        if isinstance(self.score, tuple):
            object.__setattr__(self, "score", [float(item) for item in self.score])
        elif isinstance(self.score, list):
            object.__setattr__(self, "score", [float(item) for item in self.score])
        object.__setattr__(self, "labels", labels)
        object.__setattr__(
            self,
            "topk",
            [
                item
                if isinstance(item, CoarseTopKItem)
                else CoarseTopKItem(
                    category_id=getattr(item, "category_id", None),
                    category_name=str(getattr(item, "category_name", "")),
                    score=float(getattr(item, "score", 0.0)),
                )
                for item in (self.topk or [])
            ],
        )

    @property
    def primary_category_id(self) -> int | None:
        category_ids = _ensure_list(self.category_id)
        return category_ids[0] if category_ids else None

    @property
    def primary_category_name(self) -> str:
        category_names = [str(item) for item in _ensure_list(self.category_name)]
        return category_names[0] if category_names else ""

    @property
    def primary_score(self) -> float:
        scores = [float(item) for item in _ensure_list(self.score)]
        return scores[0] if scores else 0.0


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
