"""
模块名称：direct_judge
作用：直接 tag 类别的细分处理器，YOLOWorld 命中后直接返回业务类别结果。
使用方法：
  - DirectTagJudge(decision="block") 注册到暴恐、赌博等直接返回类别
"""

from __future__ import annotations

from app.models.base import CoarsePrediction, FineJudge, FinePrediction


class DirectTagJudge(FineJudge):
    def __init__(self, decision: str = "block") -> None:
        self._decision = decision

    def predict(self, image, coarse: CoarsePrediction) -> FinePrediction:
        detections = [
            {
                "raw_tag": item.raw_tag,
                "score": item.score,
                "box": list(item.box) if item.box else None,
            }
            for item in coarse.detections
            if item.category_name == coarse.primary_category_name
        ]
        score = max([item["score"] for item in detections], default=coarse.primary_score)
        return FinePrediction(
            implemented=True,
            category_name=coarse.primary_category_name,
            hit=True,
            label="tag_direct_hit",
            score=float(score),
            decision=self._decision,
            chain="direct",
            detail={
                "message": "Coarse classifier hit direct-return category.",
                "detections": detections,
            },
        )
