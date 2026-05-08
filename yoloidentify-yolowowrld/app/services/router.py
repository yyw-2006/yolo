"""
模块名称：router
作用：根据粗分类结果决定是否进入细分判断，并从注册表选择对应的细分处理器。
使用方法：
  - next_stage, fine = route(image, coarse, registry, normal_category_name)
"""

from __future__ import annotations

from app.models.base import CoarsePrediction, FinePrediction
from app.services.judge_registry import JudgeRegistry


def route(
    image,
    coarse: CoarsePrediction,
    registry: JudgeRegistry,
    normal_category_name: str,
) -> tuple[str, list[FinePrediction]]:
    labels = list(dict.fromkeys(coarse.labels or [coarse.category_name]))
    labels = [x for x in labels if x and x != normal_category_name]

    if not labels:
        return "coarse_only", []

    fine_results: list[FinePrediction] = []
    for label in labels:
        topk_item = next((item for item in coarse.topk if item.category_name == label), None)
        label_detections = [item for item in coarse.detections if item.category_name == label]
        coarse_for_label = CoarsePrediction(
            category_id=topk_item.category_id if topk_item else coarse.category_id,
            category_name=label,
            score=topk_item.score if topk_item else coarse.score,
            topk=coarse.topk,
            labels=[label],
            detections=label_detections,
        )
        judge = registry.get(label)
        fine_results.append(judge.predict(image, coarse_for_label))

    chains = list(dict.fromkeys(item.chain or "unknown" for item in fine_results))
    return "+".join(chains) if chains else "fine_judge_multi", fine_results

