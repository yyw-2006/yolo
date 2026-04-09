"""
模块名称：pipeline
作用：审核主流程编排（图片 -> 粗分类 -> 分流 -> 细分接口 -> 统一结果）。
使用方法：
  - pipeline = AuditPipeline(...)
  - resp = pipeline.audit_image(pil_image)
"""

from __future__ import annotations

import uuid

from app.models.base import CoarseClassifier, CoarsePrediction, FinePrediction
from app.schemas import AuditResponse, CoarseResult, FineResult, TopKItem
from app.services.judge_registry import JudgeRegistry
from app.services.router import route


class AuditPipeline:
    def __init__(
        self,
        coarse: CoarseClassifier,
        registry: JudgeRegistry,
        normal_category_name: str = "normal",
    ) -> None:
        self._coarse = coarse
        self._registry = registry
        self._normal_category_name = normal_category_name

    def audit_image(self, image) -> AuditResponse:
        request_id = uuid.uuid4().hex

        coarse_pred = self._coarse.predict(image)
        next_stage, fine_preds = route(
            image=image,
            coarse=coarse_pred,
            registry=self._registry,
            normal_category_name=self._normal_category_name,
        )

        coarse_result = _to_coarse_result(coarse_pred)

        labels = list(dict.fromkeys(coarse_pred.labels or [coarse_pred.category_name]))
        non_normal_labels = [x for x in labels if x and x != self._normal_category_name]

        if not non_normal_labels:
            return AuditResponse(
                request_id=request_id,
                coarse_result=coarse_result,
                next_stage=next_stage,
                fine_result=None,
                fine_results=[],
                final_decision="safe",
            )

        fine_results = _to_fine_results(fine_preds, coarse_pred)

        # 第一版：细分模块占位，统一返回 review，便于后续替换真实模型后细化决策策略
        return AuditResponse(
            request_id=request_id,
            coarse_result=coarse_result,
            next_stage=next_stage,
            fine_result=fine_results[0] if fine_results else None,
            fine_results=fine_results,
            final_decision="review",
        )


def _to_coarse_result(pred: CoarsePrediction) -> CoarseResult:
    return CoarseResult(
        category_id=pred.category_id,
        category_name=pred.category_name,
        score=float(pred.score),
        topk=[
            TopKItem(category_id=item.category_id, category_name=item.category_name, score=float(item.score))
            for item in (pred.topk or [])
        ],
        labels=list(pred.labels or []),
    )


def _to_fine_results(fines: list[FinePrediction], coarse: CoarsePrediction) -> list[FineResult]:
    if not fines:
        return [
            FineResult(
                implemented=False,
                category_name=coarse.category_name,
                hit=False,
                label="not_implemented",
                score=None,
                detail={"message": "fine stage skipped"},
            )
        ]

    results: list[FineResult] = []
    for fine in fines:
        results.append(
            FineResult(
                implemented=bool(fine.implemented),
                category_name=fine.category_name,
                hit=bool(fine.hit),
                label=str(fine.label),
                score=None if fine.score is None else float(fine.score),
                detail=dict(fine.detail or {}),
            )
        )
    return results

