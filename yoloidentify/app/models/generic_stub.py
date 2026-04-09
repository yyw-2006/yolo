"""
模块名称：generic_stub
作用：未注册风险类别的通用细分占位实现，保证新增类别但细分模型未完成时链路可用。
使用方法：
  - 在 `JudgeRegistry` 未命中时作为兜底返回
"""

from __future__ import annotations

from app.models.base import CoarsePrediction, FineJudge, FinePrediction


class GenericStubJudge(FineJudge):
    def predict(self, image, coarse: CoarsePrediction) -> FinePrediction:
        return FinePrediction(
            implemented=False,
            category_name=coarse.category_name,
            hit=False,
            label="not_implemented",
            score=None,
            detail={"message": "fine model placeholder (generic)"},
        )

