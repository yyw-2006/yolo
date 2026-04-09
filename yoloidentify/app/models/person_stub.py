"""
模块名称：person_stub
作用：人物相关细分判断模型的占位实现（接口先行，后续替换为真实模型）。
使用方法：
  - 注册到 `JudgeRegistry`，类别名通常为 `person`
"""

from __future__ import annotations

from app.models.base import CoarsePrediction, FineJudge, FinePrediction


class PersonStubJudge(FineJudge):
    def predict(self, image, coarse: CoarsePrediction) -> FinePrediction:
        return FinePrediction(
            implemented=False,
            category_name=coarse.category_name,
            hit=False,
            label="not_implemented",
            score=None,
            detail={"message": "fine model placeholder (person)"},
        )

