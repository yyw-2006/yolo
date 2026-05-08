"""
模块名称：ocr_placeholder
作用：OCR 链路占位实现，本轮不接真实 OCR 模型，但固定后续接入的返回结构。
使用方法：
  - registry 注册到 `politics`、`protest` 等 OCR 类别
  - 后续用 RapidOCR/RKNN 实现替换该类即可
"""

from __future__ import annotations

from app.models.base import CoarsePrediction, FinePrediction
from app.models.ocr_base import OcrJudge


class OcrPlaceholderJudge(OcrJudge):
    def predict(self, image, coarse: CoarsePrediction) -> FinePrediction:
        return FinePrediction(
            implemented=False,
            category_name=coarse.primary_category_name,
            hit=False,
            label="not_implemented",
            score=None,
            decision="review",
            chain="ocr",
            detail={
                "message": "OCR stage is reserved. Implement RapidOCR/PP-OCR and rule matching here.",
                "expected_input": {
                    "image": "PIL.Image.Image or numpy image",
                    "coarse": "YOLOWorld coarse prediction with matched tags and boxes",
                },
                "expected_output": {
                    "texts": ["recognized text item"],
                    "keyword_hits": ["matched business keyword"],
                    "hit": "true when OCR rules confirm this category",
                },
            },
        )
