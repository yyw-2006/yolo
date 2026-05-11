"""
模块名称：ocr_base
作用：定义 OCR 链路接口，后续可接入 RapidOCR、PP-OCR 或 RKNN/FastDeploy 实现。
使用方法：
  - class MyOcrJudge(OcrJudge): ...
  - 返回 `FinePrediction`，detail 中放置 OCR 文本、命中规则和模型信息。
"""

from __future__ import annotations

from typing import Any, Protocol

from app.models.base import CoarsePrediction, FinePrediction


class OcrJudge(Protocol):
    def predict(self, image: Any, coarse: CoarsePrediction) -> FinePrediction:
        raise NotImplementedError
