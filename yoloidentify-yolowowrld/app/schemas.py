"""
模块名称：schemas
作用：定义 YOLO 识别 API 的统一响应结构，覆盖主干 tag、OCR 占位和人脸占位结果。
使用方法：
  - 在 `app.main` 中作为 FastAPI response_model
  - 在 `app.pipeline` 中构造并返回
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


FinalDecision = Literal["safe", "review", "block", "error"]


class TopKItem(BaseModel):
    category_id: int | None = None
    category_name: str
    score: float
    display_name: str = ""
    chain: str = ""


class TagDetectionItem(BaseModel):
    raw_tag: str
    category_name: str
    score: float
    chain: str
    box: list[float] | None = None


class CoarseResult(BaseModel):
    category_id: int | None = None
    category_name: str
    score: float
    topk: list[TopKItem] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    detections: list[TagDetectionItem] = Field(default_factory=list)


class FineResult(BaseModel):
    implemented: bool
    category_name: str
    hit: bool
    label: str
    score: float | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    decision: FinalDecision = "review"
    chain: str = ""

class AuditResponse(BaseModel):
    request_id: str
    coarse_result: CoarseResult
    next_stage: str
    # fine_result 保持兼容（多细分时填第一个）；推荐新字段 fine_results
    fine_result: FineResult | None = None
    fine_results: list[FineResult] = Field(default_factory=list)
    final_decision: FinalDecision

