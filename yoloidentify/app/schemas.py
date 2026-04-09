"""
模块名称：schemas
作用：定义 API 请求/响应的统一数据结构，确保后续新增类别或替换模型不破坏接口形状。
使用方法：
  - 在 `app.main` 中作为 FastAPI response_model
  - 在 `app.pipeline` 中构造并返回
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TopKItem(BaseModel):
    category_id: int | None = None
    category_name: str
    score: float


class CoarseResult(BaseModel):
    category_id: int | None = None
    category_name: str
    score: float
    topk: list[TopKItem] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)


class FineResult(BaseModel):
    implemented: bool
    category_name: str
    hit: bool
    label: str
    score: float | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


FinalDecision = Literal["safe", "review", "block", "error"]


class AuditResponse(BaseModel):
    request_id: str
    coarse_result: CoarseResult
    next_stage: str
    # fine_result 保持兼容（多细分时填第一个）；推荐新字段 fine_results
    fine_result: FineResult | None = None
    fine_results: list[FineResult] = Field(default_factory=list)
    final_decision: FinalDecision

