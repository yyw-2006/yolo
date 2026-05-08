"""
模块名称：judge_registry
作用：按风险类别名注册/获取细分判断处理器，支持后续扩展（政治/游行等）而不改主流程。
使用方法：
  - registry = JudgeRegistry(...)
  - judge = registry.get(category_name)
"""

from __future__ import annotations

from dataclasses import dataclass

from app.models.base import FineJudge
from app.models.generic_stub import GenericStubJudge


@dataclass(frozen=True)
class JudgeRegistry:
    _mapping: dict[str, FineJudge]
    _fallback: FineJudge

    @classmethod
    def build_default(cls, mapping: dict[str, FineJudge] | None = None) -> "JudgeRegistry":
        return cls(_mapping=mapping or {}, _fallback=GenericStubJudge())

    def get(self, category_name: str) -> FineJudge:
        return self._mapping.get(category_name, self._fallback)

