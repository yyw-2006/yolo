"""
模块名称：settings
作用：集中管理图片审核 MVP 的配置项（权重路径、阈值、类别映射、上传限制等）。
使用方法：
  - 通过环境变量覆盖，例如：
      AUDIT_COARSE_WEIGHTS=weights/coarse_cls.pt
      AUDIT_MAX_IMAGE_BYTES=10485760
"""

from __future__ import annotations

import json
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AUDIT_", env_file=".env", extra="ignore")

    # 上传限制
    max_image_bytes: int = 5 * 1024 * 1024  # 5MB

    # 粗分类模型
    coarse_weights: str = "weights/coarse_cls.pt"
    device: str = "cpu"

    # 类别别名映射：把模型的 class name 映射为业务 category_name
    # 例如：{"violent":"violence","flag_illegal":"flag","sensitive_person":"person"}
    # 以 JSON 字符串方式从环境变量注入：AUDIT_CATEGORY_ALIASES='{"a":"b"}'
    category_aliases: dict[str, str] = {}

    # 保留 topk 便于后续调参/分析
    topk: int = 5

    # 统一的“正常类别名”
    normal_category_name: str = "normal"

    @classmethod
    def parse_env_var(cls, field_name: str, raw_val: str) -> Any:
        if field_name == "category_aliases":
            if not raw_val:
                return {}
            return json.loads(raw_val)
        return raw_val


settings = Settings()

