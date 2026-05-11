"""
模块名称：face_base
作用：定义敏感人脸链路接口，后续可接入 RetinaFace + ArcFace 或等价 RKNN 实现。
使用方法：
  - class MyFaceJudge(FaceJudge): ...
  - 返回 `FinePrediction`，detail 中放置人脸框、姓名、人物标签和相似度。
"""

from __future__ import annotations

from typing import Any, Protocol

from app.models.base import CoarsePrediction, FinePrediction


class FaceJudge(Protocol):
    def predict(self, image: Any, coarse: CoarsePrediction) -> FinePrediction:
        raise NotImplementedError
