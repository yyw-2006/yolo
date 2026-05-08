"""
模块名称：face_placeholder
作用：敏感人脸链路占位实现，本轮不接真实 RetinaFace/ArcFace 模型。
使用方法：
  - registry 注册到 `fallen_official`、`disgraced_artist`
  - 后续用真实人脸检测、特征提取和人物库比对实现替换该类即可
"""

from __future__ import annotations

from app.models.base import CoarsePrediction, FinePrediction
from app.models.face_base import FaceJudge


class FacePlaceholderJudge(FaceJudge):
    def predict(self, image, coarse: CoarsePrediction) -> FinePrediction:
        return FinePrediction(
            implemented=False,
            category_name=coarse.primary_category_name,
            hit=False,
            label="not_implemented",
            score=None,
            decision="review",
            chain="face",
            detail={
                "message": "Face stage is reserved. Implement RetinaFace + ArcFace comparison here.",
                "expected_input": {
                    "image": "PIL.Image.Image or numpy image",
                    "coarse": "YOLOWorld coarse prediction with person/face trigger tags",
                },
                "expected_output": {
                    "faces": [
                        {
                            "box": [0, 0, 0, 0],
                            "name": "person name",
                            "person_tags": ["fallen_official or disgraced_artist"],
                            "similarity": 0.0,
                        }
                    ],
                    "hit": "true when a known sensitive person is matched",
                },
            },
        )
