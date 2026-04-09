"""
模块名称：main
作用：FastAPI 服务入口，提供图片审核 API（粗分类 + 可扩展细分接口占位）。
使用方法：
  - 安装依赖：pip install -r requirements.txt
  - 启动服务：uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
  - 调用接口：POST /audit/image 上传图片
"""

from __future__ import annotations

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from app.models.coarse_yolo_cls import YoloV8CoarseClassifier
from app.models.flag_stub import FlagStubJudge
from app.models.person_stub import PersonStubJudge
from app.models.violence_stub import ViolenceStubJudge
from app.pipeline import AuditPipeline
from app.schemas import AuditResponse
from app.services.judge_registry import JudgeRegistry
from app.settings import settings


app = FastAPI(title="Image Audit MVP", version="0.1.0")


def _build_pipeline() -> AuditPipeline:
    coarse = YoloV8CoarseClassifier(
        weights_path=settings.coarse_weights,
        device=settings.device,
        topk=settings.topk,
        category_aliases=settings.category_aliases,
    )

    registry = JudgeRegistry.build_default(
        mapping={
            "violence": ViolenceStubJudge(),
            "flag": FlagStubJudge(),
            "person": PersonStubJudge(),
            # 后续扩展示例：
            # "politics": PoliticsStubJudge(),
            # "protest": ProtestStubJudge(),
        }
    )

    return AuditPipeline(coarse=coarse, registry=registry, normal_category_name=settings.normal_category_name)


_pipeline: AuditPipeline | None = None
_pipeline_init_error: str | None = None


@app.on_event("startup")
def _startup() -> None:
    global _pipeline, _pipeline_init_error
    try:
        _pipeline = _build_pipeline()
        _pipeline_init_error = None
    except Exception as e:
        _pipeline = None
        _pipeline_init_error = f"{type(e).__name__}: {str(e)}"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/audit/image", response_model=AuditResponse)
async def audit_image(file: UploadFile = File(...)) -> AuditResponse:
    if _pipeline is None:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline not ready. {_pipeline_init_error or 'unknown init error'}",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Unsupported content_type: {file.content_type}")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(raw) > settings.max_image_bytes:
        raise HTTPException(status_code=413, detail=f"File too large: {len(raw)} bytes.")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}") from e

    try:
        return _pipeline.audit_image(image)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audit failed: {type(e).__name__}: {str(e)}") from e

