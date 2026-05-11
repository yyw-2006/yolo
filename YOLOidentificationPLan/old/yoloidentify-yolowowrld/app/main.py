"""
模块名称：main
作用：FastAPI 服务入口，提供 YOLOWorld 主干分流 + OCR/人脸占位接口。
使用方法：
  - 安装依赖：pip install -r requirements.txt
  - 启动服务：uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
  - 调用接口：POST /audit/image 上传图片
"""

from __future__ import annotations

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from app.models.composite_coarse import CompositeCoarseClassifier
from app.models.coarse_yolo_cls import RknnYoloWorldClassifier, YoloWorldClassifier
from app.models.direct_judge import DirectTagJudge
from app.models.face_placeholder import FacePlaceholderJudge
from app.models.image_multi_detect import ImageMultiDetectClassifier
from app.models.ocr_placeholder import OcrPlaceholderJudge
from app.pipeline import AuditPipeline
from app.schemas import AuditResponse
from app.services.category_config import CategoryMapping, load_category_mapping
from app.services.judge_registry import JudgeRegistry
from app.settings import settings


app = FastAPI(title="RK3588 YOLO Image Audit", version="0.2.0")


def _filter_mapping(mapping: CategoryMapping, category_names: set[str]) -> CategoryMapping:
    keep_names = set(category_names)
    keep_names.add(mapping.normal_category)
    return CategoryMapping(
        normal_category=mapping.normal_category,
        default_threshold=mapping.default_threshold,
        topk=mapping.topk,
        categories=tuple(item for item in mapping.categories if item.name in keep_names),
    )


def _build_classifier(mapping: CategoryMapping):
    backend = settings.inference_backend.strip().lower()
    yolo_mapping = _filter_mapping(
        mapping,
        category_names={"politics", "protest", "fallen_official", "disgraced_artist"},
    )

    if backend == "rknn":
        yolo_classifier = RknnYoloWorldClassifier(
            rknn_model_path=settings.yolo_world_rknn_model,
            text_model_path=settings.yolo_world_text_model,
            classes_file=settings.yolo_world_classes_file,
            mapping=yolo_mapping,
            topk=settings.topk,
        )
    else:
        yolo_classifier = YoloWorldClassifier(
            weights_path=settings.yolo_world_weights,
            mapping=yolo_mapping,
            device=settings.device,
            topk=settings.topk,
            label_threshold=settings.label_threshold,
            iou_threshold=settings.iou_threshold,
            max_det=settings.max_det,
            imgsz=settings.imgsz,
            warmup=settings.warmup,
        )

    image_multi_classifier = ImageMultiDetectClassifier(
        model_path=settings.image_multi_detect_model,
        mapping=mapping,
        topk=settings.topk,
        weapon_threshold=settings.image_multi_detect_weapon_threshold,
        violence_threshold=settings.image_multi_detect_violence_threshold,
        hate_threshold=settings.image_multi_detect_hate_threshold,
        drugs_threshold=settings.image_multi_detect_drugs_threshold,
        sensitive_as_drugs_threshold=settings.image_multi_detect_sensitive_as_drugs_threshold,
        nsfw_as_drugs_threshold=settings.image_multi_detect_nsfw_as_drugs_threshold,
    )
    return CompositeCoarseClassifier(
        mapping=mapping,
        classifiers=[yolo_classifier, image_multi_classifier],
        topk=settings.topk,
    )


def _build_registry(mapping: CategoryMapping) -> JudgeRegistry:
    ocr_placeholder = OcrPlaceholderJudge()
    face_placeholder = FacePlaceholderJudge()
    registry_mapping = {}
    for category in mapping.categories:
        if category.chain == "direct":
            registry_mapping[category.name] = DirectTagJudge(decision=category.decision)
        elif category.chain == "ocr":
            registry_mapping[category.name] = ocr_placeholder
        elif category.chain == "face":
            registry_mapping[category.name] = face_placeholder
    return JudgeRegistry.build_default(mapping=registry_mapping)


def _build_pipeline() -> AuditPipeline:
    mapping = load_category_mapping(settings.category_mapping_path)
    coarse = _build_classifier(mapping)
    registry = _build_registry(mapping)

    return AuditPipeline(coarse=coarse, registry=registry, normal_category_name=mapping.normal_category)


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

