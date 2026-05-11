"""
模块名称：settings
作用：集中管理 YOLO 识别服务配置项（上传限制、类别映射、YOLOWorld/RKNN 路径等）。
使用方法：
  - 通过环境变量覆盖，例如：
      AUDIT_CATEGORY_MAPPING_PATH=config/category_mapping.json
      AUDIT_YOLO_WORLD_WEIGHTS=weights/yolov8s-world.pt
      AUDIT_INFERENCE_BACKEND=ultralytics
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AUDIT_", env_file=".env", extra="ignore")

    # 上传限制
    max_image_bytes: int = 5 * 1024 * 1024  # 5MB

    # 类别映射配置
    category_mapping_path: str = "config/category_mapping.json"

    # 推理后端：ultralytics 用于开发机验证；rknn 用于 RK3588 板端模型入口。
    inference_backend: str = "ultralytics"

    # YOLOWorld 模型
    yolo_world_weights: str = "weights/yolov8s-world.pt"
    yolo_world_rknn_model: str = "weights/yolo_world_v2s.rknn"
    yolo_world_text_model: str = "weights/clip_text.rknn"
    yolo_world_classes_file: str = "weights/detect_classes.txt"
    device: str = "cpu"

    # 保留 topk 便于后续调参/分析
    topk: int = 5

    # YOLOWorld 推理参数
    label_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_det: int = 64
    imgsz: int = 640
    warmup: bool = True

    # image-multi-detect 安全图片分类器
    image_multi_detect_model: str = "weights/image-multi-detect/model.onnx"
    image_multi_detect_weapon_threshold: float = 0.5
    image_multi_detect_violence_threshold: float = 0.3
    image_multi_detect_hate_threshold: float = 0.5
    image_multi_detect_drugs_threshold: float = 0.3
    image_multi_detect_sensitive_as_drugs_threshold: float = 0.7
    image_multi_detect_nsfw_as_drugs_threshold: float = 0.4


settings = Settings()

