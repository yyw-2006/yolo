# 权重目录说明

## 实现思路/结论

当前版本默认用 Ultralytics YOLOWorld 做开发机 fallback，类别映射来自 `config/category_mapping.json`。RK3588 方向保留 `.rknn` 模型配置入口，但 OCR 与人脸真实模型暂不接入。后续补齐 RKNN 后，只需要把模型放入本目录并设置对应环境变量。

## 使用方法/运行方式

开发机默认权重：

- `weights/yolov8s-world.pt`

如果该文件不存在，Ultralytics 会尝试按 `yolov8s-world.pt` 名称自动下载。

RK3588 YOLOWorld 预期文件：

- `weights/yolo_world_v2s.rknn`
- `weights/clip_text.rknn`
- `weights/detect_classes.txt`

板端环境变量示例：

```bash
export AUDIT_INFERENCE_BACKEND=rknn
export AUDIT_YOLO_WORLD_RKNN_MODEL=weights/yolo_world_v2s.rknn
export AUDIT_YOLO_WORLD_TEXT_MODEL=weights/clip_text.rknn
export AUDIT_YOLO_WORLD_CLASSES_FILE=weights/detect_classes.txt
```

OCR 和人脸后续模型建议路径：

- `weights/ocr/det.rknn`
- `weights/ocr/rec.rknn`
- `weights/ocr/cls.rknn`
- `weights/face/retinaface.rknn`
- `weights/face/arcface.rknn`
- `weights/person_db/person_db.json`

具体接口和人物库格式见 `docs/ocr_face_integration.md`。

