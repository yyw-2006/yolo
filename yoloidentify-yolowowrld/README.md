# RK3588 YOLO 识别服务

## 实现思路/结论

本项目按 `YOLOidentificationPLan/yolo识别方案.md` 实现为 `输入 -> YOLOWorld -> 根据 tag 分流 -> 输出类别结果`。当前覆盖 10 个业务类别：涉政、游行集会、落马官员、劣迹艺人、暴恐、违禁毒品、非法宗教、赌博、纳粹符号、不雅手势。类别/tag 映射统一放在 `config/category_mapping.json`，后续调整 prompt、阈值、优先级和处理链路时优先改配置。OCR 与人脸识别本轮只保留接口和占位返回，完整接入说明见 `docs/ocr_face_integration.md`。

## 使用方法/运行方式

在 `yoloidentify/` 目录下安装依赖并启动服务：

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

如果系统没有 `python` 命令，请使用 `python3` 与 `pip3`。上传图片测试：

```bash
curl -X POST "http://127.0.0.1:8000/audit/image" \
  -F "file=@samples/nazi_symbol/nazi_symbol_Peckerwood_(hand_sign)_3.webp"
```

## 目录结构

```text
yoloidentify/
  app/
    main.py
    settings.py
    schemas.py
    pipeline.py
    services/
      router.py
      judge_registry.py
      category_config.py
    models/
      base.py
      coarse_yolo_cls.py
      direct_judge.py
      ocr_base.py
      ocr_placeholder.py
      face_base.py
      face_placeholder.py
      generic_stub.py
  config/
    category_mapping.json
  docs/
    ocr_face_integration.md
  tools/
    run_yoloworld.py
  weights/
    yolov8s-world.pt
    yolo_world_v2s.rknn
    clip_text.rknn
  samples/
    README.md
  tests/
    import_public_samples.py
    benchmark_samples.py
    test_api.md
  requirements.txt
```

## 配置（环境变量）

- `AUDIT_MAX_IMAGE_BYTES`：上传图片最大字节数，默认 `5242880`（5MB）
- `AUDIT_CATEGORY_MAPPING_PATH`：类别映射配置，默认 `config/category_mapping.json`
- `AUDIT_INFERENCE_BACKEND`：推理后端，默认 `ultralytics`；RK3588 板端可设为 `rknn`
- `AUDIT_YOLO_WORLD_WEIGHTS`：开发机 YOLOWorld 权重，默认 `weights/yolov8s-world.pt`，不存在时回退 `yolov8s-world.pt`
- `AUDIT_YOLO_WORLD_RKNN_MODEL`：RKNN YOLOWorld 检测模型路径
- `AUDIT_YOLO_WORLD_TEXT_MODEL`：RKNN CLIP text 模型路径
- `AUDIT_YOLO_WORLD_CLASSES_FILE`：RKNN prompt/class 文件路径
- `AUDIT_DEVICE`：推理设备，默认 `cpu`（可设为 `0` 等 GPU 设备号，取决于环境）
- `AUDIT_LABEL_THRESHOLD`：YOLOWorld tag 置信度阈值
- `AUDIT_TOPK`：返回 topk 数量

## 类别映射

`config/category_mapping.json` 是类别调整入口。每个类别包含：

- `name`：API 内部类别名，例如 `terrorism`
- `display_name`：中文显示名，例如 `暴恐`
- `chain`：处理链路，支持 `normal`、`direct`、`ocr`、`face`
- `decision`：命中后的最终决策倾向，支持 `safe`、`review`、`block`
- `prompts`：传给 YOLOWorld 的开放词表 prompt
- `tags`：模型输出 tag 到业务类别的匹配词

## 当前链路行为

- `normal`：返回 `safe`
- `direct`：YOLOWorld 命中后直接返回 `block`
- `ocr`：返回 `review` 和 `not_implemented`，等待后续接 OCR
- `face`：返回 `review` 和 `not_implemented`，等待后续接人脸识别

## 快速测试

先准备一张图片（例如放到 `samples/test.jpg`），然后执行：

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

另开终端测试：

```bash
curl -X POST "http://127.0.0.1:8000/audit/image" \
  -F "file=@samples/test.jpg"
```

导入公开样本示例：

```bash
SOURCE=github_hate_symbols CATEGORY=nazi_symbol TAKE=2 python3 -m tests.import_public_samples
```

运行目录级验证：

```bash
python3 -m tests.benchmark_samples
```

单张图片调试 YOLOWorld 主干：

```bash
python3 tools/run_yoloworld.py samples/normal/normal2.jpg
```

