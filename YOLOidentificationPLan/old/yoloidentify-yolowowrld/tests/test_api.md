图片审核 API 测试说明

## 1) 启动服务
在项目目录下执行：
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 2) 健康检查
```bash
curl "http://127.0.0.1:8000/health"
```

## 3) 上传图片进行审核
准备一张测试图片，例如 `samples/test.jpg`，然后执行：
```bash
curl -X POST "http://127.0.0.1:8000/audit/image" \
  -F "file=@samples/test.jpg"
```

## 4) 期望返回结构（示例字段）
你应该至少能看到以下字段（具体值取决于粗分类权重与图片内容）：
- `request_id`
- `coarse_result.category_name`
- `coarse_result.score`
- `coarse_result.topk`
- `coarse_result.labels`（YOLOWorld 命中的业务类别列表）
- `coarse_result.detections`（原始 tag、业务类别、置信度、可选坐标）
- `next_stage`
- `fine_results`（当 coarse.labels 命中非 normal 类别时出现，可能包含直接命中、OCR 占位或人脸占位）
- `fine_result`（兼容字段：当出现多个时取 `fine_results[0]`）
- `final_decision`（`normal` 返回 `safe`；直接 tag 类命中返回 `block`；OCR/人脸占位返回 `review`）

## 5) 常见问题排查
- 如果报 `Unsupported content_type`：确认上传的是图片文件（`curl -F file=@xx.jpg`）
- 如果报 `File too large`：调大 `AUDIT_MAX_IMAGE_BYTES`
- 如果报模型加载失败：
  - 开发机确认 `AUDIT_INFERENCE_BACKEND=ultralytics`
  - 放置 YOLOWorld 权重到 `weights/yolov8s-world.pt`
  - 或保持联网让 `ultralytics` 自动下载 `yolov8s-world.pt`
- 如果要调整类别/tag：
  - 修改 `config/category_mapping.json`

