权重目录说明

默认粗分类权重路径为：
- `weights/coarse_cls.pt`

如果该文件不存在，程序会回退尝试使用 `yolov8n-cls.pt`（ultralytics 可能会联网下载）。

建议你把自训粗分类权重放到 `weights/coarse_cls.pt`，并确保类别名与业务类别一致，或用环境变量 `AUDIT_CATEGORY_ALIASES` 做映射。

