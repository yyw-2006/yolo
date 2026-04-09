# 图片审核 MVP（粗分类 + 可扩展框架）
#
# 作用：
# - 在本机/虚拟机快速跑通“图片输入 -> 粗分类 -> 分流 -> 细分接口（占位）-> JSON 返回”的闭环。
# - 细分判断模型（如政治/游行/旗帜/暴力/人物等）第一版只实现接口与占位返回，便于后续逐步替换为真实模型。
#
# 使用方法（最小示例）：
# 1) 安装依赖：pip install -r requirements.txt
# 2) 启动服务：uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# 3) 调用接口：curl -X POST http://127.0.0.1:8000/audit/image -F "file=@samples/test.jpg"

## 实现思路/结论
本项目按“粗分类先行、命中后分流”的漏斗链路实现：所有图片先进入粗分类模型得到 `category_name`，`normal` 直接返回 `safe`；其他风险类别进入 `JudgeRegistry` 查找对应细分处理器。细分处理器第一版以 stub 形式存在，并提供 `generic_stub` 兜底，保证未来新增 `politics`、`protest` 等类别时无需重写主链路，只需新增实现并注册即可。整体输出协议从第一版就固定，后续替换模型时只改适配层，不改 API 形状与编排逻辑。

## 使用方法/运行方式
在虚拟机中建议使用 `Python 3.10+`。先执行 `pip install -r requirements.txt` 安装依赖，然后用 `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000` 启动服务，打开 `http://127.0.0.1:8000/docs` 可直接在 Swagger UI 上传图片测试。粗分类权重默认读取 `weights/coarse_cls.pt`，若不存在会尝试使用 `YOLOv8n-cls` 的默认权重名（需要联网自动下载）；你也可以通过环境变量覆盖（见下文配置）。使用 `curl` 或 `httpie` 上传图片后，返回 JSON 会包含粗分类结果、下一阶段去向、细分占位结果与最终决策，便于你验证链路是否跑通。

如果你系统里没有 `python` 命令，请使用 `python3` 与 `pip3`（Ubuntu/Debian 常见）。

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
    models/
      base.py
      coarse_yolo_cls.py
      violence_stub.py
      flag_stub.py
      person_stub.py
      generic_stub.py
  weights/
    coarse_cls.pt
  samples/
    README.md
  tests/
    test_api.md
  requirements.txt
```

## 配置（环境变量）
- `AUDIT_MAX_IMAGE_BYTES`：上传图片最大字节数，默认 `5242880`（5MB）
- `AUDIT_COARSE_WEIGHTS`：粗分类权重路径，默认 `weights/coarse_cls.pt`，若文件不存在则回退到 `yolov8n-cls.pt`
- `AUDIT_DEVICE`：推理设备，默认 `cpu`（可设为 `0` 等 GPU 设备号，取决于环境）
- `AUDIT_CATEGORY_ALIASES`：类别别名映射（JSON 字符串），将模型的 class name 映射为业务 `category_name`

类别扩展方式（例如新增 `politics` / `protest`）：
- 粗分类模型侧：新增类别并训练/导出权重，或至少保证输出的 class name 能映射到目标 `category_name`
- 配置侧：通过 `AUDIT_CATEGORY_ALIASES` 把模型 class name 映射为 `politics` / `protest`
- 代码侧：新增对应 stub/实现，并在 `app/main.py` 的 `JudgeRegistry` 注册表中注册

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

如果你还没有自训粗分类权重，可先联网让 `ultralytics` 自动下载默认 `yolov8n-cls.pt`（首次较慢）。后续你把自训权重放到 `weights/coarse_cls.pt`，并保证类别名或别名映射正确即可。

