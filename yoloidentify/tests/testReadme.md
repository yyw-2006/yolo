测试与基准脚本说明（tests/testReadme）

本目录下的脚本主要用于：
- API 手工测试（启动服务后用 curl 调用）
- 基准跑分（直接在进程内用 FastAPI TestClient 调用 `/audit/image`）
- 样本导入/整理脚本（可选）

## 1) 前置条件
- Python 3.10+（建议）
- 已安装依赖：`pip install -r yoloidentify/requirements.txt`
- 已准备好样本图片目录：默认使用 `yoloidentify/samples/<tag>/*`

说明：`benchmark_samples.py` 会把 `samples/` 下的每个子目录名当作标签（tag），预测值与该 tag 相等则记为正确。

## 2) 启动服务并用 curl 测试（手测）
在仓库根目录执行：

```bash
cd "/home/yao/桌面/yoloidentify"
source yoloidentify/.venv/bin/activate  # 如果你使用了 venv
pip install -r yoloidentify/requirements.txt

cd yoloidentify
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

另开一个终端进行健康检查与上传测试：

```bash
curl "http://127.0.0.1:8000/health"
curl -X POST "http://127.0.0.1:8000/audit/image" -F "file=@samples/normal/normal.jpg"
```

更完整的接口字段说明见 `tests/test_api.md`。

## 3) 运行基准脚本（推荐）
该脚本**不需要启动 uvicorn**，会在同进程内创建 `TestClient(app)` 调用接口。

```bash
cd "/home/yao/桌面/yoloidentify/yoloidentify"
python -m tests.benchmark_samples
```

如需指定样本目录：

```bash
cd "/home/yao/桌面/yoloidentify/yoloidentify"
SAMPLES_DIR="/abs/path/to/samples" python -m tests.benchmark_samples
```

输出包含：
- 总图片数、总耗时、平均耗时、整体准确率
- 每个 tag 的数量/准确率/平均耗时
- 前 20 个不匹配样本（tag vs pred）

## 4) 常见问题
- 找不到样本：确认 `yoloidentify/samples/` 下存在子目录（如 `normal/`、`human/`、`violence/`）且里面有图片文件。
- 依赖缺失：先执行 `pip install -r yoloidentify/requirements.txt`。
- 模型下载/加载慢：首次运行 `ultralytics` 可能需要联网下载权重；可提前把权重放到 `yoloidentify/weights/` 并用环境变量覆盖（见项目 `README.md` 的“配置（环境变量）”）。

