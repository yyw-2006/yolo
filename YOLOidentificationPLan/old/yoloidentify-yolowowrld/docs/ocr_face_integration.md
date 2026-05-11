# OCR 与人脸识别接口说明

## 实现思路/结论

当前版本只实现 YOLOWorld 主干分流，OCR 与人脸识别链路保留为空实现。空实现的目的不是做假识别，而是固定后续接入真实模型时必须遵守的输入、输出、配置和错误边界。后续补齐 RapidOCR/PP-OCR、RetinaFace、ArcFace 时，只需要替换 `OcrPlaceholderJudge` 或 `FacePlaceholderJudge`，不需要改 FastAPI 入口和主流程。

## OCR 接口

接口文件：`app/models/ocr_base.py`

占位实现：`app/models/ocr_placeholder.py`

注册类别：`politics`、`protest`

调用签名：

```python
def predict(self, image: Any, coarse: CoarsePrediction) -> FinePrediction:
    ...
```

输入约定：

- `image`：当前上传图片，默认是 `PIL.Image.Image`，如果实现需要 OpenCV，可在模型内部转换为 `numpy.ndarray`。
- `coarse`：YOLOWorld 命中的当前类别结果，包含 `category_name`、`score`、`detections`。
- `coarse.detections`：包含触发 OCR 的原始 tag、置信度和可选框坐标，可用于只裁剪横幅、标语、旗帜区域做 OCR。

输出约定：

```python
FinePrediction(
    implemented=True,
    category_name="politics",
    hit=True,
    label="ocr_rule_hit",
    score=0.86,
    decision="block" or "review",
    chain="ocr",
    detail={
        "texts": ["识别到的文本"],
        "keyword_hits": ["命中的关键词"],
        "regions": [{"box": [x1, y1, x2, y2], "text": "...", "score": 0.9}],
        "model": {"name": "RapidOCR or PP-OCR", "backend": "onnxruntime or rknn"}
    },
)
```

实现建议：

- 开发机可先接 `rapidocr` + `onnxruntime`，验证文本提取和规则是否可靠。
- RK3588 NPU 化时，建议将 PP-OCR det/rec/cls 转为 RKNN，或用 FastDeploy RKNPU2 后端封装同样的 `predict` 接口。
- OCR 不负责纯图形符号识别，只负责标语、横幅、字幕、口号等文字内容。

## 人脸识别接口

接口文件：`app/models/face_base.py`

占位实现：`app/models/face_placeholder.py`

注册类别：`fallen_official`、`disgraced_artist`

调用签名：

```python
def predict(self, image: Any, coarse: CoarsePrediction) -> FinePrediction:
    ...
```

输入约定：

- `image`：当前上传图片。
- `coarse`：YOLOWorld 命中的当前敏感人脸候选类别。
- `coarse.detections`：通常来自 `person`、`face`、`official`、`celebrity` 等 tag，可作为是否进入人脸检测的触发依据。

输出约定：

```python
FinePrediction(
    implemented=True,
    category_name="fallen_official",
    hit=True,
    label="face_match",
    score=0.82,
    decision="block" or "review",
    chain="face",
    detail={
        "faces": [
            {
                "box": [x1, y1, x2, y2],
                "name": "人物姓名",
                "person_tags": ["fallen_official"],
                "similarity": 0.82,
                "threshold": 0.75
            }
        ],
        "model": {"detector": "RetinaFace", "recognizer": "ArcFace", "backend": "rknn"}
    },
)
```

人物库建议格式：

```json
{
  "version": 1,
  "metric": "cosine",
  "threshold": 0.75,
  "persons": [
    {
      "name": "人物姓名",
      "tags": ["fallen_official"],
      "embedding_file": "weights/person_db/person_name.npy"
    }
  ]
}
```

实现建议：

- 人脸检测阶段使用 RetinaFace 输出人脸框和关键点。
- 裁剪对齐后使用 ArcFace 提取 embedding。
- 使用余弦相似度与人物库 embedding 比对，命中后先返回姓名，再根据 `tags` 映射到 `fallen_official` 或 `disgraced_artist`。
- 不建议把真实敏感人物库硬编码进代码，应放在 `weights/person_db/` 或外部私有存储中。

## RK3588 部署注意事项

- `.rknn` 模型需要在 x86 开发机用 RKNN-Toolkit2 转换，再复制到 RK3588 板端运行。
- 板端 Python 推理依赖 `rknn-toolkit-lite2`，并需要系统已启用 NPU。
- 运行时可用 `watch -n 0.5 cat /sys/kernel/debug/rknpu/load` 查看 3 核 NPU 负载。
- 如果某模型存在不支持算子，RKNN 可能回退 CPU 或转换失败，需要回到模型导出/后处理阶段处理。
