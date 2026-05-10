"""Run YOLOv8 item classification on one image and print a result list."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def project_root() -> Path:
    script = Path(__file__).resolve()
    for parent in script.parents:
        if (parent / "data").is_dir() and (parent / "weights").is_dir() and (parent / "scripts").is_dir():
            return parent
    return script.parents[2]


def add_local_deps(root: Path) -> None:
    for candidate_root in (root, *root.parents):
        deps = candidate_root / ".deps"
        if deps.exists():
            sys.path.insert(0, str(deps))


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input image path.")
    parser.add_argument("--weights", type=Path, default=root / "weights" / "识别物品" / "yolov8n_cls_best.pt", help="YOLOv8 classification weight.")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    root = project_root()
    add_local_deps(root)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing ultralytics. Install project-local deps with:\n"
            "python -m pip install --target .deps ultralytics ultralytics-thop "
            "torchvision==0.22.1 polars"
        ) from exc

    args = parse_args()
    image = args.image.resolve()
    weights = args.weights.resolve()
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")
    if not weights.exists():
        raise FileNotFoundError(f"Weight file not found: {weights}")

    model = YOLO(str(weights))
    names = {int(key): str(value) for key, value in dict(model.names).items()}
    result = model.predict(source=str(image), imgsz=args.imgsz, device=args.device, verbose=False)[0]
    scores = result.probs.data.detach().cpu()
    topk = max(1, min(int(args.topk), int(scores.numel())))
    values, indices = scores.topk(topk)

    results = [
        {
            "type": "item",
            "label": names.get(int(index), str(int(index))),
            "confidence": float(value),
        }
        for value, index in zip(values.tolist(), indices.tolist())
    ]
    payload = {
        "image": str(image),
        "weights": str(weights),
        "results": results,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
