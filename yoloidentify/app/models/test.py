from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
import time

from PIL import Image
import torch


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models.coarse_yolo_cls import YoloV8CoarseClassifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLOWorld coarse recognition on a single image.")
    parser.add_argument("image", help="Absolute or relative path to the input image.")
    parser.add_argument(
        "--weights",
        default="",
        help="Optional YOLOWorld weights path. Falls back to yolov8s-world.pt when omitted or missing.",
    )
    parser.add_argument("--device", default="auto", help="Inference device, for example auto, cpu or 0.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold.")
    parser.add_argument("--max-det", type=int, default=64, help="Maximum number of detections.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of untimed warmup runs before measuring.")
    return parser


def synchronize_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> int:
    args = build_parser().parse_args()
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    classifier = YoloV8CoarseClassifier(
        weights_path=args.weights,
        device=args.device,
        label_threshold=args.conf,
        iou_threshold=args.iou,
        max_det=args.max_det,
    )

    for _ in range(max(0, int(args.warmup_runs))):
        classifier.predict(image)

    synchronize_if_needed()
    start = time.perf_counter()
    prediction = classifier.predict(image)
    synchronize_if_needed()
    end = time.perf_counter()
    print(json.dumps(asdict(prediction), ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "device": args.device,
                "warmup_runs": max(0, int(args.warmup_runs)),
                "elapsed_seconds": round(end - start, 6),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
