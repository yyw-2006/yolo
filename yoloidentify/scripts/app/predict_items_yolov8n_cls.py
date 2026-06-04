"""Run YOLOv8 item detection on one image and print recognized item classes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DEFAULT_CLASS_THRESHOLDS = {
    "banner": 0.50,
    "blood": 0.35,
    "chip": 0.50,
    "crowd": 0.50,
    "dice": 0.50,
    "flag": 0.50,
    "gun": 0.35,
    "knife": 0.35,
    "middle_finger": 0.50,
    "nazi_symbol": 0.50,
    "pill": 0.50,
    "poker_card": 0.50,
    "police": 0.50,
    "powder": 0.50,
    "roulette": 0.50,
    "signboard": 0.50,
    "slot_machine": 0.35,
    "swastika": 0.50,
    "syringe": 0.50,
}


def project_root() -> Path:
    script = Path(__file__).resolve()
    for parent in script.parents:
        if (parent / "weights").is_dir() and (parent / "scripts").is_dir():
            return parent
    return script.parents[2]


def add_local_deps(root: Path) -> None:
    for candidate_root in (root, *root.parents):
        deps = candidate_root / ".deps"
        if deps.exists():
            sys.path.insert(0, str(deps))


def default_weights(root: Path) -> Path:
    candidates = [
        root / "weights" / "识别物品" / "yolov8n_cls_best.pt",
        root / "weights" / "identify_items" / "yolov8n_cls_best.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input image path.")
    parser.add_argument("--weights", type=Path, default=default_weights(root), help="YOLOv8 detection weight.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iou", type=float, default=0.70, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--threshold", type=float, default=None, help="Optional global confidence threshold override.")
    parser.add_argument("--min-conf", type=float, default=None, help="Internal YOLO predict confidence threshold.")
    parser.add_argument("--include-boxes", action="store_true", help="Include per-box xyxy detections in the JSON output.")
    parser.add_argument("--topk", type=int, default=0, help="Accepted for backwards compatibility and ignored.")
    return parser.parse_args()


def class_thresholds(global_threshold: float | None) -> dict[str, float]:
    if global_threshold is None:
        return dict(DEFAULT_CLASS_THRESHOLDS)
    return {name: float(global_threshold) for name in DEFAULT_CLASS_THRESHOLDS}


def detection_results(result, thresholds: dict[str, float], include_boxes: bool) -> list[dict[str, object]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    names = {int(key): str(value) for key, value in dict(result.names).items()}
    xyxy = boxes.xyxy.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    confs = boxes.conf.detach().cpu().numpy()

    grouped: dict[str, dict[str, object]] = {}
    for box, class_id, confidence in zip(xyxy, classes, confs):
        label = names.get(int(class_id), str(int(class_id)))
        threshold = thresholds.get(label, 0.50)
        confidence = float(confidence)
        if confidence < threshold:
            continue

        item = grouped.setdefault(
            label,
            {
                "type": "item",
                "label": label,
                "confidence": confidence,
                "threshold": threshold,
                "detection_count": 0,
                "boxes": [],
            },
        )
        item["confidence"] = max(float(item["confidence"]), confidence)
        item["detection_count"] = int(item["detection_count"]) + 1
        if include_boxes:
            item["boxes"].append(
                {
                    "confidence": confidence,
                    "bbox_xyxy": [round(float(value), 3) for value in box],
                }
            )

    output = []
    for item in grouped.values():
        if not include_boxes:
            item.pop("boxes", None)
        output.append(item)
    return sorted(output, key=lambda row: float(row["confidence"]), reverse=True)


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

    thresholds = class_thresholds(args.threshold)
    predict_conf = float(args.min_conf) if args.min_conf is not None else min(thresholds.values())

    model = YOLO(str(weights))
    result = model.predict(
        source=str(image),
        imgsz=args.imgsz,
        conf=predict_conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        verbose=False,
    )[0]

    results = detection_results(result, thresholds, args.include_boxes)
    payload = {
        "image": str(image),
        "weights": str(weights),
        "threshold_profile": "blood/gun/knife/slot_machine=0.35, others=0.50",
        "labels": [str(item["label"]) for item in results],
        "results": results,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
