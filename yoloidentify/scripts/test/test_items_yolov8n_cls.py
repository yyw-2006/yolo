"""Evaluate a trained YOLOv8 classification model on data/识别物品/yolov8_classification test images."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


def iter_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def chunks(items: list[Path], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root / "data" / "识别物品" / "yolov8_classification")
    parser.add_argument("--weights", type=Path, default=root / "weights" / "识别物品" / "yolov8n_cls_best.pt")
    parser.add_argument("--output-csv", type=Path, default=root / "weights" / "识别物品" / "test_metrics.csv")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
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
    data_root = args.data_root.resolve()
    weights = args.weights.resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Weight file not found: {weights}")

    model = YOLO(str(weights))
    model_names = {
        int(k): str(v) for k, v in dict(model.names).items()
    }
    known_labels = [p.name for p in sorted(data_root.iterdir()) if p.is_dir()]

    true_counts: Counter[str] = Counter()
    pred_counts: Counter[str] = Counter()
    true_positive: Counter[str] = Counter()
    rows: list[dict[str, object]] = []

    for label in known_labels:
        images = iter_images(data_root / label / "test")
        if not images:
            rows.append(
                {
                    "label": label,
                    "test_images": 0,
                    "predicted_as_label": 0,
                    "true_positive": 0,
                    "correct_rate_precision": "",
                    "recall": "",
                }
            )
            continue

        true_counts[label] += len(images)
        for batch_paths in chunks(images, args.batch):
            results = model.predict(
                source=[str(p) for p in batch_paths],
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                verbose=False,
            )
            for image_path, result in zip(batch_paths, results):
                pred_idx = int(result.probs.top1)
                pred_label = model_names.get(pred_idx, str(pred_idx))
                pred_counts[pred_label] += 1
                if pred_label == label:
                    true_positive[label] += 1

    for label in known_labels:
        support = true_counts[label]
        predicted = pred_counts[label]
        tp = true_positive[label]
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        rows.append(
            {
                "label": label,
                "test_images": support,
                "predicted_as_label": predicted,
                "true_positive": tp,
                "correct_rate_precision": precision if support else "",
                "recall": recall if support else "",
            }
        )

    total = sum(true_counts.values())
    total_correct = sum(true_positive.values())
    overall_accuracy = total_correct / total if total else 0.0

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "test_images",
                "predicted_as_label",
                "true_positive",
                "correct_rate_precision",
                "recall",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"weights: {weights}")
    print(f"test images: {total}")
    print(f"overall accuracy: {overall_accuracy:.4f}")
    print(f"metrics csv: {args.output_csv.resolve()}")
    print()
    print(f"{'label':<16} {'test':>6} {'pred':>6} {'tp':>6} {'precision':>10} {'recall':>10}")
    for row in rows:
        precision = row["correct_rate_precision"]
        recall = row["recall"]
        precision_text = "" if precision == "" else f"{float(precision):.4f}"
        recall_text = "" if recall == "" else f"{float(recall):.4f}"
        print(
            f"{str(row['label']):<16} {int(row['test_images']):>6} "
            f"{int(row['predicted_as_label']):>6} {int(row['true_positive']):>6} "
            f"{precision_text:>10} {recall_text:>10}"
        )


if __name__ == "__main__":
    main()
