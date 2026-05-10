#!/usr/bin/env python
"""Evaluate the RetinaFace + ArcFace recognizer and write per-person metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


def add_candidate_deps() -> None:
    for parent in Path(__file__).resolve().parents:
        deps = parent / ".deps"
        if deps.exists():
            sys.path.insert(0, str(deps))


add_candidate_deps()

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from face_arcface_common import ArcFaceBackbone, eval_transform, load_checkpoint, read_label_map, project_root, save_json


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root, help="yoloidentify project root.")
    parser.add_argument("--data", type=Path, default=root / "data" / "识别人物" / "pubfig_retinaface_arcface", help="Prepared face crop dataset.")
    parser.add_argument("--weights", type=Path, default=root / "weights" / "识别人物" / "arcface_retinaface_best.pt", help="ArcFace checkpoint.")
    parser.add_argument("--out", type=Path, default=root / "weights" / "识别人物" / "arcface_retinaface_per_class_metrics.csv", help="Output CSV path.")
    parser.add_argument("--summary", type=Path, default=root / "weights" / "识别人物" / "arcface_retinaface_eval_summary.json", help="Output summary JSON path.")
    parser.add_argument("--batch", type=int, default=128, help="Batch size.")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--threshold", type=float, default=-1.0, help="Cosine threshold; below it is treated as unknown.")
    parser.add_argument("--device", default=None, help="Device: 0, cuda, or cpu.")
    return parser.parse_args()


@torch.no_grad()
def predict_dataset(model: ArcFaceBackbone, centroids: torch.Tensor, dataset: ImageFolder, device: torch.device, batch: int, workers: int, threshold: float):
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available())
    true_labels: list[int] = []
    pred_labels: list[int] = []
    scores: list[float] = []
    centroids = centroids.to(device)
    model.eval()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        embeddings = model(images)
        logits = embeddings @ centroids.T
        best_scores, preds = torch.max(logits, dim=1)
        for label, pred, score in zip(labels.tolist(), preds.cpu().tolist(), best_scores.cpu().tolist()):
            true_labels.append(int(label))
            pred_labels.append(int(pred) if score >= threshold else -1)
            scores.append(float(score))
    return true_labels, pred_labels, scores


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    data = args.data.resolve()
    checkpoint = load_checkpoint(args.weights.resolve(), "cpu")

    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args.device.isdigit():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device(args.device)

    dataset = ImageFolder(data / "test", transform=eval_transform())
    model = ArcFaceBackbone(embedding_dim=int(checkpoint["embedding_dim"]), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    centroids = checkpoint["centroids"].float()
    classes = list(checkpoint["classes"])

    if dataset.classes != classes:
        raise SystemExit("Test dataset classes do not match the checkpoint classes.")

    _, safe_to_people = read_label_map(root / "data" / "识别人物" / "pubfig" / "metadata" / "label_map.csv")
    true_labels, pred_labels, scores = predict_dataset(model, centroids, dataset, device, args.batch, args.workers, args.threshold)

    support = Counter(true_labels)
    predicted_count = Counter(label for label in pred_labels if label >= 0)
    true_positive = Counter(t for t, p in zip(true_labels, pred_labels) if t == p)
    total = len(true_labels)
    correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    unknown = sum(1 for p in pred_labels if p < 0)

    rows = []
    for class_id, safe_label in enumerate(classes):
        tp = true_positive[class_id]
        fp = predicted_count[class_id] - tp
        fn = support[class_id] - tp
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        accuracy = 0.0 if support[class_id] == 0 else tp / support[class_id]
        class_scores = [score for score, true_label in zip(scores, true_labels) if true_label == class_id]
        rows.append(
            {
                "class_id": class_id,
                "person": safe_to_people.get(safe_label, safe_label),
                "safe_label": safe_label,
                "support": support[class_id],
                "predicted": predicted_count[class_id],
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "accuracy": f"{accuracy:.6f}",
                "precision": f"{precision:.6f}",
                "recall": f"{recall:.6f}",
                "f1": f"{f1:.6f}",
                "mean_true_score": f"{(sum(class_scores) / len(class_scores)) if class_scores else 0.0:.6f}",
            }
        )

    fieldnames = [
        "class_id",
        "person",
        "safe_label",
        "support",
        "predicted",
        "true_positive",
        "false_positive",
        "false_negative",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "mean_true_score",
    ]
    with args.out.resolve().open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "root": str(root),
        "data": str(data),
        "weights": str(args.weights.resolve()),
        "csv": str(args.out.resolve()),
        "classes": len(classes),
        "test_images": total,
        "correct": correct,
        "unknown": unknown,
        "overall_accuracy": 0.0 if total == 0 else correct / total,
        "macro_precision": sum(float(row["precision"]) for row in rows) / max(len(rows), 1),
        "macro_recall": sum(float(row["recall"]) for row in rows) / max(len(rows), 1),
        "macro_f1": sum(float(row["f1"]) for row in rows) / max(len(rows), 1),
        "threshold": args.threshold,
        "device": str(device),
    }
    save_json(args.summary.resolve(), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
