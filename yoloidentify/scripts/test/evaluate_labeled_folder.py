#!/usr/bin/env python
"""Evaluate RetinaFace + ArcFace on an external folder labeled by subdirectory."""

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
from retinaface import detect_faces, read_image

from face_arcface_common import (
    ArcFaceBackbone,
    IMAGE_EXTENSIONS,
    align_face,
    eval_transform,
    load_checkpoint,
    pil_from_rgb_array,
    project_root,
    read_label_map,
    retinaface_to_detected,
    save_json,
)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root, help="yoloidentify project root.")
    parser.add_argument("--dataset", type=Path, required=True, help="External labeled image folder.")
    parser.add_argument("--weights", type=Path, default=root / "weights" / "识别人物" / "arcface_retinaface_best.pt", help="ArcFace checkpoint.")
    parser.add_argument("--metrics-csv", type=Path, default=root / "weights" / "识别人物" / "external_labeled_arcface_metrics.csv", help="Per-label metrics CSV.")
    parser.add_argument("--predictions-csv", type=Path, default=root / "weights" / "识别人物" / "external_labeled_arcface_predictions.csv", help="Per-image predictions CSV.")
    parser.add_argument("--summary", type=Path, default=root / "weights" / "识别人物" / "external_labeled_arcface_summary.json", help="Summary JSON.")
    parser.add_argument("--score-threshold", type=float, default=0.70, help="RetinaFace score threshold.")
    parser.add_argument("--nms-threshold", type=float, default=0.40, help="RetinaFace NMS threshold.")
    parser.add_argument("--threshold", type=float, default=-1.0, help="ArcFace cosine threshold; below it is ignored.")
    parser.add_argument("--device", default=None, help="Device: 0, cuda, or cpu.")
    return parser.parse_args()


def collect_images(dataset: Path) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for label_dir in sorted(path for path in dataset.iterdir() if path.is_dir()):
        for image_path in sorted(label_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((label_dir.name, image_path))
    return items


def resolve_true_person(label: str, people_to_safe: dict[str, str], safe_to_people: dict[str, str]) -> str:
    if label in people_to_safe:
        return label
    if label in safe_to_people:
        return safe_to_people[label]
    return label.replace("_", " ")


@torch.no_grad()
def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    dataset = args.dataset.resolve()
    checkpoint = load_checkpoint(args.weights.resolve(), "cpu")

    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args.device.isdigit():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device(args.device)

    people_to_safe, safe_to_people = read_label_map(root / "data" / "识别人物" / "pubfig" / "metadata" / "label_map.csv")
    model = ArcFaceBackbone(embedding_dim=int(checkpoint["embedding_dim"]), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    centroids = checkpoint["centroids"].float().to(device)
    classes = list(checkpoint["classes"])
    transform = eval_transform()

    images = collect_images(dataset)
    prediction_rows = []
    support = Counter()
    correct_contains = Counter()
    top1_correct = Counter()
    predicted_count = Counter()
    no_face = Counter()
    errors = Counter()

    for folder_label, image_path in images:
        true_person = resolve_true_person(folder_label, people_to_safe, safe_to_people)
        support[true_person] += 1
        people_scores: dict[str, float] = {}
        face_count = 0
        error = ""
        try:
            image_rgb = read_image(str(image_path))
            faces = [retinaface_to_detected(face) for face in detect_faces(image_rgb, score_threshold=args.score_threshold, nms_threshold=args.nms_threshold)]
            face_count = len(faces)
            tensors = []
            for face in faces:
                face_rgb = align_face(image_rgb, face)
                tensors.append(transform(pil_from_rgb_array(face_rgb)))
            if tensors:
                batch = torch.stack(tensors).to(device)
                embeddings = model(batch)
                logits = embeddings @ centroids.T
                best_scores, preds = torch.max(logits, dim=1)
                for score, pred in zip(best_scores.cpu().tolist(), preds.cpu().tolist()):
                    if float(score) >= args.threshold:
                        safe_label = classes[int(pred)]
                        person = safe_to_people.get(safe_label, safe_label)
                        people_scores[person] = max(people_scores.get(person, -1.0), float(score))
            else:
                no_face[true_person] += 1
        except Exception as exc:
            errors[true_person] += 1
            error = repr(exc)

        people = [name for name, _ in sorted(people_scores.items(), key=lambda item: item[1], reverse=True)]
        for person in set(people):
            predicted_count[person] += 1
        is_correct = true_person in people
        if is_correct:
            correct_contains[true_person] += 1
        if people and people[0] == true_person:
            top1_correct[true_person] += 1

        prediction_rows.append(
            {
                "folder_label": folder_label,
                "true_person": true_person,
                "image": str(image_path),
                "face_count": face_count,
                "predicted_people": "|".join(people),
                "top1": people[0] if people else "",
                "correct_contains_true": int(is_correct),
                "top1_correct": int(bool(people and people[0] == true_person)),
                "error": error,
            }
        )

    labels = sorted(support)
    metric_rows = []
    for label in labels:
        tp = correct_contains[label]
        fp = predicted_count[label] - tp
        fn = support[label] - tp
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        metric_rows.append(
            {
                "person": label,
                "support": support[label],
                "predicted": predicted_count[label],
                "correct_contains_true": tp,
                "top1_correct": top1_correct[label],
                "no_face": no_face[label],
                "errors": errors[label],
                "accuracy_contains_true": f"{(tp / support[label]) if support[label] else 0.0:.6f}",
                "top1_accuracy": f"{(top1_correct[label] / support[label]) if support[label] else 0.0:.6f}",
                "precision": f"{precision:.6f}",
                "recall": f"{recall:.6f}",
                "f1": f"{f1:.6f}",
            }
        )

    with args.predictions_csv.resolve().open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "folder_label",
                "true_person",
                "image",
                "face_count",
                "predicted_people",
                "top1",
                "correct_contains_true",
                "top1_correct",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    with args.metrics_csv.resolve().open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "person",
                "support",
                "predicted",
                "correct_contains_true",
                "top1_correct",
                "no_face",
                "errors",
                "accuracy_contains_true",
                "top1_accuracy",
                "precision",
                "recall",
                "f1",
            ],
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    total = len(images)
    total_correct = sum(correct_contains.values())
    total_top1 = sum(top1_correct.values())
    summary = {
        "dataset": str(dataset),
        "weights": str(args.weights.resolve()),
        "metrics_csv": str(args.metrics_csv.resolve()),
        "predictions_csv": str(args.predictions_csv.resolve()),
        "labels": len(labels),
        "images": total,
        "correct_contains_true": total_correct,
        "top1_correct": total_top1,
        "accuracy_contains_true": 0.0 if total == 0 else total_correct / total,
        "top1_accuracy": 0.0 if total == 0 else total_top1 / total,
        "no_face": sum(no_face.values()),
        "errors": sum(errors.values()),
        "threshold": args.threshold,
        "score_threshold": args.score_threshold,
        "device": str(device),
    }
    save_json(args.summary.resolve(), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
