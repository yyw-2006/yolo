#!/usr/bin/env python
"""Detect faces with RetinaFace, recognize them with ArcFace, and output unique names."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_candidate_deps() -> None:
    for parent in Path(__file__).resolve().parents:
        deps = parent / ".deps"
        if deps.exists():
            sys.path.insert(0, str(deps))


add_candidate_deps()

import cv2
import torch
from retinaface import detect_faces, read_image

from face_arcface_common import (
    ArcFaceBackbone,
    align_face,
    eval_transform,
    load_checkpoint,
    pil_from_rgb_array,
    project_root,
    read_label_map,
    retinaface_to_detected,
)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input image path.")
    parser.add_argument("--root", type=Path, default=root, help="yoloidentify project root.")
    parser.add_argument("--weights", type=Path, default=root / "weights" / "识别人物" / "arcface_retinaface_best.pt", help="ArcFace checkpoint.")
    parser.add_argument("--score-threshold", type=float, default=0.70, help="RetinaFace detection threshold.")
    parser.add_argument("--nms-threshold", type=float, default=0.40, help="RetinaFace NMS threshold.")
    parser.add_argument("--threshold", type=float, default=-1.0, help="ArcFace cosine threshold; below it is ignored.")
    parser.add_argument("--device", default=None, help="Device: 0, cuda, or cpu.")
    parser.add_argument("--save", type=Path, default=None, help="Optional annotated image output path.")
    return parser.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    image_path = args.image.resolve()
    checkpoint = load_checkpoint(args.weights.resolve(), "cpu")

    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args.device.isdigit():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device(args.device)

    _, safe_to_people = read_label_map(root / "data" / "识别人物" / "pubfig" / "metadata" / "label_map.csv")
    model = ArcFaceBackbone(embedding_dim=int(checkpoint["embedding_dim"]), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    centroids = checkpoint["centroids"].float().to(device)
    classes = list(checkpoint["classes"])
    transform = eval_transform()

    image_rgb = read_image(str(image_path))
    faces = [retinaface_to_detected(face) for face in detect_faces(image_rgb, score_threshold=args.score_threshold, nms_threshold=args.nms_threshold)]

    detections = []
    people_scores: dict[str, float] = {}
    face_tensors = []
    for face in faces:
        face_rgb = align_face(image_rgb, face)
        face_tensors.append(transform(pil_from_rgb_array(face_rgb)))

    if face_tensors:
        batch = torch.stack(face_tensors).to(device)
        embeddings = model(batch)
        logits = embeddings @ centroids.T
        best_scores, preds = torch.max(logits, dim=1)
        for face, score, pred in zip(faces, best_scores.cpu().tolist(), preds.cpu().tolist()):
            safe_label = classes[int(pred)]
            person = safe_to_people.get(safe_label, safe_label)
            accepted = float(score) >= args.threshold
            if accepted:
                people_scores[person] = max(people_scores.get(person, -1.0), float(score))
            detections.append(
                {
                    "person": person if accepted else "",
                    "best_person": person,
                    "confidence": float(score),
                    "accepted": accepted,
                    "box": [float(v) for v in face.box],
                    "face_score": float(face.score),
                }
            )

    if args.save is not None:
        annotated = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        for item in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in item["box"]]
            color = (0, 200, 0) if item["accepted"] else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = item["person"] or item["best_person"]
            cv2.putText(annotated, f"{label} {item['confidence']:.2f}", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        args.save.resolve().parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save.resolve()), annotated)

    people = [name for name, _ in sorted(people_scores.items(), key=lambda item: item[1], reverse=True)]
    payload = {
        "image": str(image_path),
        "weights": str(args.weights.resolve()),
        "people": people,
        "detections": detections,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
