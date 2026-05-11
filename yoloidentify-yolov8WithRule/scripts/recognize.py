from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO
from uniface import RetinaFace

# Add yoloidentify scripts to path so we can import face_arcface_common
ROOT = Path(__file__).resolve().parents[1]
YOLOIDENTIFY_ROOT = ROOT.parent / "yoloidentify"
sys.path.insert(0, str(YOLOIDENTIFY_ROOT / "scripts" / "app"))

from face_arcface_common import (
    ArcFaceBackbone,
    align_face,
    eval_transform,
    load_checkpoint,
    pil_from_rgb_array,
    read_label_map,
    DetectedFace,
)

ITEM_WEIGHTS = YOLOIDENTIFY_ROOT / "weights/识别物品/yolov8n_cls_best.pt"
FACE_WEIGHTS = YOLOIDENTIFY_ROOT / "weights/识别人物/arcface_retinaface_best.pt"
POLITICAL_PERSONS_PATH = ROOT / "data/political_persons.txt"
LABEL_MAP_PATH = YOLOIDENTIFY_ROOT / "data/识别人物/pubfig/metadata/label_map.csv"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ITEM_IMGSZ = 224
ITEM_CONF = 0.25
FACE_SCORE_THRESH = 0.70
FACE_COSINE_THRESH = 0.45


def load_name_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip().casefold() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def has_political_person(face_recognition: dict | None, political_person_names: set[str]) -> bool:
    if not face_recognition:
        return False
    return any(
        str(identity.get("name", "")).strip().casefold() in political_person_names
        for identity in face_recognition.get("identities", [])
    )


def map_business_category(
    categories: list[str],
    face_count: int,
    face_recognition: dict | None,
    political_person_names: set[str],
) -> str:
    def has(label: str) -> bool:
        return label in categories

    if has_political_person(face_recognition, political_person_names):
        return "涉政"
    if has("gun") or has("knife") or has("blood") or has("police") :
        return "暴恐"
    if has("syringe") or has("powder"):
        return "违禁毒品"
    if has("swastika") :
        return "纳粹符号"
    if has("poker_card") or has("dice") or has("chip") or has("roulette") or has("slot_machine"):
        return "赌博"
    if has("middle_finger"):
        return "不雅手势"
    if (has("crowd") and face_count >= 1) or has("banner") or has("flag"):
        return "游行集会"
    return "正常"


def uniface_to_detected(face) -> DetectedFace:
    box = tuple(float(x) for x in face.bbox_xyxy)
    landmarks = np.array(face.landmarks, dtype=np.float32) if face.landmarks is not None else None
    return DetectedFace(score=float(face.confidence), box=box, landmarks=landmarks)


class Recognizer:
    def __init__(self, enable_face: bool = False) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.item_classifier = YOLO(str(ITEM_WEIGHTS))
        self.item_names = {int(k): v for k, v in self.item_classifier.names.items()}
        
        self.enable_face = enable_face
        if self.enable_face:
            self.retinaface = RetinaFace()
            checkpoint = load_checkpoint(FACE_WEIGHTS, "cpu")
            self.arcface = ArcFaceBackbone(embedding_dim=int(checkpoint["embedding_dim"]), pretrained=False).to(self.device)
            self.arcface.load_state_dict(checkpoint["model_state"])
            self.arcface.eval()
            self.face_centroids = checkpoint["centroids"].float().to(self.device)
            self.face_classes = list(checkpoint["classes"])
            self.face_transform = eval_transform()
            try:
                _, self.safe_to_people = read_label_map(LABEL_MAP_PATH)
            except FileNotFoundError:
                self.safe_to_people = {}
        
        self.political_person_names = load_name_list(POLITICAL_PERSONS_PATH)

    def predict_image(self, image_path: Path) -> dict:
        started = time.perf_counter()
        
        # 1. Item Classification
        item_result = self.item_classifier.predict(
            source=str(image_path), imgsz=ITEM_IMGSZ, device=self.device, verbose=False
        )[0]
        scores = item_result.probs.data.detach().cpu()
        
        detections = []
        categories = []
        counts = {}
        
        for i, score in enumerate(scores.tolist()):
            if score >= ITEM_CONF:
                label = self.item_names[i]
                categories.append(label)
                counts[label] = counts.get(label, 0) + 1
                detections.append({
                    "label": label,
                    "confidence": float(score),
                    "type": "item"
                })

        # 2. Face Recognition
        face_recognition = None
        face_count = 0
        if self.enable_face:
            image_rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            unifaces = self.retinaface.detect(image_rgb) or []
            faces = [uniface_to_detected(f) for f in unifaces if f.confidence >= FACE_SCORE_THRESH]
            face_count = len(faces)
            
            identities = []
            if faces:
                face_tensors = []
                for face in faces:
                    face_rgb = align_face(image_rgb, face)
                    face_tensors.append(self.face_transform(pil_from_rgb_array(face_rgb)))
                
                batch = torch.stack(face_tensors).to(self.device)
                with torch.no_grad():
                    embeddings = self.arcface(batch)
                    logits = embeddings @ self.face_centroids.T
                    best_scores, preds = torch.max(logits, dim=1)
                
                for face, score, pred in zip(faces, best_scores.cpu().tolist(), preds.cpu().tolist()):
                    safe_label = self.face_classes[int(pred)]
                    person = self.safe_to_people.get(safe_label, safe_label.replace("_", " "))
                    accepted = float(score) >= FACE_COSINE_THRESH
                    
                    name = person if accepted else "普通人"
                    identities.append({
                        "name": name,
                        "matched": accepted,
                        "confidence": float(score),
                        "box": face.box
                    })
                    
                    detections.append({
                        "label": "face",
                        "person": name,
                        "confidence": float(face.score),
                        "face_recognition_confidence": float(score),
                        "box_xyxy": face.box,
                        "type": "face"
                    })
            
            face_recognition = {
                "provider": "RetinaFace + ArcFace",
                "status": "ok",
                "face_count": face_count,
                "identities": identities
            }

        detections.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "image": str(image_path.resolve()),
            "elapsed_seconds": time.perf_counter() - started,
            "device": str(self.device),
            "categories": categories,
            "category_counts": counts,
            "business_category": map_business_category(categories, face_count, face_recognition, self.political_person_names),
            "face_recognition": face_recognition,
            "detections": detections,
        }


def collect_images(source: Path) -> list[Path]:
    return [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.suffix.lower() in IMG_EXTS)


def save_annotated(image_path: Path, detections: list[dict], output_path: Path) -> None:
    with Image.open(image_path).convert("RGB") as image:
        draw = ImageDraw.Draw(image)
        for item in detections:
            if item["type"] == "face":
                x1, y1, x2, y2 = item["box_xyxy"]
                draw.rectangle((x1, y1, x2, y2), outline=(255, 40, 40), width=3)
                label = f'{item["person"]} {item["face_recognition_confidence"]:.2f}'
                draw.text((x1, max(0, y1 - 14)), label, fill=(255, 40, 40))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOv8 recognition.")
    parser.add_argument("source", type=Path, help="Image file or image directory.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument("--save-image", type=Path, help="Optional annotated image path for one image.")
    parser.add_argument("--enable-face", action="store_true", help="Enable pretrained face detection.")
    args = parser.parse_args()

    images = collect_images(args.source)
    if not images:
        raise FileNotFoundError(f"No supported images found: {args.source}")
    if args.save_image and len(images) != 1:
        raise ValueError("--save-image only supports one image.")

    recognizer = Recognizer(enable_face=args.enable_face)
    results = [recognizer.predict_image(path) for path in images]
    payload = results[0] if len(results) == 1 else {"images": results}
    if args.save_image:
        save_annotated(images[0], results[0]["detections"], args.save_image)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
