from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from person_recognition import FaceRecognitionService
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DETECTOR_WEIGHTS = ROOT / "weights/detector_19class.pt"
BLOOD_WEIGHTS = ROOT / "weights/blood_classifier.pt"
FACE_WEIGHTS = ROOT / "weights/face_detector.pt"
POLITICAL_PERSONS_PATH = ROOT / "data/political_persons.txt"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DETECT_IMGSZ = FACE_IMGSZ = 640
BLOOD_IMGSZ = 224
GENERAL_CONF, FACE_CONF = 0.25, 0.25
BLOOD_DET_CONF, BLOOD_CLS_CONF, BLOOD_IOU = 0.05, 0.40, 0.50
MAX_DET = 300


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.array([], dtype=float)
    x1, y1 = np.maximum(box[:2], boxes[:, :2]).T
    x2, y2 = np.minimum(box[2:], boxes[:, 2:]).T
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return np.divide(inter, area_a + area_b - inter, out=np.zeros_like(inter), where=(area_a + area_b - inter) > 0)


def nms(items: list[dict], threshold: float) -> list[dict]:
    kept: list[dict] = []
    for item in sorted(items, key=lambda x: x["confidence"], reverse=True):
        box = np.array(item["box_xyxy"], dtype=float)
        old = np.array([x["box_xyxy"] for x in kept], dtype=float)
        if not kept or np.all(iou(box, old) <= threshold):
            kept.append(item)
    return kept


def crop_blood_candidate(image: Image.Image, box: np.ndarray) -> Image.Image:
    w, h = image.size
    x1, y1, x2, y2 = box.astype(float)
    pad = 0.18 * max(1.0, x2 - x1, y2 - y1)
    crop_box = (
        max(0, int(round(x1 - pad))),
        max(0, int(round(y1 - pad))),
        min(w, int(round(x2 + pad))),
        min(h, int(round(y2 + pad))),
    )
    return ImageOps.fit(image.crop(crop_box), (BLOOD_IMGSZ, BLOOD_IMGSZ), method=Image.Resampling.BICUBIC)


def make_detection(label: str, conf: float, box, blood_prob: float | None = None) -> dict:
    return {
        "label": label,
        "confidence": float(conf),
        "box_xyxy": [float(x) for x in box],
        "detector_confidence": float(conf),
        "blood_probability": blood_prob,
    }


def load_name_list(path: Path) -> set[str]:
    return {line.strip().casefold() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def has_political_person(face_recognition: dict | None, political_person_names: set[str]) -> bool:
    if not face_recognition:
        return False
    return any(
        str(identity.get("name", "")).strip().casefold() in political_person_names
        for identity in face_recognition.get("identities", [])
    )


def map_business_category(
    category_counts: dict[str, int],
    face_recognition: dict | None,
    political_person_names: set[str],
) -> str:
    def has(label: str) -> bool:
        return category_counts.get(label, 0) > 0

    face_count = category_counts.get("face", 0)
    if has_political_person(face_recognition, political_person_names):
        return "涉政"
    if has("gun") or has("knife") or has("blood") or has("police") or has("riot_shield"):
        return "暴恐"
    if has("syringe") or has("powder"):
        return "违禁毒品"
    if has("swastika") or has("nazi_symbol"):
        return "纳粹符号"
    if has("poker_card") or has("dice") or has("chip") or has("roulette") or has("slot_machine"):
        return "赌博"
    if has("middle_finger"):
        return "不雅手势"
    if (has("crowd") or face_count >= 8) and (has("flag") or has("banner") or has("signboard")):
        return "游行集会"
    return "正常"


class Recognizer:
    def __init__(self, enable_face: bool = False) -> None:
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.detector = YOLO(str(DETECTOR_WEIGHTS))
        self.blood_classifier = YOLO(str(BLOOD_WEIGHTS))
        self.face_detector = YOLO(str(FACE_WEIGHTS)) if enable_face else None
        self.face_recognition = FaceRecognitionService()
        self.political_person_names = load_name_list(POLITICAL_PERSONS_PATH)
        self.names = {int(k): v for k, v in self.detector.names.items()}
        self.blood_id = next(k for k, v in self.names.items() if v == "blood")
        self.blood_cls_id = next(int(k) for k, v in self.blood_classifier.names.items() if v == "blood")

    def predict_image(self, image_path: Path) -> dict:
        started = time.perf_counter()
        result = self.detector.predict(
            [image_path], imgsz=DETECT_IMGSZ, conf=0.001, device=self.device,
            batch=1, verbose=False, save=False, max_det=MAX_DET,
        )[0]
        detections, blood_candidates = [], []
        if result.boxes is not None and len(result.boxes):
            for cls_id, conf, box in zip(
                result.boxes.cls.cpu().numpy().astype(int),
                result.boxes.conf.cpu().numpy(),
                result.boxes.xyxy.cpu().numpy(),
            ):
                label, conf = self.names[int(cls_id)], float(conf)
                if int(cls_id) == self.blood_id:
                    if conf >= BLOOD_DET_CONF:
                        blood_candidates.append((box.astype(float), conf))
                elif conf >= GENERAL_CONF:
                    detections.append(make_detection(label, conf, box))

        detections += self._classify_blood(image_path, blood_candidates)
        if self.face_detector:
            detections += self._detect_faces(image_path)
        detections.sort(key=lambda x: x["confidence"], reverse=True)

        counts, categories = {}, []
        for item in detections:
            counts[item["label"]] = counts.get(item["label"], 0) + 1
            if item["label"] not in categories:
                categories.append(item["label"])
        face_detections = [item for item in detections if item["label"] == "face"]
        face_recognition = self.face_recognition.recognize(image_path, face_detections) if face_detections else None
        return {
            "image": str(image_path.resolve()),
            "elapsed_seconds": time.perf_counter() - started,
            "device": str(self.device),
            "categories": categories,
            "category_counts": counts,
            "business_category": map_business_category(counts, face_recognition, self.political_person_names),
            "face_recognition": face_recognition,
            "detections": detections,
        }

    def _classify_blood(self, image_path: Path, candidates: list[tuple[np.ndarray, float]]) -> list[dict]:
        if not candidates:
            return []
        with Image.open(image_path) as image:
            crops = [crop_blood_candidate(image.convert("RGB"), box) for box, _ in candidates]
        results = self.blood_classifier.predict(
            crops, imgsz=BLOOD_IMGSZ, device=self.device, batch=min(64, len(crops)),
            verbose=False, save=False,
        )
        detections = []
        for (box, det_conf), result in zip(candidates, results):
            blood_prob = float(result.probs.data[self.blood_cls_id].cpu().item())
            if blood_prob >= BLOOD_CLS_CONF:
                item = make_detection("blood", det_conf * blood_prob, box, blood_prob)
                item["detector_confidence"] = float(det_conf)
                detections.append(item)
        return nms(detections, BLOOD_IOU)

    def _detect_faces(self, image_path: Path) -> list[dict]:
        result = self.face_detector.predict(
            [image_path], imgsz=FACE_IMGSZ, conf=FACE_CONF, device=self.device,
            batch=1, verbose=False, save=False, max_det=MAX_DET,
        )[0]
        if result.boxes is None or not len(result.boxes):
            return []
        return [make_detection("face", conf, box) for conf, box in zip(result.boxes.conf.cpu().numpy(), result.boxes.xyxy.cpu().numpy())]


def collect_images(source: Path) -> list[Path]:
    return [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.suffix.lower() in IMG_EXTS)


def save_annotated(image_path: Path, detections: list[dict], output_path: Path) -> None:
    with Image.open(image_path).convert("RGB") as image:
        draw = ImageDraw.Draw(image)
        for item in detections:
            x1, y1, x2, y2 = item["box_xyxy"]
            draw.rectangle((x1, y1, x2, y2), outline=(255, 40, 40), width=3)
            draw.text((x1, max(0, y1 - 14)), f'{item["label"]} {item["confidence"]:.2f}', fill=(255, 40, 40))
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
