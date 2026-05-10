from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def add_candidate_deps() -> None:
    for parent in Path(__file__).resolve().parents:
        deps = parent / ".deps"
        if deps.exists():
            sys.path.insert(0, str(deps))


add_candidate_deps()

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
ARC_FACE_SIZE = 112
ARC_FACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def project_root() -> Path:
    script = Path(__file__).resolve()
    for parent in script.parents:
        if (parent / "data").is_dir() and (parent / "weights").is_dir() and (parent / "scripts").is_dir():
            return parent
    return script.parents[2]


def image_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def normalized_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").lower()


def read_label_map(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    people_to_safe: dict[str, str] = {}
    safe_to_people: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            people_to_safe[row["person"]] = row["safe_label"]
            safe_to_people[row["safe_label"]] = row["person"]
    return people_to_safe, safe_to_people


def load_split_manifest(pubfig_root: Path, root: Path) -> dict[str, dict[str, str]]:
    manifest = pubfig_root / "metadata" / "split_manifest.csv"
    rows: dict[str, dict[str, str]] = {}
    if not manifest.exists():
        return rows
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            split_file = Path(row["split_file"].replace("\\", "/"))
            if not split_file.is_absolute():
                split_file = root / split_file
            rows[normalized_path(split_file)] = row
    return rows


def parse_rect(rect: str) -> tuple[float, float, float, float] | None:
    if not rect:
        return None
    try:
        x1, y1, x2, y2 = [float(part.strip()) for part in rect.split(",")]
    except Exception:
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0.0 else inter / denom


@dataclass
class DetectedFace:
    score: float
    box: tuple[float, float, float, float]
    landmarks: np.ndarray | None

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def retinaface_to_detected(face) -> DetectedFace:
    box = (float(face.x), float(face.y), float(face.x + face.width), float(face.y + face.height))
    landmarks = None
    if getattr(face, "landmarks", None):
        landmarks = np.array([[point.x, point.y] for point in face.landmarks], dtype=np.float32)
    return DetectedFace(score=float(face.score), box=box, landmarks=landmarks)


def choose_face(faces: list[DetectedFace], target_rect: tuple[float, float, float, float] | None) -> tuple[DetectedFace | None, float]:
    if not faces:
        return None, 0.0
    if target_rect is not None:
        scored = [(iou(face.box, target_rect), face) for face in faces]
        best_iou, best_face = max(scored, key=lambda item: (item[0], item[1].score, item[1].area))
        if best_iou > 0.01:
            return best_face, best_iou
    return max(faces, key=lambda face: (face.score, face.area)), 0.0


def crop_with_margin(image_rgb: np.ndarray, box: tuple[float, float, float, float], margin: float = 0.20) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    x1 -= box_w * margin
    x2 += box_w * margin
    y1 -= box_h * margin
    y2 += box_h * margin
    x1 = int(max(0, math.floor(x1)))
    y1 = int(max(0, math.floor(y1)))
    x2 = int(min(width, math.ceil(x2)))
    y2 = int(min(height, math.ceil(y2)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("empty crop")
    crop = image_rgb[y1:y2, x1:x2]
    return cv2.resize(crop, (ARC_FACE_SIZE, ARC_FACE_SIZE), interpolation=cv2.INTER_AREA)


def align_face(image_rgb: np.ndarray, face: DetectedFace | None, fallback_rect: tuple[float, float, float, float] | None = None) -> np.ndarray:
    if face is not None and face.landmarks is not None and face.landmarks.shape == (5, 2):
        matrix, _ = cv2.estimateAffinePartial2D(face.landmarks, ARC_FACE_TEMPLATE, method=cv2.LMEDS)
        if matrix is not None:
            return cv2.warpAffine(
                image_rgb,
                matrix,
                (ARC_FACE_SIZE, ARC_FACE_SIZE),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
    if face is not None:
        return crop_with_margin(image_rgb, face.box)
    if fallback_rect is not None:
        return crop_with_margin(image_rgb, fallback_rect)
    raise ValueError("no face or fallback rectangle")


def train_transform(image_size: int = ARC_FACE_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.15, 0.15, 0.15, 0.05)], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def eval_transform(image_size: int = ARC_FACE_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


class ArcFaceBackbone(nn.Module):
    def __init__(self, embedding_dim: int = 256, pretrained: bool = False):
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
            except Exception:
                weights = None
        base = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base.fc.in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.embedding(x)
        return F.normalize(x, dim=1)


class ArcMarginProduct(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 30.0, margin: float = 0.5, easy_margin: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - torch.square(cosine), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.scale


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def pil_from_rgb_array(image_rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(np.ascontiguousarray(image_rgb.astype(np.uint8)), mode="RGB")
