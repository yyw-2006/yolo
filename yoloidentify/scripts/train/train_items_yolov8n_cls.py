"""Train a YOLOv8n classification model from data/识别物品/yolov8_classification images.

The repository data layout is:

    data/识别物品/yolov8_classification/<label>/train/*.jpg
    data/识别物品/yolov8_classification/<label>/test/*.jpg

Ultralytics classification expects:

    <prepared>/train/<label>/*.jpg
    <prepared>/val/<label>/*.jpg

This script builds a hard-linked prepared view under .yolo_cls_dataset,
trains yolov8n-cls, and copies the final weights into weights/.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
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


def safe_filename(label: str, index: int, source: Path) -> str:
    stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in source.stem)
    stem = stem.strip("._")[:90] or "image"
    return f"{label}_{index:05d}_{stem}{source.suffix.lower()}"


def hardlink_or_copy(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    try:
        os.link(source, dest)
    except OSError:
        shutil.copy2(source, dest)


def build_ultralytics_view(data_root: Path, prepared_root: Path) -> list[str]:
    """Create train/val folder view and return class names.

    The validation view points at the existing test split. This is only used
    by Ultralytics for validation bookkeeping; training images still come only
    from data/<label>/train.
    """

    if prepared_root.exists():
        shutil.rmtree(prepared_root)
    (prepared_root / "train").mkdir(parents=True)
    (prepared_root / "val").mkdir(parents=True)

    class_names: list[str] = []
    for label_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        label = label_dir.name
        train_images = iter_images(label_dir / "train")
        test_images = iter_images(label_dir / "test")
        if not train_images:
            continue

        class_names.append(label)
        for split_name, images in (("train", train_images), ("val", test_images)):
            target_dir = prepared_root / split_name / label
            target_dir.mkdir(parents=True, exist_ok=True)
            for index, image_path in enumerate(images, start=1):
                hardlink_or_copy(
                    image_path,
                    target_dir / safe_filename(label, index, image_path),
                )

    if not class_names:
        raise RuntimeError(f"No train images found under {data_root}")
    return class_names


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root / "data" / "识别物品" / "yolov8_classification")
    parser.add_argument("--prepared-root", type=Path, default=root / ".prepared" / "识别物品" / "yolov8_classification")
    parser.add_argument("--weights-dir", type=Path, default=root / "weights" / "识别物品")
    parser.add_argument("--model", default="yolov8n-cls.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--run-name", default="yolov8n_cls")
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
    prepared_root = args.prepared_root.resolve()
    weights_dir = args.weights_dir.resolve()
    runs_dir = weights_dir / "runs"

    class_names = build_ultralytics_view(data_root, prepared_root)
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / "class_names.json").write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    model = YOLO(args.model)
    model.train(
        data=str(prepared_root),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        project=str(runs_dir),
        name=args.run_name,
        exist_ok=True,
        verbose=True,
    )

    run_dir = runs_dir / args.run_name
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if best.exists():
        shutil.copy2(best, weights_dir / "yolov8n_cls_best.pt")
    if last.exists():
        shutil.copy2(last, weights_dir / "yolov8n_cls_last.pt")

    print(f"Prepared dataset: {prepared_root}")
    print(f"Run directory: {run_dir}")
    print(f"Best weight: {weights_dir / 'yolov8n_cls_best.pt'}")
    print(f"Last weight: {weights_dir / 'yolov8n_cls_last.pt'}")


if __name__ == "__main__":
    main()
