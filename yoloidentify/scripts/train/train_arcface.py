#!/usr/bin/env python
"""Train an ArcFace recognizer on RetinaFace-aligned face crops."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path


def add_candidate_deps() -> None:
    for parent in Path(__file__).resolve().parents:
        deps = parent / ".deps"
        if deps.exists():
            sys.path.insert(0, str(deps))


add_candidate_deps()

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from face_arcface_common import ArcFaceBackbone, ArcMarginProduct, eval_transform, project_root, save_json, train_transform


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root, help="yoloidentify project root.")
    parser.add_argument("--data", type=Path, default=root / "data" / "识别人物" / "pubfig_retinaface_arcface", help="Prepared face crop dataset.")
    parser.add_argument("--weights-dir", type=Path, default=root / "weights" / "识别人物", help="Output weights directory.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--batch", type=int, default=64, help="Batch size.")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--embedding-dim", type=int, default=256, help="Embedding dimension.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--scale", type=float, default=30.0, help="ArcFace scale.")
    parser.add_argument("--margin", type=float, default=0.5, help="ArcFace angular margin.")
    parser.add_argument("--device", default=None, help="Device: 0, cuda, or cpu.")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not use ImageNet-pretrained ResNet18.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset: ImageFolder, batch: int, workers: int) -> DataLoader:
    label_counts = Counter(dataset.targets)
    sample_weights = [1.0 / label_counts[target] for target in dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return DataLoader(dataset, batch_size=batch, sampler=sampler, num_workers=workers, pin_memory=torch.cuda.is_available())


@torch.no_grad()
def compute_centroids(model: ArcFaceBackbone, dataset: ImageFolder, device: torch.device, batch: int, workers: int) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available())
    sums = torch.zeros(len(dataset.classes), model.embedding[-1].num_features, device=device)
    counts = torch.zeros(len(dataset.classes), device=device)
    model.eval()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        embeddings = model(images)
        sums.index_add_(0, labels, embeddings)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
    centroids = sums / counts.clamp_min(1.0).unsqueeze(1)
    return torch.nn.functional.normalize(centroids, dim=1).detach().cpu()


def save_checkpoint(
    path: Path,
    model: ArcFaceBackbone,
    margin: ArcMarginProduct,
    dataset: ImageFolder,
    centroids: torch.Tensor,
    args: argparse.Namespace,
    epoch: int,
    train_loss: float,
) -> None:
    idx_to_class = {str(idx): name for idx, name in enumerate(dataset.classes)}
    payload = {
        "model_state": model.state_dict(),
        "arc_margin_state": margin.state_dict(),
        "centroids": centroids,
        "classes": dataset.classes,
        "class_to_idx": dataset.class_to_idx,
        "idx_to_class": idx_to_class,
        "embedding_dim": args.embedding_dim,
        "image_size": 112,
        "epoch": epoch,
        "train_loss": train_loss,
        "arcface": {"scale": args.scale, "margin": args.margin},
    }
    torch.save(payload, path)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    root = args.root.resolve()
    data = args.data.resolve()
    weights_dir = args.weights_dir.resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)

    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args.device.isdigit():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device(args.device)

    train_dataset = ImageFolder(data / "train", transform=train_transform())
    centroid_dataset = ImageFolder(data / "train", transform=eval_transform())
    if len(train_dataset.classes) < 2:
        raise SystemExit("Need at least two classes to train ArcFace.")

    loader = make_loader(train_dataset, args.batch, args.workers)
    model = ArcFaceBackbone(embedding_dim=args.embedding_dim, pretrained=not args.no_pretrained).to(device)
    margin = ArcMarginProduct(args.embedding_dim, len(train_dataset.classes), scale=args.scale, margin=args.margin).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(margin.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float("inf")
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        margin.train()
        running_loss = 0.0
        seen = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            embeddings = model(images)
            logits = margin(embeddings, labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * images.size(0)
            seen += images.size(0)
        scheduler.step()
        epoch_loss = running_loss / max(seen, 1)
        history.append({"epoch": epoch, "train_loss": epoch_loss, "lr": scheduler.get_last_lr()[0]})
        print(f"epoch {epoch:03d}/{args.epochs} train_loss={epoch_loss:.6f} lr={scheduler.get_last_lr()[0]:.8f}", flush=True)

        centroids = compute_centroids(model, centroid_dataset, device, args.batch, args.workers)
        save_checkpoint(weights_dir / "arcface_retinaface_last.pt", model, margin, centroid_dataset, centroids, args, epoch, epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            shutil.copy2(weights_dir / "arcface_retinaface_last.pt", weights_dir / "arcface_retinaface_best.pt")

    summary = {
        "root": str(root),
        "data": str(data),
        "weights_dir": str(weights_dir),
        "best_weight": str(weights_dir / "arcface_retinaface_best.pt"),
        "last_weight": str(weights_dir / "arcface_retinaface_last.pt"),
        "classes": len(train_dataset.classes),
        "train_images": len(train_dataset),
        "epochs": args.epochs,
        "batch": args.batch,
        "embedding_dim": args.embedding_dim,
        "device": str(device),
        "pretrained": not args.no_pretrained,
        "best_train_loss": best_loss,
        "seconds": time.time() - start,
        "history": history,
    }
    save_json(weights_dir / "arcface_train_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
