#!/usr/bin/env python
"""Detect faces with RetinaFace and build aligned face crops for ArcFace."""

from __future__ import annotations

import argparse
import csv
import json
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

import cv2
from retinaface import detect_faces, read_image

from face_arcface_common import (
    align_face,
    choose_face,
    image_files,
    load_split_manifest,
    normalized_path,
    parse_rect,
    project_root,
    retinaface_to_detected,
    save_json,
)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root, help="yoloidentify project root.")
    parser.add_argument("--pubfig-root", type=Path, default=root / "data" / "识别人物" / "pubfig", help="Downloaded PubFig dataset root.")
    parser.add_argument("--out", type=Path, default=root / "data" / "识别人物" / "pubfig_retinaface_arcface", help="Output face crop dataset.")
    parser.add_argument("--score-threshold", type=float, default=0.70, help="RetinaFace detection threshold.")
    parser.add_argument("--nms-threshold", type=float, default=0.40, help="RetinaFace NMS threshold.")
    parser.add_argument("--no-fallback-rect", action="store_true", help="Skip images when RetinaFace finds no face instead of using PubFig rect.")
    parser.add_argument("--limit", type=int, default=0, help="Debug limit per split.")
    parser.add_argument("--progress-every", type=int, default=250, help="Progress print interval.")
    return parser.parse_args()


def clean_output(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "test"):
        (out / split).mkdir(parents=True, exist_ok=True)


def make_output_path(out: Path, split: str, label: str, image_path: Path) -> Path:
    return out / split / label / f"{image_path.stem}.jpg"


def process_split(
    split: str,
    source_dir: Path,
    out: Path,
    manifest_rows: dict[str, dict[str, str]],
    score_threshold: float,
    nms_threshold: float,
    allow_fallback_rect: bool,
    limit: int,
    progress_every: int,
) -> tuple[list[dict[str, str]], Counter[str]]:
    rows: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    files = list(image_files(source_dir))
    if limit:
        files = files[:limit]
    start = time.time()

    for index, image_path in enumerate(files, start=1):
        label = image_path.parent.name
        manifest = manifest_rows.get(normalized_path(image_path), {})
        target_rect = parse_rect(manifest.get("rect", ""))
        status = "ok"
        method = "retinaface"
        face_score = 0.0
        face_iou = 0.0
        face_count = 0
        output_path = make_output_path(out, split, label, image_path)

        try:
            image_rgb = read_image(str(image_path))
            faces = [retinaface_to_detected(face) for face in detect_faces(image_rgb, score_threshold=score_threshold, nms_threshold=nms_threshold)]
            face_count = len(faces)
            selected, face_iou = choose_face(faces, target_rect)
            if selected is None and (not allow_fallback_rect or target_rect is None):
                status = "no_face"
                counts[status] += 1
            else:
                if selected is None:
                    method = "fallback_rect"
                else:
                    face_score = selected.score
                face_rgb = align_face(image_rgb, selected, fallback_rect=target_rect)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                counts[method] += 1
        except Exception as exc:
            status = "error"
            counts[status] += 1
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "source_file": str(image_path),
                    "face_file": "",
                    "status": status,
                    "method": "",
                    "face_count": str(face_count),
                    "face_score": f"{face_score:.6f}",
                    "target_iou": f"{face_iou:.6f}",
                    "error": repr(exc),
                }
            )
            continue

        rows.append(
            {
                "split": split,
                "label": label,
                "source_file": str(image_path),
                "face_file": str(output_path) if output_path.exists() else "",
                "status": status,
                "method": method if status == "ok" else "",
                "face_count": str(face_count),
                "face_score": f"{face_score:.6f}",
                "target_iou": f"{face_iou:.6f}",
                "error": "",
            }
        )

        if index % progress_every == 0 or index == len(files):
            elapsed = max(time.time() - start, 1e-6)
            print(
                f"{split}: {index:,}/{len(files):,} "
                f"retinaface={counts['retinaface']:,} "
                f"fallback={counts['fallback_rect']:,} "
                f"no_face={counts['no_face']:,} "
                f"error={counts['error']:,} "
                f"({index / elapsed:.1f}/s)",
                flush=True,
            )

    return rows, counts


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "split",
        "label",
        "source_file",
        "face_file",
        "status",
        "method",
        "face_count",
        "face_score",
        "target_iou",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    pubfig_root = args.pubfig_root.resolve()
    out = args.out.resolve()
    clean_output(out)

    manifest_rows = load_split_manifest(pubfig_root, root)
    all_rows: list[dict[str, str]] = []
    summary: dict[str, object] = {
        "root": str(root),
        "pubfig_root": str(pubfig_root),
        "out": str(out),
        "score_threshold": args.score_threshold,
        "nms_threshold": args.nms_threshold,
        "fallback_rect": not args.no_fallback_rect,
        "splits": {},
    }

    for split in ("train", "test"):
        rows, counts = process_split(
            split=split,
            source_dir=pubfig_root / split,
            out=out,
            manifest_rows=manifest_rows,
            score_threshold=args.score_threshold,
            nms_threshold=args.nms_threshold,
            allow_fallback_rect=not args.no_fallback_rect,
            limit=args.limit,
            progress_every=args.progress_every,
        )
        all_rows.extend(rows)
        ok_count = counts["retinaface"] + counts["fallback_rect"]
        summary["splits"][split] = {
            "source_images": len(rows),
            "face_images": ok_count,
            "retinaface": counts["retinaface"],
            "fallback_rect": counts["fallback_rect"],
            "no_face": counts["no_face"],
            "error": counts["error"],
        }

    write_manifest(out / "face_manifest.csv", all_rows)
    save_json(out / "prepare_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
