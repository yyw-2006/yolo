"""Run available recognizers on one image and print one combined result list."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def project_root() -> Path:
    script = Path(__file__).resolve()
    for parent in script.parents:
        if (parent / "data").is_dir() and (parent / "weights").is_dir() and (parent / "scripts").is_dir():
            return parent
    return script.parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input image path.")
    parser.add_argument("--mode", choices=["all", "people", "items"], default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--item-topk", type=int, default=5)
    parser.add_argument("--person-threshold", type=float, default=-1.0)
    return parser.parse_args()


def run_json(command: list[str], root: Path) -> tuple[dict, str]:
    completed = subprocess.run(command, cwd=str(root), capture_output=True, text=True, encoding="utf-8")
    if completed.returncode != 0:
        return {}, (completed.stderr or completed.stdout).strip()
    try:
        return json.loads(completed.stdout), ""
    except json.JSONDecodeError as exc:
        return {}, f"Could not parse JSON from {' '.join(command)}: {exc}"


def people_results(payload: dict) -> list[dict[str, object]]:
    scores: dict[str, float] = {}
    for detection in payload.get("detections", []):
        person = detection.get("person")
        if not person:
            continue
        scores[str(person)] = max(scores.get(str(person), -1.0), float(detection.get("confidence", 0.0)))
    return [
        {"type": "person", "label": person, "confidence": score}
        for person, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]


def main() -> int:
    args = parse_args()
    root = project_root()
    app_dir = Path(__file__).resolve().parent
    image = args.image.resolve()
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")

    results: list[dict[str, object]] = []
    errors: list[str] = []

    if args.mode in {"all", "people"}:
        payload, error = run_json(
            [
                sys.executable,
                str(app_dir / "predict_people.py"),
                str(image),
                "--device",
                args.device,
                "--threshold",
                str(args.person_threshold),
            ],
            root,
        )
        if error:
            errors.append(f"people: {error}")
        else:
            results.extend(people_results(payload))

    if args.mode in {"all", "items"}:
        payload, error = run_json(
            [
                sys.executable,
                str(app_dir / "predict_items_yolov8n_cls.py"),
                str(image),
                "--device",
                args.device,
                "--topk",
                str(args.item_topk),
            ],
            root,
        )
        if error:
            errors.append(f"items: {error}")
        else:
            results.extend(payload.get("results", []))

    print(
        json.dumps(
            {
                "image": str(image),
                "mode": args.mode,
                "results": results,
                "errors": errors,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 1 if errors and not results else 0


if __name__ == "__main__":
    raise SystemExit(main())
