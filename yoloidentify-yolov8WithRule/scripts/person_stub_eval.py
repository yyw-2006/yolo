"""
Module: person_stub_eval
Purpose: Provide a small CLI for testing the RetinaFace + ArcFace person recognizer.
Usage: python scripts/person_stub_eval.py predict path/to/image.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from person_recognition import DEFAULT_FACEBANK_PATH, FaceBank, PersonStubJudge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate or run person recognition.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_parser = subparsers.add_parser("predict", help="Recognize people in one image.")
    predict_parser.add_argument("image", type=Path, help="Path to the image to test.")
    predict_parser.add_argument("--facebank", type=Path, default=DEFAULT_FACEBANK_PATH, help="Path to facebank .npz.")
    predict_parser.add_argument(
        "--detector-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(320, 320),
        help="RetinaFace detector input size.",
    )
    return parser


def emit_json(payload: object) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    sys.stdout.flush()


def predict_single_image(args: argparse.Namespace) -> int:
    image_path = args.image.expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    facebank_path = args.facebank.expanduser().resolve()
    if not facebank_path.exists():
        raise FileNotFoundError(f"Facebank not found: {facebank_path}")

    judge = PersonStubJudge(facebank_path=facebank_path, detector_input_size=tuple(args.detector_size))
    identities = judge.recognize_faces(image_path)
    emit_json(
        {
            "image_path": str(image_path),
            "facebank_path": str(facebank_path),
            "facebank_identity_count": len(FaceBank.load(facebank_path).label_names),
            "identities": identities,
            "predicted_names": [item["name"] for item in identities],
        }
    )
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "predict":
        return predict_single_image(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
