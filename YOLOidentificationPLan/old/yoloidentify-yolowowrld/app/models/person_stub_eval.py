from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterable


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models.person_stub import FaceBank, PersonStubJudge


DEFAULT_FACEBANK_CANDIDATES = (
    PROJECT_ROOT / "data" / "celebrity_face_db" / "facebank" / "person_facebank_full.npz",
    PROJECT_ROOT / "data" / "celebrity_face_db" / "facebank" / "person_facebank.npz",
)


def resolve_default_facebank_path() -> Path:
    for candidate in DEFAULT_FACEBANK_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_FACEBANK_CANDIDATES[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate or run image prediction with PersonStubJudge."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate recognition on wiki images already included in the facebank.",
    )
    evaluate_parser.add_argument(
        "--facebank",
        default=str(resolve_default_facebank_path()),
        help="Path to the facebank .npz file.",
    )
    evaluate_parser.add_argument(
        "--source",
        default="wiki",
        help="Only evaluate samples from this source label.",
    )
    evaluate_parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Number of images to evaluate after sampling. Use 0 for all.",
    )
    evaluate_parser.add_argument(
        "--seed",
        type=int,
        default=20260419,
        help="Random seed used when sampling evaluation images.",
    )
    evaluate_parser.add_argument(
        "--detector-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(640, 640),
        help="RetinaFace detector input size.",
    )
    evaluate_parser.add_argument(
        "--max-errors",
        type=int,
        default=20,
        help="Number of mismatch/error examples to retain in the report.",
    )
    evaluate_parser.add_argument(
        "--output",
        default="",
        help="Optional JSON path to save the evaluation report.",
    )

    predict_parser = subparsers.add_parser(
        "predict",
        help="Run person recognition on one image and print the returned name list.",
    )
    predict_parser.add_argument("image", help="Path to the image to test.")
    predict_parser.add_argument(
        "--facebank",
        default=str(resolve_default_facebank_path()),
        help="Path to the facebank .npz file.",
    )
    predict_parser.add_argument(
        "--detector-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(640, 640),
        help="RetinaFace detector input size.",
    )
    return parser


def emit_json(payload: object, *, indent: int | None = None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=indent) + "\n"
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    sys.stdout.flush()


def iter_facebank_samples(facebank: FaceBank, source: str) -> Iterable[dict[str, str]]:
    source_normalized = source.strip().lower()
    for sample_path, sample_source, label_id in zip(
        facebank.sample_paths,
        facebank.sample_sources,
        facebank.label_ids.tolist(),
    ):
        if source_normalized and str(sample_source).strip().lower() != source_normalized:
            continue
        yield {
            "image_path": str(sample_path),
            "expected_name": facebank.label_names[int(label_id)],
            "source": str(sample_source),
        }


def sample_records(records: list[dict[str, str]], limit: int, seed: int) -> list[dict[str, str]]:
    if limit <= 0 or limit >= len(records):
        return records
    rng = random.Random(seed)
    return rng.sample(records, limit)


def evaluate(args: argparse.Namespace) -> int:
    facebank_path = Path(args.facebank).expanduser().resolve()
    if not facebank_path.exists():
        raise FileNotFoundError(f"Facebank not found: {facebank_path}")

    facebank = FaceBank.load(facebank_path)
    records = list(iter_facebank_samples(facebank, source=args.source))
    if not records:
        raise RuntimeError(
            f"No samples found in {facebank_path} for source={args.source!r}."
        )

    selected_records = sample_records(records, limit=max(0, int(args.limit)), seed=int(args.seed))
    judge = PersonStubJudge(
        facebank_path=facebank_path,
        detector_input_size=tuple(args.detector_size),
    )

    contains_match_count = 0
    exact_match_count = 0
    empty_prediction_count = 0
    multi_prediction_count = 0
    error_count = 0
    mismatch_examples: list[dict[str, object]] = []
    start = time.perf_counter()

    for index, record in enumerate(selected_records, start=1):
        image_path = record["image_path"]
        expected_name = record["expected_name"]
        try:
            predicted_names = judge.predict(image_path)
        except Exception as exc:  # pragma: no cover - diagnostic path
            error_count += 1
            if len(mismatch_examples) < args.max_errors:
                mismatch_examples.append(
                    {
                        "image_path": image_path,
                        "expected_name": expected_name,
                        "predicted_names": None,
                        "error": repr(exc),
                    }
                )
            continue

        if not predicted_names:
            empty_prediction_count += 1
        if len(predicted_names) > 1:
            multi_prediction_count += 1
        if expected_name in predicted_names:
            contains_match_count += 1
        else:
            if len(mismatch_examples) < args.max_errors:
                mismatch_examples.append(
                    {
                        "image_path": image_path,
                        "expected_name": expected_name,
                        "predicted_names": predicted_names,
                        "error": None,
                    }
                )
        if len(predicted_names) == 1 and predicted_names[0] == expected_name:
            exact_match_count += 1

        if index % 200 == 0 or index == len(selected_records):
            elapsed = time.perf_counter() - start
            rate = index / elapsed if elapsed > 0 else 0.0
            emit_json(
                {
                    "progress": index,
                    "total": len(selected_records),
                    "contains_match_rate": round(contains_match_count / index, 6),
                    "exact_match_rate": round(exact_match_count / index, 6),
                    "images_per_second": round(rate, 3),
                }
            )

    elapsed_seconds = time.perf_counter() - start
    total = len(selected_records)
    report = {
        "facebank_path": str(facebank_path),
        "evaluated_source": args.source,
        "available_source_sample_count": len(records),
        "evaluated_sample_count": total,
        "seed": int(args.seed),
        "contains_match_count": contains_match_count,
        "contains_match_rate": contains_match_count / total if total else 0.0,
        "exact_match_count": exact_match_count,
        "exact_match_rate": exact_match_count / total if total else 0.0,
        "empty_prediction_count": empty_prediction_count,
        "empty_prediction_rate": empty_prediction_count / total if total else 0.0,
        "multi_prediction_count": multi_prediction_count,
        "multi_prediction_rate": multi_prediction_count / total if total else 0.0,
        "error_count": error_count,
        "error_rate": error_count / total if total else 0.0,
        "elapsed_seconds": elapsed_seconds,
        "images_per_second": total / elapsed_seconds if elapsed_seconds > 0 else 0.0,
        "mismatch_examples": mismatch_examples,
    }
    emit_json(report, indent=2)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


def predict_single_image(args: argparse.Namespace) -> int:
    facebank_path = Path(args.facebank).expanduser().resolve()
    if not facebank_path.exists():
        raise FileNotFoundError(f"Facebank not found: {facebank_path}")

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    judge = PersonStubJudge(
        facebank_path=facebank_path,
        detector_input_size=tuple(args.detector_size),
    )
    predicted_names = judge.predict(image_path)
    emit_json({"image_path": str(image_path), "predicted_names": predicted_names}, indent=2)
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "evaluate":
        return evaluate(args)
    if args.command == "predict":
        return predict_single_image(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
