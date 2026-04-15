"""
模块名称：benchmark_samples
作用：遍历 `yoloidentify/samples/<tag>/*` 调用 `/audit/image`，统计总耗时/平均耗时/图片准确率。
使用方法：
  - 直接运行：python -m tests.benchmark_samples
  - 可选参数：SAMPLES_DIR=/abs/path/to/samples python -m tests.benchmark_samples
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


@dataclass(frozen=True)
class OneResult:
    path: Path
    tag: str
    pred: str
    elapsed_s: float
    ok: bool


def _default_samples_dir() -> Path:
    # repo/yoloidentify/tests/benchmark_samples.py -> repo/yoloidentify/samples
    return Path(__file__).resolve().parents[1] / "samples"


def _iter_sample_images(samples_dir: Path) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    for child in sorted(samples_dir.iterdir()):
        if not child.is_dir():
            continue
        tag = child.name
        for p in sorted(child.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                continue
            items.append((p, tag))
    return items


def _post_image(client: TestClient, image_path: Path) -> dict:
    with image_path.open("rb") as f:
        files = {"file": (image_path.name, f, "image/jpeg")}
        resp = client.post("/audit/image", files=files)
    resp.raise_for_status()
    return resp.json()


def run_benchmark(samples_dir: Path) -> list[OneResult]:
    images = _iter_sample_images(samples_dir)
    if not images:
        raise RuntimeError(f"No sample images found under: {samples_dir}")

    results: list[OneResult] = []
    with TestClient(app) as client:
        for image_path, tag in images:
            t0 = time.perf_counter()
            data = _post_image(client, image_path)
            elapsed = time.perf_counter() - t0

            pred = (
                (data.get("coarse_result") or {}).get("category_name")
                or (data.get("coarse_result") or {}).get("labels", [None])[0]
                or ""
            )
            ok = pred == tag
            results.append(OneResult(path=image_path, tag=tag, pred=pred, elapsed_s=elapsed, ok=ok))
    return results


def _print_report(results: list[OneResult]) -> None:
    total = len(results)
    total_s = sum(r.elapsed_s for r in results)
    avg_s = total_s / total if total else 0.0
    acc = sum(1 for r in results if r.ok) / total if total else 0.0

    print("=== benchmark_samples report ===")
    print(f"total_images={total}")
    print(f"total_time_s={total_s:.4f}")
    print(f"avg_time_s={avg_s:.4f}")
    print(f"accuracy={acc:.4%}")

    per_tag_total: dict[str, int] = defaultdict(int)
    per_tag_ok: dict[str, int] = defaultdict(int)
    per_tag_time: dict[str, float] = defaultdict(float)
    for r in results:
        per_tag_total[r.tag] += 1
        per_tag_ok[r.tag] += 1 if r.ok else 0
        per_tag_time[r.tag] += r.elapsed_s

    print("\n-- per-tag --")
    for tag in sorted(per_tag_total.keys()):
        n = per_tag_total[tag]
        ok = per_tag_ok[tag]
        t = per_tag_time[tag]
        print(f"{tag:>12}  n={n:<4}  acc={ok/n:.2%}  avg_s={t/n:.4f}  total_s={t:.4f}")

    bad = [r for r in results if not r.ok]
    if bad:
        print("\n-- mismatches (first 20) --")
        for r in bad[:20]:
            rel = r.path.as_posix()
            print(f"tag={r.tag:<12} pred={r.pred:<12} t={r.elapsed_s:.4f}s  path={rel}")


def main() -> None:
    samples_dir = Path(os.environ.get("SAMPLES_DIR", "")).expanduser().resolve() if os.environ.get("SAMPLES_DIR") else _default_samples_dir()
    results = run_benchmark(samples_dir=samples_dir)
    _print_report(results)


if __name__ == "__main__":
    main()

