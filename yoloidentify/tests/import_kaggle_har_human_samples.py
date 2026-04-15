"""
模块名称：import_kaggle_har_human_samples
作用：从 Kaggle 数据集下载到本地缓存后，从指定子目录抽取 N 张图片复制到本项目样本目录，用于接口基准测试。
使用方法：
  - HAR 数据集（test/ 抽 200 张到 samples/human/）：
      python -m tests.import_kaggle_har_human_samples
  - 枪械数据集（images/ 抽 100 张到 samples/violence/）：
      DATASET_SLUG="sissasank/guns-object-detection" SUBDIR="images" TAKE=100 OUT_DIR=./samples/violence \
        python -m tests.import_kaggle_har_human_samples
  - 常用可选项（随机抽样）：
      SHUFFLE=1 SEED=42
环境要求：
  - 已安装 `kagglehub`，并已配置 Kaggle 凭证（不要把 token 粘贴到代码或聊天里）
"""

from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATASET_SLUG = "meetnagadia/human-action-recognition-har-dataset"
DEFAULT_SUBDIR = "test"
DEFAULT_PREFIX = "har_test"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class ImportConfig:
    dataset_slug: str
    subdir: str
    prefix: str
    take: int
    shuffle: bool
    seed: int
    out_dir: Path


def _repo_samples_human_dir() -> Path:
    # repo/yoloidentify/tests/import_*.py -> repo/yoloidentify/samples/human
    return Path(__file__).resolve().parents[1] / "samples" / "human"

def _repo_samples_violence_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "samples" / "violence"


def _read_config() -> ImportConfig:
    dataset_slug = os.environ.get("DATASET_SLUG", DEFAULT_DATASET_SLUG).strip() or DEFAULT_DATASET_SLUG
    subdir = os.environ.get("SUBDIR", DEFAULT_SUBDIR).strip().strip("/\\") or DEFAULT_SUBDIR
    prefix = os.environ.get("PREFIX", DEFAULT_PREFIX).strip() or DEFAULT_PREFIX

    take = int(os.environ.get("TAKE", "200"))
    shuffle = os.environ.get("SHUFFLE", "0").strip() in {"1", "true", "True", "yes", "YES"}
    seed = int(os.environ.get("SEED", "42"))
    if os.environ.get("OUT_DIR"):
        out_dir = Path(os.environ.get("OUT_DIR", "")).expanduser().resolve()
    else:
        # 默认保持历史行为：HAR -> samples/human
        out_dir = _repo_samples_human_dir() if dataset_slug == DEFAULT_DATASET_SLUG else _repo_samples_violence_dir()
    if take <= 0:
        raise ValueError("TAKE must be > 0")
    return ImportConfig(
        dataset_slug=dataset_slug,
        subdir=subdir,
        prefix=prefix,
        take=take,
        shuffle=shuffle,
        seed=seed,
        out_dir=out_dir,
    )


def _find_subdir(dataset_root: Path, subdir: str) -> Path:
    # 做健壮查找（大小写/嵌套层级都兼容）：优先选图片最多的同名目录
    want = subdir.lower()
    candidates: list[Path] = []
    for p in dataset_root.rglob("*"):
        if p.is_dir() and p.name.lower() == want:
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(f"Cannot find a '{subdir}' directory under dataset root: {dataset_root}")

    def score(d: Path) -> int:
        n = 0
        for f in d.rglob("*"):
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                n += 1
        return n

    candidates.sort(key=score, reverse=True)
    best = candidates[0]
    if score(best) == 0:
        raise FileNotFoundError(f"Found '{subdir}' directory but no images inside: {best}")
    return best


def _list_images(root_dir: Path) -> list[Path]:
    imgs = [p for p in root_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    imgs.sort(key=lambda p: p.as_posix())
    return imgs


def _copy_images(images: list[Path], out_dir: Path, prefix: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in images:
        # 统一命名：<prefix>_<原文件名>，避免和你现有样本冲突
        dst_name = f"{prefix}_{src.name}"
        dst = out_dir / dst_name
        if dst.exists():
            # 去重：追加序号
            stem = dst.stem
            suffix = dst.suffix
            i = 2
            while True:
                cand = out_dir / f"{stem}_{i}{suffix}"
                if not cand.exists():
                    dst = cand
                    break
                i += 1

        shutil.copy2(src, dst)
        copied += 1
    return copied


def main() -> None:
    cfg = _read_config()

    try:
        import kagglehub  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("kagglehub not installed. Run: pip install kagglehub") from e

    dataset_root = Path(kagglehub.dataset_download(cfg.dataset_slug)).resolve()
    sub_dir = _find_subdir(dataset_root, cfg.subdir)
    all_images = _list_images(sub_dir)

    if cfg.shuffle:
        rng = random.Random(cfg.seed)
        rng.shuffle(all_images)

    picked = all_images[: min(cfg.take, len(all_images))]
    copied = _copy_images(picked, cfg.out_dir, cfg.prefix)

    print("=== import_kaggle_har_human_samples ===")
    print(f"dataset_root={dataset_root}")
    print(f"sub_dir={sub_dir}")
    print(f"found_images={len(all_images)}")
    print(f"dataset_slug={cfg.dataset_slug} subdir={cfg.subdir} prefix={cfg.prefix}")
    print(f"take={cfg.take} shuffle={int(cfg.shuffle)} seed={cfg.seed}")
    print(f"out_dir={cfg.out_dir}")
    print(f"copied={copied}")


if __name__ == "__main__":
    main()

