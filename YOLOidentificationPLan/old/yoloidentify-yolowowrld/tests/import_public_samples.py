"""
模块名称：import_public_samples
作用：从公开数据源少量导入测试图片到 `samples/<category>/`，并在受限时输出获取方式。
使用方法：
  - SOURCE=kaggle DATASET_SLUG=jubaerad/weapons-in-images-segmented-videos CATEGORY=terrorism TAKE=20 python -m tests.import_public_samples
  - SOURCE=github_hate_symbols CATEGORY=nazi_symbol TAKE=20 python -m tests.import_public_samples
  - SOURCE=url_list URLS_FILE=./urls.txt CATEGORY=gambling TAKE=10 python -m tests.import_public_samples
"""

from __future__ import annotations

import os
import random
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
GITHUB_HATE_SYMBOLS_ZIP = "https://github.com/OpnBrdrsAdvct/hate-symbols/archive/refs/heads/main.zip"

PUBLIC_SOURCE_NOTES = {
    "terrorism": [
        "Kaggle: jubaerad/weapons-in-images-segmented-videos",
        "Kaggle: sissasank/guns-object-detection",
        "images.cv: guns image classification dataset, review license before use",
    ],
    "nazi_symbol": [
        "GitHub: OpnBrdrsAdvct/hate-symbols, contains uncensored hateful/NSFW imagery",
        "Roboflow Universe: zhiwei/nazi-symbols, may require Roboflow account/API key",
    ],
    "protest": [
        "GDELT VGKG protest image annotations provide URLs, not bundled images",
        "Protest Image Dataset provides metadata publicly; image archive usually requires request",
    ],
    "fallen_official": [
        "LFW/PubFig can validate face pipeline, but real sensitive person DB must be prepared privately",
    ],
    "disgraced_artist": [
        "LFW/PubFig can validate face pipeline, but real sensitive person DB must be prepared privately",
    ],
}


@dataclass(frozen=True)
class ImportConfig:
    source: str
    category: str
    take: int
    shuffle: bool
    seed: int
    out_dir: Path
    dataset_slug: str
    subdir: str
    prefix: str
    urls_file: Path | None


def _default_samples_dir(category: str) -> Path:
    return Path(__file__).resolve().parents[1] / "samples" / category


def _read_config() -> ImportConfig:
    category = os.environ.get("CATEGORY", "terrorism").strip()
    source = os.environ.get("SOURCE", "notes").strip().lower()
    take = int(os.environ.get("TAKE", "20"))
    if take <= 0:
        raise ValueError("TAKE must be > 0")

    out_dir = (
        Path(os.environ["OUT_DIR"]).expanduser().resolve()
        if os.environ.get("OUT_DIR")
        else _default_samples_dir(category)
    )
    urls_file = Path(os.environ["URLS_FILE"]).expanduser().resolve() if os.environ.get("URLS_FILE") else None
    return ImportConfig(
        source=source,
        category=category,
        take=take,
        shuffle=os.environ.get("SHUFFLE", "1").strip().lower() in {"1", "true", "yes"},
        seed=int(os.environ.get("SEED", "42")),
        out_dir=out_dir,
        dataset_slug=os.environ.get("DATASET_SLUG", "").strip(),
        subdir=os.environ.get("SUBDIR", "").strip().strip("/\\"),
        prefix=os.environ.get("PREFIX", category).strip() or category,
        urls_file=urls_file,
    )


def _list_images(root_dir: Path) -> list[Path]:
    images = [p for p in root_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    images.sort(key=lambda item: item.as_posix())
    return images


def _pick(items: list[Path], cfg: ImportConfig) -> list[Path]:
    items = list(items)
    if cfg.shuffle:
        rng = random.Random(cfg.seed)
        rng.shuffle(items)
    return items[: min(cfg.take, len(items))]


def _copy_images(images: list[Path], cfg: ImportConfig) -> int:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in images:
        dst = cfg.out_dir / f"{cfg.prefix}_{src.name}"
        if dst.exists():
            dst = cfg.out_dir / f"{cfg.prefix}_{copied + 1}_{src.name}"
        shutil.copy2(src, dst)
        copied += 1
    return copied


def _import_kaggle(cfg: ImportConfig) -> int:
    if not cfg.dataset_slug:
        raise ValueError("SOURCE=kaggle requires DATASET_SLUG.")
    try:
        import kagglehub  # type: ignore
    except Exception as exc:
        raise RuntimeError("Install kagglehub and configure Kaggle credentials before SOURCE=kaggle.") from exc

    dataset_root = Path(kagglehub.dataset_download(cfg.dataset_slug)).resolve()
    search_root = dataset_root / cfg.subdir if cfg.subdir else dataset_root
    if not search_root.exists():
        raise FileNotFoundError(f"SUBDIR not found under dataset: {search_root}")
    images = _pick(_list_images(search_root), cfg)
    copied = _copy_images(images, cfg)
    print(f"dataset_root={dataset_root}")
    print(f"search_root={search_root}")
    return copied


def _import_github_hate_symbols(cfg: ImportConfig) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "hate-symbols.zip"
        extract_dir = Path(tmp) / "hate-symbols"
        urllib.request.urlretrieve(GITHUB_HATE_SYMBOLS_ZIP, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        images = _pick(_list_images(extract_dir), cfg)
        return _copy_images(images, cfg)


def _import_url_list(cfg: ImportConfig) -> int:
    if cfg.urls_file is None or not cfg.urls_file.exists():
        raise FileNotFoundError("SOURCE=url_list requires URLS_FILE.")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    urls = [line.strip() for line in cfg.urls_file.read_text(encoding="utf-8").splitlines()]
    urls = [line for line in urls if line and not line.startswith("#")]
    if cfg.shuffle:
        rng = random.Random(cfg.seed)
        rng.shuffle(urls)

    copied = 0
    for url in urls[: cfg.take]:
        suffix = Path(url.split("?")[0]).suffix.lower()
        if suffix not in IMG_EXTS:
            suffix = ".jpg"
        dst = cfg.out_dir / f"{cfg.prefix}_{copied + 1}{suffix}"
        urllib.request.urlretrieve(url, dst)
        copied += 1
    return copied


def _print_notes(category: str) -> None:
    print("=== public sample source notes ===")
    notes = PUBLIC_SOURCE_NOTES.get(category, [])
    if not notes:
        print(f"No curated source notes for category={category}. Use SOURCE=url_list with your own URLS_FILE.")
        return
    for item in notes:
        print(f"- {item}")


def main() -> None:
    cfg = _read_config()
    print("=== import_public_samples ===")
    print(f"source={cfg.source} category={cfg.category} take={cfg.take} out_dir={cfg.out_dir}")

    if cfg.source == "notes":
        _print_notes(cfg.category)
        return
    if cfg.source == "kaggle":
        copied = _import_kaggle(cfg)
    elif cfg.source == "github_hate_symbols":
        copied = _import_github_hate_symbols(cfg)
    elif cfg.source == "url_list":
        copied = _import_url_list(cfg)
    else:
        raise ValueError(f"Unsupported SOURCE: {cfg.source}")

    print(f"copied={copied}")
    if copied == 0:
        _print_notes(cfg.category)


if __name__ == "__main__":
    main()
