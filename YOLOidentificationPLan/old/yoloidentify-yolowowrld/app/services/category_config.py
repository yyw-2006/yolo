"""
模块名称：category_config
作用：加载并校验类别/tag 映射配置，供 YOLOWorld 主干和分流链路共用。
使用方法：
  - mapping = load_category_mapping("config/category_mapping.json")
  - matches = mapping.match_tag("gun", 0.8)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VALID_CHAINS = {"normal", "direct", "ocr", "face"}
VALID_DECISIONS = {"safe", "review", "block", "error"}


def normalize_tag(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", normalized)


@dataclass(frozen=True)
class CategoryRule:
    id: int
    name: str
    display_name: str
    chain: str
    priority: int
    decision: str
    threshold: float
    prompts: tuple[str, ...]
    tags: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any], default_threshold: float) -> "CategoryRule":
        name = str(raw.get("name", "")).strip()
        chain = str(raw.get("chain", "")).strip()
        decision = str(raw.get("decision", "")).strip()
        if not name:
            raise ValueError("Category rule missing name.")
        if chain not in VALID_CHAINS:
            raise ValueError(f"Invalid chain for category {name}: {chain}")
        if decision not in VALID_DECISIONS:
            raise ValueError(f"Invalid decision for category {name}: {decision}")

        prompts = tuple(str(item).strip() for item in raw.get("prompts", []) if str(item).strip())
        tags = tuple(str(item).strip() for item in raw.get("tags", []) if str(item).strip())

        return cls(
            id=int(raw["id"]),
            name=name,
            display_name=str(raw.get("display_name") or name),
            chain=chain,
            priority=int(raw.get("priority", raw["id"])),
            decision=decision,
            threshold=float(raw.get("threshold", default_threshold)),
            prompts=prompts,
            tags=tags,
        )


@dataclass(frozen=True)
class CategoryMatch:
    category: CategoryRule
    raw_tag: str
    score: float


@dataclass(frozen=True)
class CategoryMapping:
    normal_category: str
    default_threshold: float
    topk: int
    categories: tuple[CategoryRule, ...]

    @property
    def category_ids(self) -> dict[str, int]:
        return {item.name: item.id for item in self.categories}

    @property
    def category_by_name(self) -> dict[str, CategoryRule]:
        return {item.name: item for item in self.categories}

    @property
    def ordered_categories(self) -> tuple[CategoryRule, ...]:
        return tuple(sorted(self.categories, key=lambda item: (item.priority, item.id)))

    @property
    def yolo_prompts(self) -> list[str]:
        seen: dict[str, None] = {}
        for category in self.ordered_categories:
            if category.chain == "normal":
                continue
            for prompt in category.prompts:
                seen.setdefault(prompt, None)
        return list(seen.keys())

    def get(self, category_name: str) -> CategoryRule:
        try:
            return self.category_by_name[category_name]
        except KeyError as exc:
            raise KeyError(f"Unknown category: {category_name}") from exc

    def match_tag(self, raw_tag: str, score: float) -> list[CategoryMatch]:
        normalized = normalize_tag(raw_tag)
        matches: list[CategoryMatch] = []
        for category in self.ordered_categories:
            if category.chain == "normal" or score < category.threshold:
                continue
            tag_names = [normalize_tag(tag) for tag in category.tags]
            prompt_names = [normalize_tag(prompt) for prompt in category.prompts]
            candidates = set(tag_names + prompt_names)
            if normalized in candidates:
                matches.append(CategoryMatch(category=category, raw_tag=raw_tag, score=float(score)))
                continue
            if any(candidate and candidate in normalized for candidate in candidates):
                matches.append(CategoryMatch(category=category, raw_tag=raw_tag, score=float(score)))
        return matches


def load_category_mapping(path: str | Path) -> CategoryMapping:
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        cwd_path = Path.cwd() / config_path
        project_path = Path(__file__).resolve().parents[2] / config_path
        config_path = cwd_path if cwd_path.exists() else project_path
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    default_threshold = float(raw.get("default_threshold", 0.25))
    categories = tuple(CategoryRule.from_dict(item, default_threshold) for item in raw.get("categories", []))
    if not categories:
        raise ValueError("category_mapping.json must contain at least one category.")

    names = [item.name for item in categories]
    if len(names) != len(set(names)):
        raise ValueError("category_mapping.json contains duplicate category names.")

    ids = [item.id for item in categories]
    if len(ids) != len(set(ids)):
        raise ValueError("category_mapping.json contains duplicate category ids.")

    normal_category = str(raw.get("normal_category", "normal"))
    if normal_category not in names:
        raise ValueError(f"normal_category not found in categories: {normal_category}")

    return CategoryMapping(
        normal_category=normal_category,
        default_threshold=default_threshold,
        topk=int(raw.get("topk", 5)),
        categories=categories,
    )
