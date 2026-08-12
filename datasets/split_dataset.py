from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from utils.config import PROJECT_ROOT, load_config


SPLIT_NAMES = ("train", "val", "test")


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_image(path: Path, expected_size: int) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.mode != "RGB":
                raise ValueError(f"图片不是 RGB 模式: {path} ({image.mode})")
            if image.size != (expected_size, expected_size):
                raise ValueError(
                    f"图片尺寸不是 {expected_size}x{expected_size}: {path} ({image.size})"
                )
    except (OSError, SyntaxError) as error:
        raise ValueError(f"图片无法读取: {path}") from error


def _relative_project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _allocate_counts(total: int, ratios: list[float]) -> list[int]:
    raw_counts = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw_counts]
    remainder = total - sum(counts)
    order = sorted(
        range(len(ratios)), key=lambda index: raw_counts[index] - counts[index], reverse=True
    )
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def _build_inventory(image_paths: list[Path]) -> dict[str, list[int]]:
    inventory: dict[str, list[int]] = {}
    for path in image_paths:
        stat = path.stat()
        inventory[path.name] = [stat.st_size, stat.st_mtime_ns]
    return inventory


def prepare_splits(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    """校验图片、按内容哈希去重，并生成可复现的数据清单。"""
    data_config = config["data"]
    raw_dir = Path(data_config["raw_dir"])
    split_dir = Path(data_config["split_dir"])
    metadata_path = split_dir / "metadata.json"
    manifest_paths = {name: split_dir / f"{name}.txt" for name in SPLIT_NAMES}

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"找不到原始图片目录: {raw_dir}")

    image_paths = sorted(raw_dir.glob("*.png"), key=lambda path: path.name)
    if not image_paths:
        raise FileNotFoundError(f"目录中没有 PNG 图片: {raw_dir}")
    current_inventory = _build_inventory(image_paths)
    split_settings = {
        "raw_dir": _relative_project_path(raw_dir),
        "image_size": int(data_config["image_size"]),
        "seed": int(data_config["split_seed"]),
        "remove_exact_duplicates": bool(data_config["remove_exact_duplicates"]),
        "ratios": {name: float(data_config[f"{name}_ratio"]) for name in SPLIT_NAMES},
    }
    if not force and metadata_path.is_file() and all(path.is_file() for path in manifest_paths.values()):
        with metadata_path.open("r", encoding="utf-8") as file:
            cached_metadata = json.load(file)
        cached_inventory = cached_metadata.get("inventory")
        if cached_metadata.get("split_settings") == split_settings and isinstance(
            cached_inventory, dict
        ):
            if cached_inventory == current_inventory:
                return cached_metadata

    hash_groups: dict[str, list[Path]] = defaultdict(list)
    for path in image_paths:
        validate_image(path, int(data_config["image_size"]))
        hash_groups[file_sha256(path)].append(path)

    groups = list(hash_groups.values())
    random.Random(int(data_config["split_seed"])).shuffle(groups)
    remove_duplicates = bool(data_config["remove_exact_duplicates"])
    samples = [group[0] for group in groups] if remove_duplicates else groups

    ratios = [float(data_config[f"{name}_ratio"]) for name in SPLIT_NAMES]
    if remove_duplicates:
        counts = _allocate_counts(len(samples), ratios)
        split_values: dict[str, list[Path]] = {}
        start = 0
        for name, count in zip(SPLIT_NAMES, counts):
            split_values[name] = samples[start : start + count]
            start += count
    else:
        # 保留重复图片时也让同一哈希组只进入一个集合，避免数据泄漏。
        targets = _allocate_counts(len(image_paths), ratios)
        split_values = {name: [] for name in SPLIT_NAMES}
        for group in groups:
            available = [
                targets[index] - len(split_values[name]) for index, name in enumerate(SPLIT_NAMES)
            ]
            target_index = max(range(len(SPLIT_NAMES)), key=lambda index: available[index])
            split_values[SPLIT_NAMES[target_index]].extend(group)

    split_dir.mkdir(parents=True, exist_ok=True)
    for name, paths in split_values.items():
        manifest_paths[name].write_text(
            "".join(f"{_relative_project_path(path)}\n" for path in paths), encoding="utf-8"
        )

    duplicate_files = sum(len(group) - 1 for group in groups)
    metadata: dict[str, Any] = {
        "raw_dir": _relative_project_path(raw_dir),
        "seed": int(data_config["split_seed"]),
        "remove_exact_duplicates": remove_duplicates,
        "total_files": len(image_paths),
        "unique_images": len(groups),
        "duplicate_files": duplicate_files,
        "splits": {name: len(paths) for name, paths in split_values.items()},
        "split_settings": split_settings,
        "inventory": current_inventory,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="按图片内容去重并划分数据集")
    parser.add_argument("--config", default="config/default.yaml", help="YAML 配置文件")
    parser.add_argument("--force", action="store_true", help="覆盖已有的数据清单")
    args = parser.parse_args()
    metadata = prepare_splits(load_config(args.config), force=args.force)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
