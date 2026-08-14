from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import imagehash
from PIL import Image

from utils.config import PROJECT_ROOT, load_config


SPLIT_NAMES = ("train", "val", "test")


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parents = list(range(size))

    def find(self, value: int) -> int:
        while self.parents[value] != value:
            self.parents[value] = self.parents[self.parents[value]]
            value = self.parents[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parents[right_root] = left_root


class _BKTree:
    """使用汉明距离检索接近的整数感知哈希。"""

    def __init__(self) -> None:
        self.root: dict[str, Any] | None = None

    def add(self, hash_value: int, image_index: int) -> None:
        if self.root is None:
            self.root = {"hash": hash_value, "indexes": [image_index], "children": {}}
            return

        node = self.root
        while True:
            distance = (hash_value ^ node["hash"]).bit_count()
            if distance == 0:
                node["indexes"].append(image_index)
                return
            child = node["children"].get(distance)
            if child is None:
                node["children"][distance] = {
                    "hash": hash_value,
                    "indexes": [image_index],
                    "children": {},
                }
                return
            node = child

    def query(self, hash_value: int, max_distance: int) -> set[int]:
        if self.root is None:
            return set()
        matches: set[int] = set()
        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = (hash_value ^ node["hash"]).bit_count()
            if distance <= max_distance:
                matches.update(node["indexes"])
            lower = distance - max_distance
            upper = distance + max_distance
            pending.extend(
                child
                for edge_distance, child in node["children"].items()
                if lower <= edge_distance <= upper
            )
        return matches


def _center_crop(image: Image.Image, ratio: float) -> Image.Image:
    width, height = image.size
    crop_width = max(8, round(width * ratio))
    crop_height = max(8, round(height * ratio))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height)).resize(
        image.size, Image.Resampling.LANCZOS
    )


def _perceptual_hashes(path: Path, hash_size: int, crop_ratios: list[float]) -> tuple[int, ...]:
    with Image.open(path) as source:
        image = source.convert("RGB")
        variants = [image, *(_center_crop(image, ratio) for ratio in crop_ratios)]
        return tuple(int(str(imagehash.phash(variant, hash_size=hash_size)), 16) for variant in variants)


def _cluster_near_duplicates(
    exact_groups: list[list[Path]],
    hash_size: int,
    threshold: int,
    crop_ratios: list[float],
) -> tuple[list[list[Path]], list[dict[str, Any]]]:
    hashes = [
        _perceptual_hashes(group[0], hash_size=hash_size, crop_ratios=crop_ratios)
        for group in exact_groups
    ]
    tree = _BKTree()
    for image_index, variants in enumerate(hashes):
        for hash_value in variants:
            tree.add(hash_value, image_index)

    union_find = _UnionFind(len(exact_groups))
    edge_distances: dict[tuple[int, int], int] = {}
    for image_index, variants in enumerate(hashes):
        candidates: set[int] = set()
        for hash_value in variants:
            candidates.update(tree.query(hash_value, threshold))
        for candidate in candidates:
            if candidate <= image_index:
                continue
            distance = min(
                (left_hash ^ right_hash).bit_count()
                for left_hash in variants
                for right_hash in hashes[candidate]
            )
            if distance <= threshold:
                union_find.union(image_index, candidate)
                edge_distances[(image_index, candidate)] = distance

    clustered_indexes: dict[int, list[int]] = defaultdict(list)
    for image_index in range(len(exact_groups)):
        clustered_indexes[union_find.find(image_index)].append(image_index)

    clustered_groups = [
        [path for image_index in indexes for path in exact_groups[image_index]]
        for indexes in clustered_indexes.values()
    ]
    matches = [
        {
            "left": _relative_project_path(exact_groups[left][0]),
            "right": _relative_project_path(exact_groups[right][0]),
            "hamming_distance": distance,
        }
        for (left, right), distance in sorted(edge_distances.items())
    ]
    return clustered_groups, matches


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
    """校验图片、执行精确和感知去重，并生成可复现的数据清单。"""
    data_config = config["data"]
    raw_dir = Path(data_config["raw_dir"])
    split_dir = Path(data_config["split_dir"])
    metadata_path = split_dir / "metadata.json"
    dedup_report_path = split_dir / "dedup_report.json"
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
        "remove_near_duplicates": bool(data_config["remove_near_duplicates"]),
        "phash_size": int(data_config["phash_size"]),
        "phash_threshold": int(data_config["phash_threshold"]),
        "phash_crop_ratios": [float(value) for value in data_config["phash_crop_ratios"]],
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

    exact_groups = list(hash_groups.values())
    exact_unique_images = len(exact_groups)
    near_matches: list[dict[str, Any]] = []
    if bool(data_config["remove_near_duplicates"]):
        groups, near_matches = _cluster_near_duplicates(
            exact_groups,
            hash_size=int(data_config["phash_size"]),
            threshold=int(data_config["phash_threshold"]),
            crop_ratios=[float(value) for value in data_config["phash_crop_ratios"]],
        )
    else:
        groups = exact_groups
    random.Random(int(data_config["split_seed"])).shuffle(groups)
    remove_duplicates = bool(data_config["remove_exact_duplicates"]) or bool(
        data_config["remove_near_duplicates"]
    )
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

    exact_duplicate_files = len(image_paths) - exact_unique_images
    near_duplicate_files = exact_unique_images - len(groups)
    duplicate_files = exact_duplicate_files + near_duplicate_files
    metadata: dict[str, Any] = {
        "raw_dir": _relative_project_path(raw_dir),
        "seed": int(data_config["split_seed"]),
        "remove_exact_duplicates": bool(data_config["remove_exact_duplicates"]),
        "remove_near_duplicates": bool(data_config["remove_near_duplicates"]),
        "total_files": len(image_paths),
        "unique_images": len(groups),
        "duplicate_files": duplicate_files,
        "exact_unique_images": exact_unique_images,
        "exact_duplicate_files": exact_duplicate_files,
        "near_duplicate_files": near_duplicate_files,
        "splits": {name: len(paths) for name, paths in split_values.items()},
        "split_settings": split_settings,
        "inventory": current_inventory,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dedup_report = {
        "settings": split_settings,
        "summary": {
            "total_files": len(image_paths),
            "exact_duplicate_files": exact_duplicate_files,
            "near_duplicate_files": near_duplicate_files,
            "retained_images": len(groups),
        },
        "near_duplicate_matches": near_matches,
        "clusters": [
            {
                "representative": _relative_project_path(group[0]),
                "excluded": [_relative_project_path(path) for path in group[1:]],
            }
            for group in groups
            if len(group) > 1
        ],
    }
    dedup_report_path.write_text(
        json.dumps(dedup_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
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
