import shutil
from copy import deepcopy
from pathlib import Path

from PIL import Image, ImageDraw

from datasets.split_dataset import (
    _allocate_counts,
    _cluster_near_duplicates,
    prepare_splits,
)
from utils.config import load_config


def test_allocate_counts_uses_every_sample() -> None:
    counts = _allocate_counts(16_945, [0.9, 0.05, 0.05])

    assert sum(counts) == 16_945
    assert counts == [15_251, 847, 847]


def test_perceptual_hash_groups_reencoded_image(tmp_path: Path) -> None:
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    image = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((12, 8, 52, 56), fill="black")
    image.save(first_path, compress_level=0)
    image.save(second_path, compress_level=9)

    groups, matches = _cluster_near_duplicates(
        [[first_path], [second_path]],
        hash_size=8,
        threshold=4,
        crop_ratios=[0.9],
    )

    assert len(groups) == 1
    assert matches[0]["hamming_distance"] <= 4


def test_prepare_splits_combines_sources_filters_small_images_and_exact_deduplicates(
    tmp_path: Path,
) -> None:
    first_source = tmp_path / "first"
    second_source = tmp_path / "second"
    first_source.mkdir()
    second_source.mkdir()

    accepted = Image.new("RGB", (64, 64), "red")
    accepted_path = first_source / "accepted.jpg"
    accepted.save(accepted_path, quality=95)
    shutil.copyfile(accepted_path, second_source / "duplicate.jpg")
    Image.new("RGB", (80, 72), "blue").save(second_source / "second.png")
    Image.new("P", (64, 64)).save(first_source / "palette.png")
    Image.new("RGB", (48, 48), "green").save(first_source / "too_small.jpg")

    config = deepcopy(load_config("config/v3_unconditional.yaml"))
    config["data"].update(
        {
            "raw_dirs": [str(first_source), str(second_source)],
            "split_dir": str(tmp_path / "splits"),
            "remove_near_duplicates": False,
        }
    )

    metadata = prepare_splits(config, force=True)

    assert metadata["total_files"] == 5
    assert metadata["accepted_files"] == 4
    assert metadata["rejected_files"] == 1
    assert metadata["exact_duplicate_files"] == 1
    assert metadata["unique_images"] == 3
    assert sum(metadata["splits"].values()) == 3
    report = (tmp_path / "splits" / "dedup_report.json").read_text(encoding="utf-8")
    assert "source_too_small" in report
