from pathlib import Path

from PIL import Image, ImageDraw

from datasets.split_dataset import _allocate_counts, _cluster_near_duplicates


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
