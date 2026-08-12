from datasets.split_dataset import _allocate_counts


def test_allocate_counts_uses_every_sample() -> None:
    counts = _allocate_counts(16_945, [0.9, 0.05, 0.05])

    assert sum(counts) == 16_945
    assert counts == [15_251, 847, 847]
