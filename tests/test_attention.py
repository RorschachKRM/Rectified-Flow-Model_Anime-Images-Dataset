import torch

from models import AttentionBlock


def test_attention_preserves_shape_and_starts_as_identity() -> None:
    block = AttentionBlock(channels=32, num_heads=4)
    images = torch.randn(2, 32, 8, 8)

    output = block(images)

    assert output.shape == images.shape
    assert torch.equal(output, images)
    assert torch.isfinite(output).all()


def test_attention_rejects_incompatible_head_count() -> None:
    try:
        AttentionBlock(channels=30, num_heads=8)
    except ValueError as error:
        assert "num_heads" in str(error)
    else:
        raise AssertionError("通道数不能被注意力头数整除时应抛出 ValueError")
