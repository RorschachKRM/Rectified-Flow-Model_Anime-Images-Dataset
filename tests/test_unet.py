import torch

from models import UNet
from models.attention import AttentionBlock
from models.blocks import ResidualBlock


def test_unet_preserves_image_shape() -> None:
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=16,
        channel_multipliers=(1, 2, 4),
        num_res_blocks=1,
        time_embedding_dim=64,
    )
    images = torch.randn(2, 3, 32, 32)
    time = torch.rand(2)

    output = model(images, time)

    assert output.shape == images.shape
    assert torch.isfinite(output).all()


def test_unet_rejects_invalid_time_shape() -> None:
    model = UNet(
        base_channels=16,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        time_embedding_dim=32,
    )
    images = torch.randn(2, 3, 16, 16)
    invalid_time = torch.rand(3)

    try:
        model(images, invalid_time)
    except ValueError as error:
        assert "batch_size" in str(error)
    else:
        raise AssertionError("无效的 time 形状应抛出 ValueError")


def test_residual_block_uses_scale_shift_time_projection() -> None:
    block = ResidualBlock(in_channels=16, out_channels=32, time_dim=64, dropout=0.0)
    images = torch.randn(2, 16, 8, 8)
    time_embedding = torch.randn(2, 64)

    output = block(images, time_embedding)

    assert block.time_projection.out_features == 64
    assert output.shape == (2, 32, 8, 8)
    assert torch.isfinite(output).all()


def test_unet_with_attention_preserves_skip_connections_and_shape() -> None:
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=16,
        channel_multipliers=(1, 2, 4),
        num_res_blocks=1,
        time_embedding_dim=64,
        image_size=32,
        attention_resolutions=(16, 8),
        attention_num_heads=4,
    )
    images = torch.randn(2, 3, 32, 32)
    time = torch.rand(2)

    output = model(images, time)
    attention_blocks = [
        module for module in model.modules() if isinstance(module, AttentionBlock)
    ]

    assert output.shape == images.shape
    assert torch.isfinite(output).all()
    assert len(attention_blocks) == 7


def test_unet_without_attention_keeps_legacy_architecture() -> None:
    model = UNet(
        base_channels=16,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        time_embedding_dim=32,
        image_size=16,
    )

    assert not any(isinstance(module, AttentionBlock) for module in model.modules())
