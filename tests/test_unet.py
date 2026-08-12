import torch

from models import UNet


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
