import torch
from torch import nn

from utils.ema import ExponentialMovingAverage


def test_ema_updates_parameters_with_configured_decay() -> None:
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    ema = ExponentialMovingAverage(model, decay=0.5)

    with torch.no_grad():
        model.weight.fill_(2.0)
    ema.update(model)

    assert torch.equal(ema.model.weight, torch.ones_like(ema.model.weight))
    assert not ema.model.weight.requires_grad


def test_ema_rejects_invalid_decay() -> None:
    try:
        ExponentialMovingAverage(nn.Linear(1, 1), decay=1.0)
    except ValueError as error:
        assert "decay" in str(error)
    else:
        raise AssertionError("无效 EMA decay 应抛出 ValueError")
