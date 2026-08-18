from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from flow import RectifiedFlow
from train import train_one_epoch
from utils.ema import ExponentialMovingAverage


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, lr: float) -> None:
        super().__init__(params, lr=lr)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)


class TinyVelocity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        del time
        return self.conv(x)


def test_train_one_epoch_accumulates_before_optimizer_step(tmp_path: Path) -> None:
    model = TinyVelocity()
    flow = RectifiedFlow(model)
    ema = ExponentialMovingAverage(model, decay=0.9)
    loader = DataLoader(torch.randn(6, 1, 4, 4), batch_size=2)
    optimizer = CountingSGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    with SummaryWriter(log_dir=tmp_path) as writer:
        loss, global_step = train_one_epoch(
            flow=flow,
            ema=ema,
            loader=loader,
            optimizer=optimizer,
            scaler=scaler,
            device=torch.device("cpu"),
            use_amp=False,
            gradient_clip=1.0,
            writer=writer,
            global_step=0,
            log_every_steps=1,
            gradient_accumulation_steps=2,
        )

    assert torch.isfinite(torch.tensor(loss))
    assert optimizer.step_count == 2
    assert global_step == 2
