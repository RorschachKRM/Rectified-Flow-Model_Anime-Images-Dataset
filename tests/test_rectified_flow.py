import torch
from torch import nn

from flow import RectifiedFlow


class ConstantVelocity(nn.Module):
    def __init__(self, velocity: float) -> None:
        super().__init__()
        self.velocity = velocity

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        del time
        return torch.full_like(x, self.velocity)


def test_heun_solver_moves_in_positive_time_direction() -> None:
    flow = RectifiedFlow(ConstantVelocity(2.0))
    noise = torch.zeros(2, 3, 4, 4)
    samples, trajectory = flow.sample(
        noise, num_steps=10, trajectory_frames=3, solver="heun"
    )

    assert torch.allclose(samples, torch.full_like(samples, 2.0), atol=1e-6)
    assert len(trajectory) == 3
    assert torch.equal(trajectory[0], noise)


def test_heun_uses_end_of_step_velocity() -> None:
    class TimeVelocity(nn.Module):
        def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            del x
            return time[:, None, None, None]

    flow = RectifiedFlow(TimeVelocity())
    noise = torch.zeros(1, 1, 1, 1)
    samples, _ = flow.sample(noise, num_steps=10, solver="heun")

    assert torch.allclose(samples, torch.full_like(samples, 0.5), atol=1e-6)


def test_sample_rejects_unknown_solver() -> None:
    flow = RectifiedFlow(ConstantVelocity(1.0))

    try:
        flow.sample(torch.zeros(1, 1, 1, 1), num_steps=1, solver="unknown")
    except ValueError as error:
        assert "solver" in str(error)
    else:
        raise AssertionError("未知求解器应抛出 ValueError")


def test_training_loss_is_a_finite_scalar() -> None:
    class ZeroVelocity(nn.Module):
        def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            del x, time
            return torch.zeros(2, 3, 4, 4)

    flow = RectifiedFlow(ZeroVelocity())
    real = torch.zeros(2, 3, 4, 4)
    loss = flow.training_loss(real)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_training_loss_is_reproducible_with_fixed_generator() -> None:
    class ZeroVelocity(nn.Module):
        def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            del time
            return torch.zeros_like(x)

    flow = RectifiedFlow(ZeroVelocity())
    real = torch.zeros(2, 3, 4, 4)
    first_generator = torch.Generator().manual_seed(2026)
    second_generator = torch.Generator().manual_seed(2026)

    first_loss = flow.training_loss(real, generator=first_generator)
    second_loss = flow.training_loss(real, generator=second_generator)

    assert torch.equal(first_loss, second_loss)
