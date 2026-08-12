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


def test_euler_solver_moves_in_positive_time_direction() -> None:
    flow = RectifiedFlow(ConstantVelocity(2.0))
    noise = torch.zeros(2, 3, 4, 4)
    samples, trajectory = flow.sample(noise, num_steps=10, trajectory_frames=3)

    assert torch.allclose(samples, torch.full_like(samples, 2.0), atol=1e-6)
    assert len(trajectory) == 3
    assert torch.equal(trajectory[0], noise)


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
