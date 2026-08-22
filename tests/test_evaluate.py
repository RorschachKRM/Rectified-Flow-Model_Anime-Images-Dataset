from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader

import evaluate


class _MetricBase:
    def to(self, device: torch.device) -> "_MetricBase":
        del device
        return self


class _FakeFID(_MetricBase):
    def __init__(self, **kwargs: Any) -> None:
        del kwargs
        self.real_count = 0
        self.generated_count = 0

    def update(self, images: torch.Tensor, real: bool) -> None:
        if real:
            self.real_count += images.shape[0]
        else:
            self.generated_count += images.shape[0]

    def compute(self) -> torch.Tensor:
        return torch.tensor(float(self.real_count + self.generated_count))


class _FakeKID(_MetricBase):
    def __init__(self, **kwargs: Any) -> None:
        del kwargs

    def update(self, images: torch.Tensor, real: bool) -> None:
        del images, real

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(0.1), torch.tensor(0.01)


class _IdentityFlow:
    def sample(
        self,
        noise: torch.Tensor,
        num_steps: int,
        solver: str,
    ) -> tuple[torch.Tensor, None]:
        del num_steps, solver
        return noise.clamp(-1.0, 1.0), None


def test_distribution_metrics_generates_configured_count(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(evaluate, "FrechetInceptionDistance", _FakeFID)
    monkeypatch.setattr(evaluate, "KernelInceptionDistance", _FakeKID)
    loader = DataLoader(torch.zeros(3, 3, 4, 4), batch_size=2)
    config = {
        "evaluation": {
            "batch_size": 2,
            "generation_seed": 7,
            "metric_seed": 7,
            "kid_subset_size": 3,
            "kid_subsets": 2,
            "num_generated": 5,
        },
        "model": {"in_channels": 3},
        "data": {"image_size": 4},
        "sampling": {"num_steps": 1, "solver": "euler"},
    }

    fid, kid_mean, kid_std, real_count, generated_count = evaluate._distribution_metrics(
        _IdentityFlow(),
        loader,
        config,
        torch.device("cpu"),
    )

    assert fid == 8.0
    assert kid_mean == pytest.approx(0.1)
    assert kid_std == pytest.approx(0.01)
    assert real_count == 3
    assert generated_count == 5
