"""Determinism contract for the PyTorch notebooks (chapter 12, notebooks 7-9).

The neural-network finale introduces PyTorch (the optional ``deep`` extra). For the notebooks'
anchors to hold, torch's CPU results must be reproducible run to run. These tests guard that
contract, and skip cleanly when the ``deep`` extra (torch) is not installed.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _seeded_train() -> "torch.Tensor":
    """Train a tiny seeded net under the determinism contract; return its final parameters."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3))
    x = torch.randn(16, 4)
    y = torch.randint(0, 3, (16,))
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    for _ in range(20):
        opt.zero_grad()
        F.cross_entropy(net(x), y).backward()
        opt.step()
    return torch.cat([p.detach().flatten() for p in net.parameters()])


def test_torch_cpu_forward_runs() -> None:
    """A basic CPU forward pass works — the smoke test for the deep extra."""
    net = torch.nn.Linear(3, 2)
    out = net(torch.zeros(5, 3))
    assert out.shape == (5, 2)


def test_seeded_training_is_reproducible() -> None:
    """Two seeded runs produce bit-identical parameters — the determinism contract."""
    first = _seeded_train()
    second = _seeded_train()
    assert torch.equal(first, second)
