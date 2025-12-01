"""Tests for the Muon optimizer with proper e3nn handling."""

import torch
from e3nn import o3

from mace.tools.optimizers import (
    MuonWithAdam,
    build_muon_param_groups,
    collect_e3nn_weight_info,
    compute_e3nn_linear_weight_slices,
)


def test_muon_optimizer_step_reduces_loss():
    """Test that Muon optimizer can reduce loss on a simple linear model."""
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3)
    inputs = torch.randn(8, 4)
    targets = torch.randn(8, 3)

    param_groups = [
        {
            "params": list(model.parameters()),
            "lr": 0.02,
            "weight_decay": 0.0,
        }
    ]
    muon_groups = build_muon_param_groups(
        param_groups,
        model=model,
        momentum=0.95,
        betas=(0.9, 0.95),
        eps=1e-10,
    )
    optimizer = MuonWithAdam(muon_groups, ns_steps=2)
    loss_fn = torch.nn.MSELoss()

    with torch.no_grad():
        baseline_loss = loss_fn(model(inputs), targets).item()

    optimizer.zero_grad()
    loss = loss_fn(model(inputs), targets)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        updated_loss = loss_fn(model(inputs), targets).item()

    assert updated_loss <= baseline_loss


def test_compute_e3nn_linear_weight_slices():
    """Test that weight slices are computed correctly for e3nn Linear."""
    # Create an e3nn Linear with multiple irreps
    irreps_in = o3.Irreps("10x0e + 5x1o")
    irreps_out = o3.Irreps("8x0e + 4x1o")
    linear = o3.Linear(irreps_in, irreps_out)

    slices = compute_e3nn_linear_weight_slices(linear)

    # Should have slices for each path with weights
    assert len(slices) > 0

    # Verify slices cover the entire weight tensor
    total_numel = sum(s.stop - s.start for s, _ in slices)
    assert total_numel == linear.weight.numel()

    # Verify each slice has valid 2D shape
    for slice_obj, (rows, cols) in slices:
        numel = slice_obj.stop - slice_obj.start
        assert rows * cols == numel
        assert rows > 0 and cols > 0


def test_collect_e3nn_weight_info():
    """Test that e3nn weight info is collected correctly from a model."""

    class SimpleE3nnModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = o3.Linear("10x0e", "8x0e")
            self.linear2 = o3.Linear("8x0e", "4x0e")
            self.mlp = torch.nn.Linear(4, 2)

        def forward(self, x):
            return self.mlp(self.linear2(self.linear1(x)))

    model = SimpleE3nnModel()
    weight_info = collect_e3nn_weight_info(model)

    # Should have info for both e3nn Linear layers
    assert len(weight_info) == 2

    # Should have info for linear1 and linear2 weights
    assert model.linear1.weight in weight_info
    assert model.linear2.weight in weight_info

    # MLP weight should NOT be in the info
    assert model.mlp.weight not in weight_info


def test_muon_with_e3nn_linear():
    """Test Muon optimizer with e3nn Linear layers."""
    torch.manual_seed(42)

    class E3nnModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = o3.Linear("10x0e", "8x0e")
            self.readout = torch.nn.Linear(8, 1)

        def forward(self, x):
            return self.readout(self.linear(x))

    model = E3nnModel()
    inputs = torch.randn(16, 10)
    targets = torch.randn(16, 1)

    param_groups = [
        {
            "params": list(model.parameters()),
            "lr": 0.02,
            "weight_decay": 0.0,
        }
    ]
    muon_groups = build_muon_param_groups(
        param_groups,
        model=model,
        momentum=0.95,
        betas=(0.9, 0.95),
        eps=1e-10,
    )
    optimizer = MuonWithAdam(muon_groups, ns_steps=2)
    loss_fn = torch.nn.MSELoss()

    # Verify e3nn linear is handled with slices
    e3nn_groups = [g for g in muon_groups if g.get("slices_e3nn_linear") is not None]
    assert len(e3nn_groups) == 1
    assert len(e3nn_groups[0]["slices_e3nn_linear"]) == 1  # One e3nn Linear

    with torch.no_grad():
        baseline_loss = loss_fn(model(inputs), targets).item()

    # Run a few steps
    for _ in range(5):
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        updated_loss = loss_fn(model(inputs), targets).item()

    assert updated_loss < baseline_loss


def test_muon_separates_param_types():
    """Test that build_muon_param_groups correctly separates parameter types."""
    torch.manual_seed(0)

    class MixedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.e3nn_linear = o3.Linear("8x0e", "4x0e")
            self.regular_linear = torch.nn.Linear(4, 4)
            self.bias_only = torch.nn.Parameter(torch.zeros(4))

        def forward(self, x):
            return self.regular_linear(self.e3nn_linear(x)) + self.bias_only

    model = MixedModel()
    param_groups = [
        {
            "params": list(model.parameters()),
            "lr": 0.01,
            "weight_decay": 0.0,
        }
    ]

    muon_groups = build_muon_param_groups(
        param_groups,
        model=model,
        momentum=0.95,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Count parameters in each category
    e3nn_muon_params = []
    regular_muon_params = []
    adam_params = []

    for g in muon_groups:
        if g.get("use_muon", False):
            if g.get("slices_e3nn_linear") is not None:
                e3nn_muon_params.extend(g["params"])
            else:
                regular_muon_params.extend(g["params"])
        else:
            adam_params.extend(g["params"])

    # e3nn Linear weight should be in e3nn_muon_params
    assert model.e3nn_linear.weight in e3nn_muon_params

    # Regular linear weight should be in regular_muon_params
    assert model.regular_linear.weight in regular_muon_params

    # Bias and 1D params should be in adam_params
    assert model.bias_only in adam_params
    assert model.regular_linear.bias in adam_params


def test_muon_backward_compatibility():
    """Test that Muon still works without model (backward compatibility)."""
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3)
    inputs = torch.randn(8, 4)
    targets = torch.randn(8, 3)

    param_groups = [
        {
            "params": list(model.parameters()),
            "lr": 0.02,
            "weight_decay": 0.0,
        }
    ]

    # Call without model - should still work
    muon_groups = build_muon_param_groups(
        param_groups,
        model=None,  # No model provided
        momentum=0.95,
        betas=(0.9, 0.95),
        eps=1e-10,
    )
    optimizer = MuonWithAdam(muon_groups, ns_steps=2)
    loss_fn = torch.nn.MSELoss()

    optimizer.zero_grad()
    loss = loss_fn(model(inputs), targets)
    loss.backward()
    optimizer.step()  # Should not crash


def test_e3nn_linear_slices_applied_correctly():
    """Test that slices are applied correctly during optimization step."""
    torch.manual_seed(123)

    # Create a simple e3nn linear
    linear = o3.Linear("4x0e + 2x1o", "3x0e + 2x1o")
    slices = compute_e3nn_linear_weight_slices(linear)

    # Verify we have multiple slices (for different irrep paths)
    assert len(slices) >= 2

    # Create model and optimizer
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = linear

        def forward(self, x):
            return self.linear(x)

    model = Model()

    param_groups = [
        {"params": [model.linear.weight], "lr": 0.01, "weight_decay": 0.0}
    ]
    muon_groups = build_muon_param_groups(
        param_groups,
        model=model,
        momentum=0.95,
        betas=(0.9, 0.95),
        eps=1e-10,
    )
    optimizer = MuonWithAdam(muon_groups, ns_steps=5)

    # Create input matching the input irreps dimension
    irreps_in = o3.Irreps("4x0e + 2x1o")
    x = torch.randn(8, irreps_in.dim)
    target = torch.randn(8, model.linear.irreps_out.dim)

    # Take a step
    optimizer.zero_grad()
    out = model(x)
    loss = ((out - target) ** 2).mean()
    loss.backward()

    # Store weights before step
    weight_before = model.linear.weight.clone()

    optimizer.step()

    # Verify weights changed
    assert not torch.allclose(model.linear.weight, weight_before)
