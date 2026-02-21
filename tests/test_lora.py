from __future__ import annotations

import math
from typing import Callable, List, Tuple

import numpy as np
import pytest
import torch
from e3nn import o3

from mace import data, modules, tools
from mace.data import Configuration
from mace.tools import torch_geometric
from mace.modules.lora import inject_lora, merge_lora_weights

try:
    from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
    from mace.cli.convert_cueq_e3nn import run as run_cueq_to_e3nn
    import cuequivariance_torch as cuet
    CUET_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CUET_AVAILABLE = False
    run_e3nn_to_cueq = None
    run_cueq_to_e3nn = None
    cuet = None


def _random_config() -> Configuration:
    atomic_numbers = np.array([6, 1, 1], dtype=int)
    positions = np.random.normal(scale=0.5, size=(3, 3))
    properties = {
        "energy": np.random.normal(scale=0.1),
        "forces": np.random.normal(scale=0.1, size=(3, 3)),
    }
    prop_weights = {"energy": 1.0, "forces": 1.0}
    return Configuration(
        atomic_numbers=atomic_numbers,
        positions=positions,
        properties=properties,
        property_weights=prop_weights,
        cell=np.eye(3) * 8.0,
        pbc=(True, True, True),
    )


def _build_model() -> Tuple[modules.MACE, tools.AtomicNumberTable]:
    table = tools.AtomicNumberTable([1, 6])
    model = modules.MACE(
        r_max=4.5,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=1,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("16x0e + 16x1o"),
        MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.random.normal(scale=0.1, size=len(table.zs)),
        avg_num_neighbors=6.0,
        atomic_numbers=table.zs,
        correlation=2,
        radial_type="bessel",
    )
    return model, table


def _atomic_data_from_config(
    config: Configuration,
    table: tools.AtomicNumberTable,
    cutoff: float = 4.5,
) -> data.AtomicData:
    return data.AtomicData.from_config(config, z_table=table, cutoff=cutoff)


def _forward_energy_forces(
    model: torch.nn.Module,
    configs: List[Configuration],
    table: tools.AtomicNumberTable,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = [_atomic_data_from_config(cfg, table) for cfg in configs]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=len(dataset),
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(loader))
    outputs = model(batch.to_dict())
    energies = outputs["energy"].detach()
    forces = outputs["forces"].detach()
    return energies, forces


def _randomize_lora_parameters(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                torch.nn.init.normal_(param, mean=0.0, std=0.05)


def _rotation_matrix() -> np.ndarray:
    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis)
    theta = np.random.uniform(0, 2 * math.pi)
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    R = np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)
    return R


def _rotate_config(config: Configuration, R: np.ndarray) -> Configuration:
    return Configuration(
        atomic_numbers=config.atomic_numbers.copy(),
        positions=config.positions @ R.T,
        properties=config.properties.copy(),
        property_weights=config.property_weights.copy(),
        cell=config.cell @ R.T if config.cell is not None else None,
        pbc=config.pbc,
        weight=config.weight,
        config_type=config.config_type,
        head=config.head,
    )


def _translate_config(config: Configuration, shift: np.ndarray) -> Configuration:
    return Configuration(
        atomic_numbers=config.atomic_numbers.copy(),
        positions=config.positions + shift.reshape(1, 3),
        properties=config.properties.copy(),
        property_weights=config.property_weights.copy(),
        cell=config.cell,
        pbc=config.pbc,
        weight=config.weight,
        config_type=config.config_type,
        head=config.head,
    )


def _reflect_config(
    config: Configuration, normal: np.ndarray
) -> Tuple[Configuration, np.ndarray]:
    normal = normal / np.linalg.norm(normal)
    R = np.eye(3) - 2.0 * np.outer(normal, normal)
    reflected = _rotate_config(config, R)
    return reflected, R


@pytest.fixture(name="random_configs")
def _random_configs() -> Tuple[Configuration, Configuration]:
    return _random_config(), _random_config()


@pytest.fixture(name="build_lora_model")
def _build_lora_model_fixture() -> (
    Callable[[int, float, bool], Tuple[modules.MACE, tools.AtomicNumberTable]]
):
    def _builder(
        rank: int = 2,
        alpha: float = 0.5,
        randomize: bool = True,
    ) -> Tuple[modules.MACE, tools.AtomicNumberTable]:
        model, table = _build_model()
        inject_lora(model, rank=rank, alpha=alpha)
        if randomize:
            _randomize_lora_parameters(model)
        return model, table

    return _builder


def test_lora_trainable_parameter_count(build_lora_model) -> None:
    model, _ = build_lora_model(rank=2, alpha=0.5, randomize=True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = sum(p.numel() for name, p in model.named_parameters() if "lora_" in name)
    assert trainable == expected

    non_lora_trainable = [
        name
        for name, p in model.named_parameters()
        if "lora_" not in name and p.requires_grad
    ]
    assert (
        not non_lora_trainable
    ), f"Non-LoRA parameters trainable: {non_lora_trainable}"

    # Ensure LoRA parameters were randomized away from zero
    for name, param in model.named_parameters():
        if "lora_B" in name:
            assert torch.any(
                torch.abs(param) > 0
            ), f"LoRA parameter {name} incorrectly zero"


def test_lora_symmetry_equivariance(build_lora_model, random_configs) -> None:
    model, table = build_lora_model(rank=2, alpha=0.5, randomize=True)
    model.eval()
    base_cfg = random_configs[0]

    energy, forces = _forward_energy_forces(model, [base_cfg], table)
    energy_val = energy.item()
    forces_val = forces.squeeze(0).detach().numpy()

    # Rotation invariance / covariance
    R = _rotation_matrix()
    rotated_cfg = _rotate_config(base_cfg, R)
    energy_rot, forces_rot = _forward_energy_forces(model, [rotated_cfg], table)
    assert np.allclose(energy_rot.item(), energy_val, rtol=1e-6, atol=1e-6)
    assert np.allclose(
        forces_val @ R.T, forces_rot.squeeze(0).detach().numpy(), rtol=1e-5, atol=1e-5
    )

    # Translation invariance
    shift = np.array([0.17, -0.05, 0.08])
    translated_cfg = _translate_config(base_cfg, shift)
    energy_trans, forces_trans = _forward_energy_forces(model, [translated_cfg], table)
    assert np.allclose(energy_trans.item(), energy_val, rtol=1e-6, atol=1e-6)
    assert np.allclose(
        forces_trans.squeeze(0).detach().numpy(), forces_val, rtol=1e-6, atol=1e-6
    )

    # Reflection invariance / covariance
    reflected_cfg, R_reflect = _reflect_config(base_cfg, np.array([1.0, -2.0, 3.0]))
    energy_ref, forces_ref = _forward_energy_forces(model, [reflected_cfg], table)
    assert np.allclose(energy_ref.item(), energy_val, rtol=1e-6, atol=1e-6)
    assert np.allclose(
        forces_val @ R_reflect.T,
        forces_ref.squeeze(0).detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


def test_lora_merge_preserves_outputs(build_lora_model, random_configs) -> None:
    """Test that merging LoRA weights produces identical outputs."""
    model, table = build_lora_model(rank=2, alpha=0.5, randomize=True)
    model.eval()

    # Get outputs before merging
    configs = list(random_configs)
    energy_before, forces_before = _forward_energy_forces(model, configs, table)

    # Merge LoRA weights
    merge_lora_weights(model)
    model.eval()

    # Get outputs after merging
    energy_after, forces_after = _forward_energy_forces(model, configs, table)

    # Outputs should be identical (within numerical precision)
    assert torch.allclose(
        energy_before, energy_after, rtol=1e-5, atol=1e-6
    ), f"Energy mismatch after merge: {energy_before} vs {energy_after}"
    assert torch.allclose(
        forces_before, forces_after, rtol=1e-5, atol=1e-6
    ), f"Forces mismatch after merge: max diff = {(forces_before - forces_after).abs().max()}"


def test_lora_merge_removes_wrappers(build_lora_model) -> None:
    """Test that merging removes LoRA wrapper modules."""
    from mace.modules.lora import LoRADenseLinear, LoRAFCLayer, LoRAO3Linear

    model, _ = build_lora_model(rank=2, alpha=0.5, randomize=True)

    # Count LoRA wrappers before merge
    def count_lora_wrappers(module):
        count = 0
        for child in module.modules():
            if isinstance(child, (LoRADenseLinear, LoRAFCLayer, LoRAO3Linear)):
                count += 1
        return count

    wrappers_before = count_lora_wrappers(model)
    assert wrappers_before > 0, "Model should have LoRA wrappers before merge"

    # Merge
    merge_lora_weights(model)

    # Count LoRA wrappers after merge
    wrappers_after = count_lora_wrappers(model)
    assert (
        wrappers_after == 0
    ), f"Model still has {wrappers_after} LoRA wrappers after merge"


def test_lora_merge_enables_gradients(build_lora_model) -> None:
    """Test that merging re-enables gradients for all parameters."""
    model, _ = build_lora_model(rank=2, alpha=0.5, randomize=True)

    # Before merge, only LoRA params have gradients
    non_lora_grads_before = [
        name
        for name, p in model.named_parameters()
        if "lora_" not in name and p.requires_grad
    ]
    assert not non_lora_grads_before, "Non-LoRA params should be frozen before merge"

    # Merge
    merge_lora_weights(model)

    # After merge, all params should have gradients
    frozen_after = [name for name, p in model.named_parameters() if not p.requires_grad]
    assert not frozen_after, f"Some parameters frozen after merge: {frozen_after}"


def test_lora_merge_preserves_equivariance(build_lora_model, random_configs) -> None:
    """Test that merged model preserves rotational equivariance."""
    model, table = build_lora_model(rank=2, alpha=0.5, randomize=True)

    # Merge LoRA weights
    merge_lora_weights(model)
    model.eval()

    base_cfg = random_configs[0]
    energy, forces = _forward_energy_forces(model, [base_cfg], table)
    energy_val = energy.item()
    forces_val = forces.squeeze(0).detach().numpy()

    # Test rotation equivariance after merge
    R = _rotation_matrix()
    rotated_cfg = _rotate_config(base_cfg, R)
    energy_rot, forces_rot = _forward_energy_forces(model, [rotated_cfg], table)

    assert np.allclose(
        energy_rot.item(), energy_val, rtol=1e-6, atol=1e-6
    ), "Energy not invariant under rotation after merge"
    assert np.allclose(
        forces_val @ R.T, forces_rot.squeeze(0).detach().numpy(), rtol=1e-5, atol=1e-5
    ), "Forces not equivariant under rotation after merge"


def test_lora_evaluate_preserves_frozen_state(build_lora_model, random_configs) -> None:
    """Test that evaluate() preserves requires_grad states for LoRA models."""
    from mace.tools import evaluate
    from mace.modules.loss import WeightedEnergyForcesLoss

    model, table = build_lora_model(rank=2, alpha=0.5, randomize=True)

    # Record which parameters should be trainable (only LoRA params)
    lora_params_before = {name: p.requires_grad for name, p in model.named_parameters()}
    trainable_before = [name for name, grad in lora_params_before.items() if grad]
    frozen_before = [name for name, grad in lora_params_before.items() if not grad]

    # Verify initial state: only LoRA params are trainable
    assert all(
        "lora_" in name for name in trainable_before
    ), "Only LoRA params should be trainable initially"
    assert len(frozen_before) > 0, "Some base params should be frozen"

    # Create a minimal data loader for evaluation
    configs = list(random_configs)
    dataset = [_atomic_data_from_config(cfg, table) for cfg in configs]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=len(dataset),
        shuffle=False,
        drop_last=False,
    )

    # Run evaluate
    loss_fn = WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=1.0)
    output_args = {"forces": True, "virials": False, "stress": False}
    evaluate(model, loss_fn, loader, output_args, device=torch.device("cpu"))

    # Check that requires_grad states are preserved
    lora_params_after = {name: p.requires_grad for name, p in model.named_parameters()}

    for name in trainable_before:
        assert lora_params_after[
            name
        ], f"LoRA param {name} should still be trainable after evaluate()"

    for name in frozen_before:
        assert not lora_params_after[
            name
        ], f"Base param {name} should still be frozen after evaluate()"


@pytest.mark.skipif(not CUET_AVAILABLE, reason="cuequivariance not installed")
def test_lora_with_cueq_conversion() -> None:
    """Test that LoRA works correctly with cueq conversion."""
    from mace.modules.lora import LoRAO3Linear
    
    # Build e3nn model
    model, table = _build_model()
    
    # Inject LoRA on e3nn model
    inject_lora(model, rank=2, alpha=0.5)
    _randomize_lora_parameters(model)
    
    # Verify LoRA was injected (should have LoRAO3Linear wrapping o3.Linear)
    lora_count_e3nn = sum(
        1 for m in model.modules() if isinstance(m, LoRAO3Linear)
    )
    assert lora_count_e3nn > 0, "Should have LoRA wrappers in e3nn model"
    
    # Get outputs from e3nn + LoRA
    config = _random_config()
    model.eval()
    with torch.no_grad():
        energy_e3nn, forces_e3nn = _forward_energy_forces(model, [config], table)
    
    # Convert to cueq (should preserve LoRA)
    model_cueq = run_e3nn_to_cueq(model, device="cpu", return_model=True)
    
    # Verify LoRA was preserved
    lora_count_cueq = sum(
        1 for m in model_cueq.modules() if isinstance(m, LoRAO3Linear)
    )
    assert lora_count_cueq == lora_count_e3nn, "LoRA should be preserved after cueq conversion"
    
    # Verify some LoRA wrappers now wrap cuet.Linear
    has_cuet_base = any(
        isinstance(m.base, cuet.Linear)
        for m in model_cueq.modules()
        if isinstance(m, LoRAO3Linear)
    )
    assert has_cuet_base, "Some LoRA wrappers should wrap cuet.Linear after conversion"
    
    # Get outputs from cueq + LoRA (should match e3nn + LoRA)
    model_cueq.eval()
    with torch.no_grad():
        energy_cueq, forces_cueq = _forward_energy_forces(model_cueq, [config], table)
    
    assert torch.allclose(energy_e3nn, energy_cueq, rtol=1e-5, atol=1e-6), \
        f"Energy mismatch after cueq conversion: {energy_e3nn} vs {energy_cueq}"
    assert torch.allclose(forces_e3nn, forces_cueq, rtol=1e-5, atol=1e-6), \
        f"Forces mismatch after cueq conversion"
    
    # Convert back to e3nn (should preserve LoRA)
    model_e3nn_back = run_cueq_to_e3nn(model_cueq, device="cpu", return_model=True)
    
    # Verify LoRA was preserved
    lora_count_back = sum(
        1 for m in model_e3nn_back.modules() if isinstance(m, LoRAO3Linear)
    )
    assert lora_count_back == lora_count_cueq, "LoRA should be preserved after e3nn conversion"
    
    # Get outputs after round-trip conversion
    model_e3nn_back.eval()
    with torch.no_grad():
        energy_back, forces_back = _forward_energy_forces(model_e3nn_back, [config], table)
    
    assert torch.allclose(energy_e3nn, energy_back, rtol=1e-5, atol=1e-6), \
        "Energy should match after round-trip conversion"
    assert torch.allclose(forces_e3nn, forces_back, rtol=1e-5, atol=1e-6), \
        "Forces should match after round-trip conversion"
    
    # Now test merging after conversion back to e3nn
    merge_lora_weights(model_e3nn_back)
    
    # Verify merge removed wrappers
    lora_count_merged = sum(
        1 for m in model_e3nn_back.modules() if isinstance(m, LoRAO3Linear)
    )
    assert lora_count_merged == 0, "LoRA wrappers should be removed after merge"
    
    # Verify outputs still match
    model_e3nn_back.eval()
    with torch.no_grad():
        energy_merged, forces_merged = _forward_energy_forces(model_e3nn_back, [config], table)
    
    assert torch.allclose(energy_e3nn, energy_merged, rtol=1e-5, atol=1e-6), \
        "Energy should match after merge"
    assert torch.allclose(forces_e3nn, forces_merged, rtol=1e-5, atol=1e-6), \
        "Forces should match after merge"


@pytest.mark.skipif(not CUET_AVAILABLE, reason="cuequivariance not installed")
def test_lora_inject_on_cueq_model() -> None:
    """Test that LoRA can be injected directly on a cueq model."""
    from mace.modules.lora import LoRAO3Linear
    from mace.modules.wrapper_ops import CuEquivarianceConfig
    
    # Build cueq model directly
    table = tools.AtomicNumberTable([1, 6])
    cueq_config = CuEquivarianceConfig(
        enabled=True,
        layout="ir_mul",
        group="O3_e3nn",
        optimize_all=True,
        conv_fusion=False,
    )
    model = modules.MACE(
        r_max=4.5,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=1,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("16x0e + 16x1o"),
        MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.random.normal(scale=0.1, size=len(table.zs)),
        avg_num_neighbors=6.0,
        atomic_numbers=table.zs,
        correlation=2,
        radial_type="bessel",
        cueq_config=cueq_config,
    )
    
    # Inject LoRA on cueq model
    inject_lora(model, rank=2, alpha=0.5)
    
    # Verify LoRA was injected
    lora_wrappers = [m for m in model.modules() if isinstance(m, LoRAO3Linear)]
    assert len(lora_wrappers) > 0, "Should have LoRA wrappers"
    
    # Verify some wrappers wrap cuet.Linear
    cuet_bases = [w for w in lora_wrappers if isinstance(w.base, cuet.Linear)]
    assert len(cuet_bases) > 0, "Some LoRA wrappers should wrap cuet.Linear"
    
    # Verify LoRA A and B are also cuet.Linear
    for wrapper in cuet_bases:
        assert isinstance(wrapper.lora_A, cuet.Linear), "lora_A should be cuet.Linear"
        assert isinstance(wrapper.lora_B, cuet.Linear), "lora_B should be cuet.Linear"
        assert wrapper.is_cueq, "Should detect cueq backend"
    
    # Test forward pass
    config = _random_config()
    model.eval()
    with torch.no_grad():
        energy, forces = _forward_energy_forces(model, [config], table)
    
    assert energy.shape == (1,), "Should produce energy output"
    assert forces.shape == (1, 3, 3), "Should produce forces output"
    
    # Test training mode
    model.train()
    energy_train, forces_train = _forward_energy_forces(model, [config], table)
    assert energy_train.requires_grad, "Should have gradients in train mode"


@pytest.mark.skipif(not CUET_AVAILABLE, reason="cuequivariance not installed")
def test_lora_cueq_gradient_flow() -> None:
    """Test that gradients flow correctly through LoRA + cueq."""
    from mace.modules.lora import LoRAO3Linear
    from mace.modules.wrapper_ops import CuEquivarianceConfig
    
    # Build cueq model
    table = tools.AtomicNumberTable([1, 6])
    cueq_config = CuEquivarianceConfig(
        enabled=True,
        layout="ir_mul",
        group="O3_e3nn",
        optimize_all=True,
        conv_fusion=False,
    )
    model = modules.MACE(
        r_max=4.5,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=1,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("16x0e + 16x1o"),
        MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.random.normal(scale=0.1, size=len(table.zs)),
        avg_num_neighbors=6.0,
        atomic_numbers=table.zs,
        correlation=2,
        radial_type="bessel",
        cueq_config=cueq_config,
    )
    
    # Inject LoRA
    inject_lora(model, rank=2, alpha=0.5)
    _randomize_lora_parameters(model)
    
    # Verify we have cuet-based LoRA
    lora_wrappers = [m for m in model.modules() if isinstance(m, LoRAO3Linear)]
    cuet_wrappers = [w for w in lora_wrappers if w.is_cueq]
    assert len(cuet_wrappers) > 0, "Should have cueq-based LoRA wrappers"
    
    # Test gradient flow
    model.train()
    config = _random_config()
    dataset = [_atomic_data_from_config(config, table)]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(loader))
    
    # Forward pass
    outputs = model(batch.to_dict())
    energy = outputs["energy"]
    forces = outputs["forces"]
    
    # Compute loss
    loss = energy.sum() + forces.abs().sum()
    
    # Backward pass
    loss.backward()
    
    # Check that LoRA parameters have gradients
    lora_params_with_grad = []
    lora_params_without_grad = []
    base_params_with_grad = []
    
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            if param.grad is not None and param.grad.abs().sum() > 0:
                lora_params_with_grad.append(name)
            else:
                lora_params_without_grad.append(name)
        elif ".base." in name:
            if param.grad is not None and param.grad.abs().sum() > 0:
                base_params_with_grad.append(name)
    
    # LoRA parameters should have gradients
    assert len(lora_params_with_grad) > 0, "LoRA parameters should have gradients"
    assert len(lora_params_without_grad) == 0, \
        f"All LoRA parameters should have gradients, but these don't: {lora_params_without_grad}"
    
    # Base parameters should NOT have gradients (frozen)
    assert len(base_params_with_grad) == 0, \
        f"Base parameters should be frozen, but these have gradients: {base_params_with_grad}"
