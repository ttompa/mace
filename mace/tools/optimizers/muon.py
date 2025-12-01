"""Muon optimizer implementation adapted for MACE models with proper e3nn handling."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from e3nn import o3

from .e3nn_utils import collect_e3nn_weight_info, is_layer_norm_param


def _zeropower_via_newtonschulz5(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Compute the zero power (orthogonalization) of a matrix using Newton-Schulz iteration.

    This implements 5 iterations of Newton-Schulz to approximate the matrix sign function,
    which orthogonalizes the input matrix.

    Args:
        matrix: Input tensor with at least 2 dimensions
        steps: Number of Newton-Schulz iterations

    Returns:
        Orthogonalized matrix of the same shape
    """
    if matrix.ndim < 2:
        raise ValueError("Muon optimizer expects tensors with at least 2 dimensions")

    a, b, c = (3.4445, -4.7750, 2.0315)
    result = matrix
    transpose = False

    if matrix.size(-2) > matrix.size(-1):
        result = result.mT
        transpose = True

    # Ensure spectral norm is at most 1
    result = result / (result.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Perform the NS iterations
    for _ in range(steps):
        gram = result @ result.mT
        update = b * gram + c * gram @ gram
        result = a * result + update @ result

    if transpose:
        result = result.mT

    return result


def _muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    *,
    beta: float,
    ns_steps: int = 5,
    slices_e3nn_linear: Optional[List[Tuple[slice, Tuple[int, int]]]] = None,
) -> torch.Tensor:
    """
    Apply the Muon update to a gradient tensor.

    For e3nn Linear layers, this properly handles the block-diagonal structure
    by applying Newton-Schulz orthogonalization to each irrep path separately.

    Args:
        grad: Gradient tensor
        momentum: Momentum buffer tensor
        beta: Momentum coefficient
        ns_steps: Number of Newton-Schulz iterations
        slices_e3nn_linear: Optional list of (slice, (rows, cols)) for e3nn weights

    Returns:
        The Muon update tensor
    """
    # Update momentum with exponential moving average
    momentum.lerp_(grad, 1 - beta)
    # Nesterov-style update
    update = grad.lerp_(momentum, beta)

    # Handle e3nn Linear weights with block-diagonal structure
    if slices_e3nn_linear is not None:
        update_list = []
        for index_slice, shape_2D in slices_e3nn_linear:
            # Extract and reshape this path's weights to 2D
            weight_slice = update[index_slice].reshape(shape_2D)
            grad_slice = grad[index_slice].reshape(shape_2D)
            # Apply Newton-Schulz to this block
            update_weight_slice = _zeropower_via_newtonschulz5(
                weight_slice, steps=ns_steps
            )
            # Scale by aspect ratio
            update_weight_slice *= (
                max(1, grad_slice.size(-2) / grad_slice.size(-1)) ** 0.5
            )
            update_list.append(update_weight_slice.flatten())
        # Concatenate all blocks back into 1D
        return torch.cat(update_list, dim=-1)

    # Standard handling for regular 2D+ weights
    if grad.ndim < 2:
        raise ValueError("Muon optimizer expects tensors with at least 2 dimensions")

    original_shape = update.shape
    if update.ndim == 2:
        matrix = update
    else:
        matrix = update.flatten(0, -2)

    matrix = _zeropower_via_newtonschulz5(matrix, steps=ns_steps)
    aspect_ratio = float(matrix.size(-2)) / float(matrix.size(-1))
    matrix = matrix * max(1.0, aspect_ratio) ** 0.5

    return matrix.view(original_shape)


class MuonWithAdam(torch.optim.Optimizer):
    """
    Combined Muon + Adam optimizer for MACE models.

    Muon is applied to 2D+ weight tensors (with proper e3nn handling),
    while Adam is used for 1D parameters like biases and layer norm weights.
    """

    def __init__(
        self,
        param_groups: Sequence[dict],
        *,
        ns_steps: int = 5,
    ) -> None:
        """
        Initialize the MuonWithAdam optimizer.

        Args:
            param_groups: Parameter groups, each must have 'use_muon' key
            ns_steps: Number of Newton-Schulz iterations for Muon
        """
        validated_groups = []
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("Param group must define 'use_muon'")
            validated_groups.append(group)

        defaults = dict(ns_steps=ns_steps)
        super().__init__(validated_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_muon", False):
                beta = group.get("momentum", 0.95)
                lr = group["lr"]
                wd = group.get("weight_decay", 0.0)

                # Get e3nn slices list if available (one per parameter)
                slices_list = group.get("slices_e3nn_linear", None)

                for i, param in enumerate(group["params"]):
                    if param.grad is None:
                        continue
                    grad = param.grad
                    state = self.state[param]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(param)

                    # Get slices for this specific parameter
                    slices = slices_list[i] if slices_list is not None else None

                    update = _muon_update(
                        grad.view_as(param),
                        state["momentum_buffer"],
                        beta=beta,
                        ns_steps=self.defaults["ns_steps"],
                        slices_e3nn_linear=slices,
                    ).reshape_as(param)

                    if wd != 0:
                        param.mul_(1 - lr * wd)
                    param.add_(update, alpha=-lr)
            else:
                # Adam update for 1D parameters
                lr = group["lr"]
                wd = group.get("weight_decay", 0.0)
                beta1, beta2 = group.get("betas", (0.9, 0.999))
                eps = group.get("eps", 1e-8)

                for param in group["params"]:
                    if param.grad is None:
                        continue
                    grad = param.grad
                    state = self.state[param]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(param)
                        state["exp_avg_sq"] = torch.zeros_like(param)

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    state["step"] += 1

                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    denom = exp_avg_sq.sqrt().add_(eps)
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    step_size = lr * (bias_correction2**0.5) / bias_correction1

                    if wd != 0:
                        param.mul_(1 - lr * wd)
                    param.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def build_muon_param_groups(
    param_groups: Sequence[dict],
    model: Optional[torch.nn.Module] = None,
    *,
    momentum: float,
    betas: Tuple[float, float],
    eps: float,
) -> List[dict]:
    """
    Build parameter groups for MuonWithAdam optimizer with proper e3nn handling.

    This function:
    1. Identifies e3nn Linear layers and extracts weight slice information
    2. Separates parameters into Muon (2D+) and Adam (1D) groups
    3. Properly handles e3nn weights with slice metadata

    Args:
        param_groups: Existing parameter groups (typically from MACE's setup)
        model: The model (needed to find e3nn layers). If None, falls back to basic splitting.
        momentum: Muon momentum coefficient
        betas: Adam beta coefficients
        eps: Adam epsilon

    Returns:
        List of parameter groups ready for MuonWithAdam
    """
    # Collect e3nn weight information if model is provided
    e3nn_weight_info = collect_e3nn_weight_info(model) if model is not None else {}

    # Also collect layer norm parameter names for proper handling
    layer_norm_params = set()
    if model is not None:
        for name, param in model.named_parameters():
            if is_layer_norm_param(name, param):
                layer_norm_params.add(param)

    muon_ready_groups: List[dict] = []

    for group in param_groups:
        params = list(group["params"])
        group["params"] = params
        lr = group.get("lr")
        weight_decay = group.get("weight_decay", 0.0)

        # Categorize parameters
        e3nn_params = []
        e3nn_slices = []
        regular_muon_params = []
        adam_params = []

        for param in params:
            # Skip frozen parameters
            if not param.requires_grad:
                continue

            # Layer norm params -> Adam
            if param in layer_norm_params:
                adam_params.append(param)
            # e3nn weights with slice info -> Muon with slices
            elif param in e3nn_weight_info:
                e3nn_params.append(param)
                e3nn_slices.append(e3nn_weight_info[param])
            # Regular 2D+ weights -> Muon
            elif param.ndim >= 2:
                regular_muon_params.append(param)
            # 1D parameters -> Adam
            else:
                adam_params.append(param)

        # Create group for e3nn Linear weights with slice metadata
        if e3nn_params:
            muon_ready_groups.append(
                {
                    "params": e3nn_params,
                    "slices_e3nn_linear": e3nn_slices,
                    "use_muon": True,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "momentum": momentum,
                }
            )

        # Create group for regular 2D+ weights (standard Muon)
        if regular_muon_params:
            muon_ready_groups.append(
                {
                    "params": regular_muon_params,
                    "use_muon": True,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "momentum": momentum,
                }
            )

        # Create group for 1D params and layer norm (use Adam)
        if adam_params:
            muon_ready_groups.append(
                {
                    "params": adam_params,
                    "use_muon": False,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "betas": betas,
                    "eps": eps,
                }
            )

    return muon_ready_groups
