"""Utilities for handling e3nn layer weights in the Muon optimizer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from e3nn import o3


def compute_e3nn_linear_weight_slices(
    linear: o3.Linear,
) -> List[Tuple[slice, Tuple[int, int]]]:
    """
    Compute weight index slices for an e3nn o3.Linear layer.

    e3nn Linear stores weights as a 1D tensor concatenating weight matrices
    for each (irrep_in, irrep_out) path. This function returns the slices
    and 2D shapes needed to properly apply Muon's Newton-Schulz orthogonalization
    to each block separately.

    Args:
        linear: An e3nn o3.Linear module

    Returns:
        List of (slice, (rows, cols)) tuples where:
        - slice: The index slice into the flattened weight tensor
        - (rows, cols): The 2D shape for this path's weight matrix
    """
    slices = []
    offset = 0

    for ins in linear.instructions:
        # ins format: (i_in, i_out, path_weight, has_weight, ...)
        # has_weight is at index 3
        if len(ins) < 4 or not ins[3]:
            continue

        i_in, i_out = ins[0], ins[1]
        mul_in = linear.irreps_in[i_in].mul
        mul_out = linear.irreps_out[i_out].mul
        numel = mul_in * mul_out

        if numel > 0:
            slices.append((slice(offset, offset + numel), (mul_in, mul_out)))
            offset += numel

    return slices


def compute_fctp_weight_slices(
    fctp: o3.FullyConnectedTensorProduct,
) -> List[Tuple[slice, Tuple[int, int]]]:
    """
    Compute weight index slices for an e3nn FullyConnectedTensorProduct.

    Similar to Linear, FCTP stores weights for multiple paths in a 1D tensor.

    Args:
        fctp: An e3nn o3.FullyConnectedTensorProduct module

    Returns:
        List of (slice, (rows, cols)) tuples for each weighted path
    """
    slices = []
    offset = 0

    for ins in fctp.instructions:
        # ins format: (i_1, i_2, i_out, connection_mode, has_weight, path_weight, weight_numel)
        if len(ins) < 5 or not ins[4]:  # has_weight at index 4
            continue

        # For FCTP, the weight shape depends on the connection mode
        # 'uvw' mode: (mul_1, mul_2, mul_out)
        # We need to determine the 2D shape for Muon
        i_1, i_2, i_out = ins[0], ins[1], ins[2]
        mul_1 = fctp.irreps_in1[i_1].mul
        mul_2 = fctp.irreps_in2[i_2].mul
        mul_out = fctp.irreps_out[i_out].mul

        # For 'uvw' connection, weights are (mul_1 * mul_2, mul_out)
        # This is the natural 2D shape for Muon
        numel = mul_1 * mul_2 * mul_out
        if numel > 0:
            slices.append((slice(offset, offset + numel), (mul_1 * mul_2, mul_out)))
            offset += numel

    return slices


def compute_tp_weight_slices(
    tp: o3.TensorProduct,
) -> Optional[List[Tuple[slice, Tuple[int, int]]]]:
    """
    Compute weight index slices for an e3nn TensorProduct with internal weights.

    Args:
        tp: An e3nn o3.TensorProduct module with internal_weights=True

    Returns:
        List of (slice, (rows, cols)) tuples, or None if no internal weights
    """
    if not tp.internal_weights:
        return None

    slices = []
    offset = 0

    for ins in tp.instructions:
        # ins format varies, but has_weight and weight info are present
        if len(ins) < 5 or not ins[4]:  # has_weight
            continue

        i_1, i_2, i_out = ins[0], ins[1], ins[2]
        connection_mode = ins[3] if len(ins) > 3 else "uvw"

        mul_1 = tp.irreps_in1[i_1].mul
        mul_2 = tp.irreps_in2[i_2].mul
        mul_out = tp.irreps_out[i_out].mul

        # Determine numel based on connection mode
        if connection_mode == "uvw":
            numel = mul_1 * mul_2 * mul_out
            shape_2d = (mul_1 * mul_2, mul_out)
        elif connection_mode == "uvu":
            numel = mul_1 * mul_2
            shape_2d = (mul_1, mul_2)
        elif connection_mode == "uvv":
            numel = mul_1 * mul_out
            shape_2d = (mul_1, mul_out)
        elif connection_mode == "uuu":
            numel = mul_1
            shape_2d = (1, mul_1)
        else:
            # For other modes, use a simple 2D reshape
            numel = getattr(ins, "weight_numel", mul_1 * mul_out)
            shape_2d = (mul_1, max(1, numel // mul_1))

        if numel > 0:
            slices.append((slice(offset, offset + numel), shape_2d))
            offset += numel

    return slices if slices else None


def get_weight_slices_for_module(
    module: torch.nn.Module,
) -> Optional[List[Tuple[slice, Tuple[int, int]]]]:
    """
    Get weight slices for any supported e3nn module type.

    Args:
        module: A torch module (potentially e3nn)

    Returns:
        List of (slice, shape) tuples, or None if not an e3nn module with weights
    """
    if isinstance(module, o3.Linear):
        return compute_e3nn_linear_weight_slices(module)
    elif isinstance(module, o3.FullyConnectedTensorProduct):
        return compute_fctp_weight_slices(module)
    elif isinstance(module, o3.TensorProduct) and module.internal_weights:
        return compute_tp_weight_slices(module)
    return None


def collect_e3nn_weight_info(
    model: torch.nn.Module,
) -> Dict[torch.nn.Parameter, List[Tuple[slice, Tuple[int, int]]]]:
    """
    Collect weight slicing information for all e3nn modules in a model.

    Args:
        model: The model to analyze

    Returns:
        Dictionary mapping parameter tensors to their slice information
    """
    e3nn_weight_info: Dict[
        torch.nn.Parameter, List[Tuple[slice, Tuple[int, int]]]
    ] = {}

    for name, module in model.named_modules():
        slices = get_weight_slices_for_module(module)
        if slices is not None and hasattr(module, "weight"):
            weight = module.weight
            if isinstance(weight, torch.nn.Parameter):
                e3nn_weight_info[weight] = slices

    return e3nn_weight_info


def is_layer_norm_param(name: str, param: torch.nn.Parameter) -> bool:
    """
    Check if a parameter belongs to a layer normalization module.

    Layer norm parameters should typically use Adam instead of Muon.

    Args:
        name: The parameter name from named_parameters()
        param: The parameter tensor

    Returns:
        True if this appears to be a layer norm parameter
    """
    name_lower = name.lower()
    return any(
        ln_name in name_lower
        for ln_name in ["layernorm", "layer_norm", "ln_", "_ln", "rmsnorm", "rms_norm"]
    )
