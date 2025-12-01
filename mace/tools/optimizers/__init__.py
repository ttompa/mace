"""Optimization utilities for MACE training."""

from .e3nn_utils import (
    collect_e3nn_weight_info,
    compute_e3nn_linear_weight_slices,
    compute_fctp_weight_slices,
    compute_tp_weight_slices,
    get_weight_slices_for_module,
    is_layer_norm_param,
)
from .muon import (
    MuonWithAdam,
    build_muon_param_groups,
)

__all__ = [
    "MuonWithAdam",
    "build_muon_param_groups",
    "collect_e3nn_weight_info",
    "compute_e3nn_linear_weight_slices",
    "compute_fctp_weight_slices",
    "compute_tp_weight_slices",
    "get_weight_slices_for_module",
    "is_layer_norm_param",
]
