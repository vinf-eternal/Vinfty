"""vinfty.barrier — barrier-economics-driven operator selection.

Core formulas from SiliconLifeOS docs/barrier_economics_numerical_fit.md:

    L₂* = L₂,min − τ·ln(λτ/ΔL₁)          # optimal computation depth
    L₁* = L₁,inf + ΔL₁·exp(−(L₂*−L₂,min)/τ)  # cognitive cost at optimum
    L_total* = L₁* + λ·L₂*                # total cost at optimum
    P_trans = exp(−L₁/σ₁ − L₂/σ₂)        # M33 penetration coefficient
"""

from .table import (
    OPERATOR_TABLE, OPERATOR_NAMES, LAMBDA_SCENARIOS,
    compute_l2_star, compute_l1_star, compute_l_total, compute_p_trans,
    compute_operator_optimal,
)
from .estimator import LambdaEstimator, auto_detect_lambda
from .scheduler import LambdaScheduler, CollapseDetector

__all__ = [
    "OPERATOR_TABLE", "OPERATOR_NAMES", "LAMBDA_SCENARIOS",
    "compute_l2_star", "compute_l1_star", "compute_l_total", "compute_p_trans",
    "compute_operator_optimal",
    "LambdaEstimator", "auto_detect_lambda",
    "LambdaScheduler", "CollapseDetector",
]
