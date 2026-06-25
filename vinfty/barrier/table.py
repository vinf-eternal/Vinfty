"""5-operator parameter table + pure barrier-economics math.

Zero dependencies — only ``math`` from stdlib.
"""

from __future__ import annotations
import math
from typing import Optional

# ═══════════════════════════════════════════════════════════════
# 5 operator parameter table
# ═══════════════════════════════════════════════════════════════

OPERATOR_TABLE: dict[str, dict[str, float]] = {
    "PRNG": {
        "L2_min": 10.0, "L1_inf": 80.0, "delta_L1": 15.0,
        "L1_max": 95.0, "tau": 50000.0,
    },
    "exp_log": {
        "L2_min": 50.0, "L1_inf": 30.0, "delta_L1": 55.0,
        "L1_max": 85.0, "tau": 500.0,
    },
    "ODE": {
        "L2_min": 5000.0, "L1_inf": 8.0, "delta_L1": 72.0,
        "L1_max": 80.0, "tau": 2000.0,
    },
    "Fractal": {
        "L2_min": 10000.0, "L1_inf": 2.0, "delta_L1": 93.0,
        "L1_max": 95.0, "tau": 3000.0,
    },
    "Series": {
        "L2_min": 1000.0, "L1_inf": 15.0, "delta_L1": 70.0,
        "L1_max": 85.0, "tau": 4000.0,
    },
}

OPERATOR_NAMES: list[str] = list(OPERATOR_TABLE.keys())

# M33 sigma constants
SIGMA_1 = 30.0
SIGMA_2 = 10000.0
SIGMA_3 = 500.0

# λ scenario labels
LAMBDA_SCENARIOS: list[tuple[float, str, str]] = [
    (1e-5, "datacenter_gpu", "算力极廉价 — 深度嵌套分形最优"),
    (1e-3, "pc_benchmark",   "个人 PC 通用 — 单层 ODE 混沌均衡"),
    (1e-1, "embedded_mcu",   "MCU 算力昂贵 — 哈希"),
    (1e2,  "human_brain",    "碳基时间最贵 — 多层递归分形"),
]


# ═══════════════════════════════════════════════════════════════
# Core math
# ═══════════════════════════════════════════════════════════════

def compute_l2_star(
    lam: float,
    tau: float,
    delta_l1: float,
    l2_min: float,
) -> tuple[Optional[float], bool]:
    """Compute optimal L₂ operating point.

    Returns (L₂*, has_inflection) where:
      - L₂* is None when λτ/ΔL₁ >= 1 (no interior optimum)
      - has_inflection is True when a proper inflection exists
    """
    ratio = lam * tau / delta_l1
    if ratio >= 1.0:
        return None, False
    l2_star = l2_min - tau * math.log(ratio)
    return max(l2_min, l2_star), True


def compute_l1_star(
    l2_star: float,
    l2_min: float,
    tau: float,
    l1_inf: float,
    delta_l1: float,
) -> float:
    """Compute L₁ at the optimal L₂ operating point."""
    exponent = -(l2_star - l2_min) / tau if tau > 0 else 0.0
    return l1_inf + delta_l1 * math.exp(exponent)


def compute_l_total(l1_star: float, l2_star: float, lam: float) -> float:
    """Compute total cost at the optimal operating point."""
    return l1_star + lam * l2_star


def compute_p_trans(
    l1: float,
    l2: float,
    lc: float = 0.0,
) -> float:
    """M33 penetration coefficient.

    Measures how deeply a signal penetrates through the barrier stack.
    """
    return math.exp(-l1 / SIGMA_1 - l2 / SIGMA_2 - lc / SIGMA_3)


def compute_operator_optimal(
    lam: float,
    op_params: dict[str, float],
    lc: float = 0.0,
) -> dict:
    """Full optimal analysis for a single operator at given λ."""
    l2_min = op_params["L2_min"]
    tau = op_params["tau"]
    delta_l1 = op_params["delta_L1"]
    l1_inf = op_params["L1_inf"]
    l1_max = op_params["L1_max"]

    l2_star_p, has_inflection = compute_l2_star(lam, tau, delta_l1, l2_min)

    if l2_star_p is None:
        l2_star = l2_min
        l1_star = l1_max
    else:
        l2_star = l2_star_p
        l1_star = compute_l1_star(l2_star, l2_min, tau, l1_inf, delta_l1)

    l_total_star = compute_l_total(l1_star, l2_star, lam)
    p_trans = compute_p_trans(l1_star, l2_star, lc)

    return {
        "L2_star": round(l2_star, 1),
        "L1_star": round(l1_star, 4),
        "L_total_star": round(l_total_star, 4),
        "P_trans": round(p_trans, 4),
        "has_inflection": has_inflection,
    }
