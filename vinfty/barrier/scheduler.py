"""LambdaScheduler — barrier-economics-driven operator selection.

Wraps the pure functions in ``table.py`` with stateful scheduling
logic and collapse detection.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional

from .table import (
    OPERATOR_TABLE, LAMBDA_SCENARIOS,
    compute_operator_optimal, compute_p_trans,
)
from .estimator import LambdaEstimator


# ═══════════════════════════════════════════════════════════════
# Silence collapse detection (§4)
# ═══════════════════════════════════════════════════════════════

@dataclass
class CollapseDetector:
    """Track structural density I_struct and detect silent collapse.

    Silence collapse = all metrics look green but the system's
    internal structure has already started degrading.
    """

    epsilon: float = 0.02          # I_struct stagnation threshold
    delta: float = 1.0             # ∂L₁/∂d stagnation threshold
    history_window: int = 10

    def __post_init__(self):
        self._l1_history: list[float] = []
        self._l2_history: list[float] = []
        self._i_struct_history: list[float] = []
        self.collapse_count: int = 0

    def update(
        self,
        l1: float,
        l2: float,
        i_struct: float,
        depth: int = 0,
    ) -> dict:
        """Check collapse conditions.

        Returns dict with:
          collapsed: bool
          reasons: list[str]
          severity: str (green / yellow / red)
        """
        self._l1_history.append(l1)
        self._l2_history.append(l2)
        self._i_struct_history.append(i_struct)

        if len(self._i_struct_history) < 3:
            return {"collapsed": False, "reasons": [], "severity": "green"}

        reasons = []
        i_struct_now = self._i_struct_history[-1]
        i_struct_prev = self._i_struct_history[-2]

        # Condition 1: I_struct stagnation
        if abs(i_struct_now - i_struct_prev) < self.epsilon:
            reasons.append(f"I_struct_stagnation Δ={abs(i_struct_now - i_struct_prev):.4f}")

        # Condition 2: L1 not dropping with depth
        if len(self._l1_history) >= self.history_window:
            recent_l1 = self._l1_history[-self.history_window:]
            d_l1_dd = (recent_l1[-1] - recent_l1[0]) / max(1, self.history_window)
            if abs(d_l1_dd) < self.delta:
                reasons.append(f"L1_stagnation ∂L₁/∂d={d_l1_dd:.2f}")

        # Condition 3: L2 still rising
        if len(self._l2_history) >= 3:
            l2_trend = (self._l2_history[-1] - self._l2_history[-3]) / 2
            if l2_trend > 0:
                reasons.append(f"L2_rising ∂L₂/∂t={l2_trend:.1f}")

        collapsed = len(reasons) >= 2

        if collapsed:
            self.collapse_count += 1

        p_trans = compute_p_trans(l1, l2)
        if p_trans < 0.05:
            severity = "red"
        elif p_trans < 0.15:
            severity = "yellow"
        else:
            severity = "green"

        return {
            "collapsed": collapsed,
            "reasons": reasons,
            "severity": severity,
            "p_trans": round(p_trans, 4),
            "i_struct": round(i_struct_now, 4),
        }


# ═══════════════════════════════════════════════════════════════
# Main scheduler
# ═══════════════════════════════════════════════════════════════

@dataclass
class LambdaScheduler:
    """Barrier-economics-driven operator scheduler.

    Given a λ estimate, selects the optimal operator type and
    computes L₂*, L₁*, L_total*, and P_trans.

    Usage::

        sched = LambdaScheduler()
        rec = sched.evaluate(lambda_estimate=0.001)
        print(rec["recommended_operator"])
    """

    lam: float = 0.001
    lc: float = 0.0       # coupling overhead (non-zero for nested operators)
    collapse_detector: CollapseDetector = field(default_factory=CollapseDetector)
    _last_recommendation: Optional[dict] = None

    def evaluate(self, lam: Optional[float] = None) -> dict:
        """Full evaluation: find optimal operator for given λ.

        Returns dict with:
          lam: float
          scenario: str
          recommended_operator: str
          operators: dict of all 5 results
          best: optimal operator details
        """
        lam = lam if lam is not None else self.lam
        self.lam = lam

        scenario_label = self._classify_scenario(lam)

        results: dict[str, dict] = {}
        best_op = None
        best_l_total = float("inf")

        for op_name, params in OPERATOR_TABLE.items():
            r = compute_operator_optimal(lam, params, self.lc)
            results[op_name] = r
            if r["L_total_star"] < best_l_total:
                best_l_total = r["L_total_star"]
                best_op = op_name

        self._last_recommendation = {
            "lam": lam,
            "scenario": scenario_label,
            "recommended_operator": best_op,
            "operators": results,
            "best": results[best_op] if best_op else {},
            "timestamp": time.time(),
        }
        return self._last_recommendation

    def recommend(
        self,
        lam: Optional[float] = None,
        current_l1: Optional[float] = None,
        current_l2: Optional[float] = None,
        current_depth: int = 0,
        i_struct: float = 0.5,
    ) -> dict:
        """Return a scheduling recommendation.

        Returns dict with keys:
          operation: str (maintain / switch_operator / collapse_warning)
          target_operator: Optional[str]
          L2_target: float
          L_total_estimated: float
          P_trans: float
          collapse_risk: str (green / yellow / red)
          details: str
        """
        ev = self.evaluate(lam)

        collapse = self.collapse_detector.update(
            current_l1 or ev["best"]["L1_star"],
            current_l2 or ev["best"]["L2_star"],
            i_struct,
            current_depth,
        )

        operation = "maintain"
        details = f"Optimal operator: {ev['recommended_operator']}"

        if collapse["collapsed"]:
            operation = "collapse_warning"
            details = f"Silence collapse risk! {'; '.join(collapse['reasons'])}"
        elif ev["scenario"] == "datacenter_gpu" and current_depth < 2:
            operation = "switch_operator"
            details = f"Deep nesting recommended: {ev['recommended_operator']}"

        return {
            "operation": operation,
            "target_operator": ev["recommended_operator"],
            "L2_optimal": ev["best"]["L2_star"],
            "L1_optimal": ev["best"]["L1_star"],
            "L_total_estimated": ev["best"]["L_total_star"],
            "P_trans": ev["best"]["P_trans"],
            "collapse_risk": collapse["severity"],
            "scenario": ev["scenario"],
            "details": details,
        }

    def _classify_scenario(self, lam: float) -> str:
        """Classify λ into one of 4 deployment scenarios."""
        if lam <= 5e-5:
            return "datacenter_gpu"
        elif lam <= 5e-2:
            return "pc_benchmark"
        elif lam <= 10.0:
            return "embedded_mcu"
        else:
            return "human_brain"

    def scenario_table(self) -> str:
        """Generate a formatted comparison table across all 4 scenarios."""
        lines = [
            f"{'Scenario':<20} {'λ':<10} {'Best Op':<12} {'L₂*':<10} {'L₁*':<10} "
            f"{'L_total*':<12} {'P_trans':<10}"
        ]
        lines.append("─" * len(lines[0]))

        for lam_val, label, _ in LAMBDA_SCENARIOS:
            ev = self.evaluate(lam_val)
            best = ev["best"]
            op = ev["recommended_operator"]
            lines.append(
                f"{label:<20} {lam_val:<10.6f} {op:<12} {best['L2_star']:<10.1f} "
                f"{best['L1_star']:<10.4f} {best['L_total_star']:<12.4f} "
                f"{best['P_trans']:<10.4f}"
            )

        lines.append("─" * len(lines[0]))
        ev_cur = self.evaluate()
        lines.append(
            f"{'current':<20} {self.lam:<10.6f} "
            f"{ev_cur['recommended_operator']:<12} "
            f"{ev_cur['best']['L2_star']:<10.1f} "
            f"{ev_cur['best']['L1_star']:<10.4f} "
            f"{ev_cur['best']['L_total_star']:<12.4f} "
            f"{ev_cur['best']['P_trans']:<10.4f}"
        )

        return "\n".join(lines)
