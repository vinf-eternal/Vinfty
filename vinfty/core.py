"""
vinfty.core — V9Orchestrator: cognitive state machine for tool-call observation.

No LLM dependency. Pure symbolic computation:
  - ont_self = Σ C_ij · I_j (self-consistency)
  - C_ij = 1 when memories share a Palace label
  - HMM K=2 tracks s₀ (stable) / s₁ (searching) state
  - Palace routing assigns tool calls to cognitive domains
  - Barrier economics adapts computation depth to hardware constraints
"""

import math
import json
from collections import defaultdict
from typing import Callable, Optional

from . import barrier as _barrier


class V9Orchestrator:
    """Lightweight cognitive engine tracking ont_self, C_ij coupling, HMM state.

    Parameters
    ----------
    active_k : int
        Max memories in active pool.
    lambda_estimate : float or None
        λ estimate for barrier economics. ``None`` = auto-detect.
    """

    def __init__(self, active_k: int = 100, lambda_estimate: Optional[float] = None):
        self.active_k = active_k
        self.memories: list[dict] = []
        self.archive: list[dict] = []
        self._palace_registry: dict[str, Callable] = {}
        self._hmm = _HMM()
        self._last_scan = -1
        # Barrier economics
        lam = lambda_estimate if lambda_estimate is not None else _barrier.auto_detect_lambda()
        self._barrier = _barrier.LambdaScheduler(lam=lam)
        self._mode = self._barrier.evaluate()["recommended_operator"]
        self._eval_counter = 0

    # ── tool registration ──

    def register(self, palace: str = "P_generic"):
        """Decorator: register a function as a tool belonging to a Palace domain."""
        def decorator(fn):
            self._palace_registry[fn.__name__] = {"fn": fn, "palace": palace}
            return fn
        return decorator

    def register_tool(self, name: str, fn: Callable, palace: str = "P_generic"):
        """Register a tool by name."""
        self._palace_registry[name] = {"fn": fn, "palace": palace}

    # ── step / run ──

    def step(self, content: str, tool: Optional[Callable] = None, **extra):
        """Process one interaction: store memory, update coupling, run HMM."""
        mem = {
            "content": content,
            "palace": extra.get("palace", self._resolve_palace(content, tool)),
            "confidence": extra.get("confidence", 0.5),
            "tick": len(self.memories) + len(self.archive),
        }
        self.memories.append(mem)
        # active pool pruning
        if len(self.memories) > self.active_k:
            excess = self.memories[:-self.active_k]
            self.archive.extend(excess)
            self.memories = self.memories[-self.active_k:]
            if len(self.archive) > 2000:
                self.archive = self.archive[-2000:]
        # HMM update (convenience: rescan every step)
        self._scan_hmm()
        # Re-evaluate barrier mode every 100 steps
        self._eval_counter += 1
        if self._eval_counter % 100 == 0:
            self._mode = self._barrier.evaluate()["recommended_operator"]

    def adapt(self, lambda_estimate: Optional[float] = None):
        """Manually adjust barrier economics and re-evaluate mode.

        Parameters
        ----------
        lambda_estimate : float or None
            New λ value. ``None`` = auto-detect.
        """
        lam = lambda_estimate if lambda_estimate is not None else _barrier.auto_detect_lambda()
        self._barrier.lam = lam
        self._mode = self._barrier.evaluate()["recommended_operator"]

    def run(self, tasks: list[str], **extra):
        """Process multiple string tasks in sequence."""
        for t in tasks:
            self.step(t, **extra)

    # ── report ──

    def report(self) -> dict:
        """Return current cognitive state summary with barrier economics."""
        ont = self.ont_self()
        c_density = self._coupling_density()
        palace_counts = self._palace_distribution()

        # Barrier status
        b = self._barrier
        best = b._last_recommendation["best"] if b._last_recommendation else {}

        return {
            "ont_self": round(ont, 4),
            "c_ij_density": round(c_density, 4),
            "hmm_state": self._hmm.state,
            "hmm_s1_posterior": round(self._hmm.posterior_s1, 4),
            "memory_count": len(self.memories),
            "archive_count": len(self.archive),
            "palace_count": len(palace_counts),
            "palace_flow": [m.get("palace", "?") for m in self.memories[-5:]],
            "palace_distribution": palace_counts,
            "barrier": {
                "lambda": b.lam,
                "mode": self._mode,
                "scenario": b._classify_scenario(b.lam),
                "L_total_star": best.get("L_total_star"),
                "P_trans": best.get("P_trans"),
            },
        }

    def ont_self(self) -> float:
        """Compute self-consistency: mean C_ij · I_j across all active pairs."""
        mems = self.memories
        n = len(mems)
        if n < 2:
            return 0.0
        total = 0.0
        pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = 1.0 if mems[i].get("palace") == mems[j].get("palace") else 0.0
                ii = mems[i].get("confidence", 0.5)
                ij = mems[j].get("confidence", 0.5)
                total += c * (ii + ij) / 2.0
                pairs += 1
        return total / pairs if pairs else 0.0

    # ── trace ──

    def trace(self) -> list[dict]:
        """Return full memory history with HMM labels for inspection."""
        return [
            {"content": m["content"][:60], "palace": m.get("palace"), "tick": m.get("tick")}
            for m in (self.memories + self.archive)
        ]

    # ── internals ──

    def _resolve_palace(self, content: str, tool: Optional[Callable] = None) -> str:
        if tool and tool.__name__ in self._palace_registry:
            return self._palace_registry[tool.__name__]["palace"]
        # fallback: hash-based palace for unrecognized content
        h = hash(content) & 0xFFFF
        return f"P_hash_{h % 64:02d}"

    def _coupling_density(self) -> float:
        mems = self.memories
        if len(mems) < 2:
            return 0.0
        same = sum(1 for i in range(len(mems)) for j in range(i + 1, len(mems))
                   if mems[i].get("palace") == mems[j].get("palace"))
        total = len(mems) * (len(mems) - 1) / 2.0
        return same / total if total else 0.0

    def _palace_distribution(self) -> dict:
        counts = defaultdict(int)
        for m in self.memories:
            counts[m.get("palace", "?")] += 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def _scan_hmm(self):
        """Update HMM state from memory delta distribution."""
        mems = self.memories
        if len(mems) < 3:
            return
        ticks = [m.get("tick", i) for i, m in enumerate(mems)]
        deltas = [ticks[i] - ticks[i - 1] for i in range(1, len(ticks))]
        # simple empirical update
        mean_delta = sum(deltas) / len(deltas) if deltas else 1.0
        large_delta_ratio = sum(1 for d in deltas if d > mean_delta * 1.5) / len(deltas) if deltas else 0.0
        self._hmm.update(large_delta_ratio)


class _HMM:
    """Minimal 2-state HMM tracking s₀ (stable) / s₁ (searching)."""

    def __init__(self):
        self.state = "s0"
        self.posterior_s1 = 0.0
        self._A = [[0.85, 0.15], [0.20, 0.80]]  # transition matrix
        self._B = [[0.70, 0.30], [0.25, 0.75]]  # emission (s0 prefers small-delta, s1 prefers large)

    def update(self, large_ratio: float):
        # simple forward pass
        like_s0 = self._B[0][0] * (1 - large_ratio) + self._B[0][1] * large_ratio
        like_s1 = self._B[1][0] * (1 - large_ratio) + self._B[1][1] * large_ratio
        total = like_s0 + like_s1
        self.posterior_s1 = like_s1 / total if total > 0 else 0.5
        if self.state == "s0":
            self.state = "s1" if self.posterior_s1 > 0.55 else "s0"
        else:
            self.state = "s0" if self.posterior_s1 < 0.40 else "s1"
