"""
vinfty.judge — dataset consistency audit helper.

No model training. No data modification.
Only diagnostic reports: ont_self scores, contradiction clusters,
palace groupings, labeling conflicts, silence collapse risk.

Usage::

    from vinfty.judge import audit_dataset

    report = audit_dataset(["text a", "text b", "text c"], lambda_estimate=0.001)
    print(report["ont_self_mean"], report["contradictions"])
"""

from __future__ import annotations
from collections import defaultdict
from typing import Any, Optional

from .core import V9Orchestrator
from .barrier.scheduler import CollapseDetector


def audit_dataset(
    texts: list[str],
    labels: Optional[list[str]] = None,
    palaces: Optional[list[str]] = None,
    lambda_estimate: Optional[float] = None,
    active_k: int = 1000,
) -> dict[str, Any]:
    """Run a consistency audit on a dataset.

    Parameters
    ----------
    texts : list[str]
        Dataset samples (required).
    labels : list[str] or None
        Optional ground-truth labels for conflict detection.
    palaces : list[str] or None
        Optional pre-assigned Palace labels.
        When None, each sample gets a hash-based Palace.
    lambda_estimate : float or None
        Barrier economics λ. None = auto-detect.
        Higher λ means lighter audit (less coupling detail).
    active_k : int
        Active memory pool size (set >= len(texts) for full audit).

    Returns
    -------
    dict with keys:
        n_samples, ont_self_mean, c_ij_density, palace_distribution,
        contradictions, label_conflicts, silent_collapse_risk, barrier
    """
    engine = V9Orchestrator(active_k=active_k, lambda_estimate=lambda_estimate)
    collapse_detector = CollapseDetector()

    # Populate engine with all samples
    for i, text in enumerate(texts):
        p = palaces[i] if palaces else None
        engine.step(text, palace=p)

    # Per-sample ont_self contribution (leave-one-out approximation)
    palace_map: dict[str, list[int]] = defaultdict(list)
    for i, mem in enumerate(engine.memories):
        palace_map[mem.get("palace", "?")].append(i)

    sample_scores: list[float] = []
    contradictions: list[dict] = []
    for i in range(len(engine.memories)):
        same = 0
        total = 0
        for j in range(len(engine.memories)):
            if i == j:
                continue
            total += 1
            if engine.memories[i].get("palace") == engine.memories[j].get("palace"):
                same += 1
        score = same / total if total else 0.0
        sample_scores.append(round(score, 4))
        if score < 0.1 and score > 0.0:
            contradictions.append({
                "sample_idx": i,
                "palace": engine.memories[i].get("palace"),
                "ont_self": score,
                "content_preview": engine.memories[i]["content"][:80],
            })

    contradictions.sort(key=lambda x: x["ont_self"])

    # Label conflicts: samples with same label map to different Palaces
    label_conflicts: list[dict] = []
    if labels and len(labels) == len(texts):
        label_to_palaces: dict[str, set[str]] = defaultdict(set)
        for i, lbl in enumerate(labels):
            label_to_palaces[lbl].add(engine.memories[i].get("palace", "?"))
        for lbl, pal_set in label_to_palaces.items():
            if len(pal_set) > 1:
                label_conflicts.append({
                    "label": lbl,
                    "palace_count": len(pal_set),
                    "palaces": sorted(pal_set),
                })

    # Silence collapse detection
    palace_counts = engine._palace_distribution()
    n_palaces = len(palace_counts)
    i_struct = engine._coupling_density()
    collapse_status = collapse_detector.update(
        l1=i_struct * 100,
        l2=n_palaces * 10,
        i_struct=i_struct,
        depth=1,
    )

    r = engine.report()
    return {
        "n_samples": len(texts),
        "ont_self_mean": r["ont_self"],
        "c_ij_density": r["c_ij_density"],
        "n_palaces": n_palaces,
        "palace_distribution": dict(palace_counts),
        "sample_scores_top5": sample_scores[:5],
        "sample_scores_bottom5": sample_scores[-5:] if len(sample_scores) >= 5 else sample_scores,
        "contradictions_n": len(contradictions),
        "contradictions_top5": contradictions[:5],
        "label_conflicts": label_conflicts,
        "silent_collapse_risk": collapse_status["severity"],
        "silent_collapse_details": collapse_status["reasons"],
        "hmM_state": r["hmm_state"],
        "barrier": r["barrier"],
    }
