"""
vinfty.judge — dataset & codebase structural audit helpers.

No model training. No data modification.
Only diagnostic reports: ont_self scores, contradiction clusters,
palace groupings, labeling conflicts, silence collapse risk,
and now: codebase topological structure audits.

Usage::

    from vinfty.judge import audit_dataset, code_review_audit

    # Dataset audit
    report = audit_dataset(["text a", "text b", "text c"], lambda_estimate=0.001)
    print(report["ont_self_mean"], report["contradictions"])

    # Codebase structural audit
    report = code_review_audit("./my_project")
    print(report["structural_ont_self"], report["module_coupling"])
"""

from __future__ import annotations
import os
import re
import ast
import math
from collections import defaultdict
from pathlib import Path
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


# ──────────────────────────────────────────
#  Codebase structural audit (v1.0)
# ──────────────────────────────────────────

IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", "venv", ".venv",
    ".tox", ".eggs", "dist", "build", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".DS_Store", "*.egg-info",
}
IGNORE_EXTS = {".pyc", ".pyo", ".so", ".dll", ".pyd", ".exe", ".svg", ".png", ".jpg"}


def _scan_files(
    root: str,
    extra_ignore_dirs: set[str] | None = None,
    extra_ignore_exts: set[str] | None = None,
) -> list[dict]:
    """Scan directory, return list of {path, rel, ext, name, lines, ast_ok}.

    Parameters
    ----------
    extra_ignore_dirs : set[str] | None
        Additional directory names to skip (merged with IGNORE_DIRS).
    extra_ignore_exts : set[str] | None
        Additional file extensions to skip (merged with IGNORE_EXTS).
    """
    files = []
    root_p = Path(root).resolve()
    ignore_dirs = IGNORE_DIRS | (extra_ignore_dirs or set())
    ignore_exts = IGNORE_EXTS | (extra_ignore_exts or set())
    for abs_path in root_p.rglob("*"):
        if not abs_path.is_file():
            continue
        rel = str(abs_path.relative_to(root_p))
        parts = rel.replace("\\", "/").split("/")
        if any(p in ignore_dirs for p in parts):
            continue
        ext = abs_path.suffix.lower()
        if ext in ignore_exts:
            continue
        lines = 0
        ast_ok = False
        imports: list[str] = []
        try:
            raw = abs_path.read_bytes()
            lines = raw.count(b"\n") + 1 if len(raw) < 500_000 else -1
            if ext == ".py" and lines > 0:
                tree = ast.parse(raw)
                ast_ok = True
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name.split(".")[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module.split(".")[0])
        except (SyntaxError, Exception):
            pass
        files.append({
            "path": str(abs_path),
            "rel": rel,
            "ext": ext,
            "name": abs_path.name,
            "lines": lines,
            "ast_ok": ast_ok,
            "imports": list(set(imports)),
        })
    return files


def _compute_directory_topology(files: list[dict]) -> dict:
    """Analyze directory tree as a topological structure.

    Returns metrics:
      - depth (max nesting)
      - breadth (avg children per dir)
      - balance (stddev of children count)
      - module_boundary_clarity (how well directories map to file clusters)
    """
    dirs: dict[str, set[str]] = defaultdict(set)
    for f in files:
        rel = f["rel"]
        d = "/".join(rel.replace("\\", "/").split("/")[:-1]) or "."
        dirs[d].add(f["name"])

    if not dirs:
        return {"depth": 0, "breadth": 0, "balance": 0, "module_boundary_clarity": 0}

    depths = [d.count("/") for d in dirs if d != "."]
    max_depth = max(depths) if depths else 0
    child_counts = [len(v) for v in dirs.values()]
    avg_breadth = sum(child_counts) / len(child_counts) if child_counts else 0
    if len(child_counts) > 1:
        variance = sum((c - avg_breadth) ** 2 for c in child_counts) / len(child_counts)
        balance = 1.0 - min(1.0, math.sqrt(variance) / max(avg_breadth, 1))
    else:
        balance = 1.0

    module_clarity = 0.0
    py_files = [f for f in files if f["ext"] == ".py" and f["imports"]]
    if py_files:
        intra = 0
        total = 0
        for f in py_files:
            f_dir = "/".join(f["rel"].replace("\\", "/").split("/")[:-1]) or "."
            for imp in f["imports"]:
                for other in py_files:
                    other_dir = "/".join(other["rel"].replace("\\", "/").split("/")[:-1]) or "."
                    if imp in other["name"] and not other["name"].endswith(".py"):
                        continue
                    if imp in other["name"]:
                        total += 1
                        if f_dir == other_dir:
                            intra += 1
        if total:
            module_clarity = intra / total

    return {
        "depth": max_depth,
        "breadth": round(avg_breadth, 2),
        "balance": round(balance, 4),
        "module_boundary_clarity": round(module_clarity, 4),
    }


def _compute_naming_consistency(files: list[dict]) -> dict:
    """Analyze naming convention consistency across the codebase."""
    snake = 0
    camel = 0
    pascal = 0
    other = 0
    total = 0
    for f in files:
        name = f["name"]
        if not name or f["ext"] == ".py" and name == "__init__.py":
            continue
        total += 1
        base = name.split(".")[0] if "." in name else name
        if re.match(r"^[a-z][a-z0-9_]*$", base):
            snake += 1
        elif re.match(r"^[a-z][a-z0-9]*([A-Z][a-z0-9]*)*$", base):
            camel += 1
        elif re.match(r"^[A-Z][a-z0-9]*([A-Z][a-z0-9]*)*$", base):
            pascal += 1
        else:
            other += 1

    total = max(total, 1)
    dominant = max(snake, camel, pascal, other)
    dominant_ratio = dominant / total if total else 0
    entropy = 0.0
    for c in [snake, camel, pascal, other]:
        if c:
            p = c / total
            entropy -= p * math.log2(p) if p > 0 else 0
    return {
        "snake_case": snake,
        "camelCase": camel,
        "PascalCase": pascal,
        "other": other,
        "dominant_ratio": round(dominant_ratio, 4),
        "entropy": round(entropy, 4),
    }


def _compute_file_type_diversity(files: list[dict]) -> dict:
    """Analyze file type distribution."""
    ext_counts: dict[str, int] = defaultdict(int)
    total_lines = 0
    for f in files:
        ext_counts[f["ext"]] += 1
        if f["lines"] > 0:
            total_lines += f["lines"]
    total_files = len(files)
    top_exts = sorted(ext_counts.items(), key=lambda x: -x[1])[:5]
    return {
        "total_files": total_files,
        "total_lines": total_lines,
        "unique_extensions": len(ext_counts),
        "extension_distribution": dict(top_exts),
    }


def _compute_structural_ont_self(topology: dict, naming: dict,
                                  file_stats: dict) -> float:
    """Compute a single structural ont_self score for the codebase.

    Higher = more coherent, self-consistent structure.
    Max theoretical ~1.0.
    """
    s = 0.0

    # Directory topology (40%)
    depth = topology["depth"]
    depth_score = 1.0 - min(1.0, depth / 10.0)  # shallower = better
    balance = topology["balance"]
    clarity = topology["module_boundary_clarity"]
    s += 0.10 * depth_score + 0.15 * balance + 0.15 * clarity

    # Naming consistency (30%)
    naming_entropy = naming["entropy"]
    naming_score = 1.0 - min(1.0, naming_entropy / 1.5)
    dominant_ratio = naming["dominant_ratio"]
    s += 0.15 * naming_score + 0.15 * dominant_ratio

    # File diversity (20%)
    if file_stats["total_files"] > 0:
        ext_div = min(1.0, file_stats["unique_extensions"] / 8.0)
        s += 0.20 * ext_div

    # Line size penalty (10%): neither too small nor too large
    avg_lines = (file_stats["total_lines"] / file_stats["total_files"]
                 if file_stats["total_files"] else 0)
    if avg_lines > 0:
        # optimal avg file = 50-200 lines
        if avg_lines < 20:
            line_score = avg_lines / 50.0
        elif avg_lines < 200:
            line_score = 1.0
        elif avg_lines < 500:
            line_score = 1.0 - (avg_lines - 200) / 300.0
        else:
            line_score = 0.3
    else:
        line_score = 0.5
    s += 0.10 * max(0, min(1, line_score))

    return round(s, 4)


def _compute_refactoring_suggestions(topology: dict, naming: dict,
                                      file_stats: dict) -> list[dict]:
    """Generate actionable refactoring suggestions."""
    suggestions = []
    if topology["depth"] > 6:
        suggestions.append({
            "type": "structure",
            "severity": "warning",
            "message": f"Deep directory nesting ({topology['depth']} levels). "
                       f"Consider flattening to ≤ 4 for better module cohesion.",
        })
    if topology["module_boundary_clarity"] < 0.5:
        suggestions.append({
            "type": "coupling",
            "severity": "warning",
            "message": f"Low module boundary clarity ({topology['module_boundary_clarity']:.2f}). "
                       f"Cross-directory imports exceed intra-directory imports. "
                       f"Consider regrouping modules by domain.",
        })
    if naming["entropy"] > 0.8:
        suggestions.append({
            "type": "consistency",
            "severity": "info",
            "message": f"Naming convention entropy is high ({naming['entropy']:.2f} bits). "
                       f"Mixing snake_case, camelCase, and PascalCase reduces scanability. "
                       f"Pick one convention per scope.",
        })
    if file_stats["total_files"] > 50 and topology["balance"] < 0.3:
        suggestions.append({
            "type": "balance",
            "severity": "info",
            "message": f"Directory size imbalance (balance={topology['balance']:.2f}). "
                       f"Some directories are very dense, others sparse. "
                       f"Consider redistributing files for even cognitive load.",
        })
    return suggestions


def code_review_audit(
    project_root: str,
    verbose: bool = False,
    lambda_estimate: Optional[float] = None,
    active_k: int = 2000,
    extra_ignore_dirs: Optional[set[str]] = None,
    extra_ignore_exts: Optional[set[str]] = None,
) -> dict[str, Any]:
    """Run a topological structure audit on a codebase.

    No LLM calls. No code execution.
    Pure structural analysis: directory topology, module coupling,
    naming consistency, file diversity.

    Parameters
    ----------
    project_root : str
        Path to the project root directory.
    verbose : bool
        Print progress to stderr.
    lambda_estimate : float or None
        Barrier economics lambda for dataset audit (None = auto).
    active_k : int
        Active memory pool size for the vinfty engine.
    extra_ignore_dirs : set[str] | None
        Extra directory names to skip (merged with built-in IGNORE_DIRS).
    extra_ignore_exts : set[str] | None
        Extra file extensions to skip (merged with built-in IGNORE_EXTS).

    Returns
    -------
    dict with keys:
        project, structural_ont_self, topology, naming_consistency,
        file_types, suggestions, code_health
    """
    root = os.path.abspath(project_root)
    if not os.path.isdir(root):
        return {"error": f"directory not found: {root}"}

    if verbose:
        print(f"[vinfty.judge] Scanning {root} ...")

    files = _scan_files(root, extra_ignore_dirs, extra_ignore_exts)
    if not files:
        return {"error": "no source files found (all ignored or empty)"}

    if verbose:
        print(f"[vinfty.judge]  Found {len(files)} files. Analyzing topology...")

    topology = _compute_directory_topology(files)
    naming = _compute_naming_consistency(files)
    file_stats = _compute_file_type_diversity(files)
    struct_ont = _compute_structural_ont_self(topology, naming, file_stats)
    suggestions = _compute_refactoring_suggestions(topology, naming, file_stats)

    # Code health summary
    if struct_ont >= 0.75:
        health = "healthy"
    elif struct_ont >= 0.50:
        health = "fair"
    elif struct_ont >= 0.30:
        health = "fragile"
    else:
        health = "critical"

    return {
        "project": root,
        "structural_ont_self": struct_ont,
        "code_health": health,
        "topology": topology,
        "naming_consistency": naming,
        "file_types": file_stats,
        "suggestions": suggestions,
        "n_files_analyzed": len(files),
    }
