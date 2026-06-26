"""
vinfty.structural_boundaries — module boundary enforcement engine.

Reads .vinfty.yml from project root, scans import statements across
all Python files, reports any import that violates allowed directions.

Usage::

    from vinfty.structural_boundaries import check_boundaries

    violations = check_boundaries("/path/to/project")
    for v in violations:
        print(f"  {v['source']} -> {v['target']}  [{v['rule']}]")
"""

from __future__ import annotations

import ast
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import yaml


DEFAULT_CONFIG = {
    "boundaries": {
        "kernel": {
            "allowed_imports": ["stdlib", "vinfty"],
            "allow_exports_to": ["*"],
        },
        "tools": {
            "allowed_imports": ["stdlib", "vinfty", "kernel"],
            "allow_exports_to": [],
        },
        "tests": {
            "allowed_imports": ["stdlib", "vinfty", "kernel"],
            "allow_exports_to": [],
        },
        "plugins": {
            "allowed_imports": ["stdlib", "kernel"],
            "allow_exports_to": [],
        },
    },
    "import_analysis": {
        "max_file_lines": 500,
        "max_depth": 5,
        "skip_dirs": [
            "__pycache__", "node_modules", ".git", ".venv",
            ".mypy_cache", ".ruff_cache", ".pytest_cache",
        ],
    },
}

STDLIB_MODULES = {
    "os", "sys", "re", "json", "math", "random", "time", "datetime",
    "collections", "itertools", "functools", "pathlib", "typing",
    "abc", "copy", "enum", "hashlib", "hmac", "io", "logging",
    "pickle", "statistics", "string", "struct", "tempfile",
    "textwrap", "threading", "uuid", "warnings", "weakref",
    "ast", "argparse", "base64", "bisect", "csv", "dis",
    "filecmp", "fnmatch", "glob", "gzip", "heapq", "importlib",
    "inspect", "numbers", "operator", "shutil", "signal", "socket",
    "ssl", "subprocess", "tomllib", "traceback", "types",
    "unittest", "xml", "zipfile", "zoneinfo",
}


def _discover_py_files(root: str, skip_dirs: list[str]) -> list[dict]:
    """Find all Python files and extract their imports via AST.

    Each import record includes:
      - module: the imported module name
      - line: source line number
      - scope: 'module' if top-level, 'function' if inside a function body
    """
    files = []
    root_p = Path(root).resolve()
    for abs_path in root_p.rglob("*.py"):
        parts = abs_path.relative_to(root_p).parts
        if any(p in skip_dirs for p in parts):
            continue
        try:
            raw = abs_path.read_bytes()
            if len(raw) > 500_000:
                continue
            tree = ast.parse(raw)
        except (SyntaxError, Exception):
            continue

        # Build set of function body line ranges (including nested functions)
        func_lines: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for n in ast.walk(node):
                    if hasattr(n, "lineno") and n.lineno != node.lineno:
                        func_lines.add(n.lineno)

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                scope = "function" if node.lineno in func_lines else "module"
                for alias in node.names:
                    imports.append({
                        "module": alias.name,
                        "line": node.lineno,
                        "scope": scope,
                    })
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    scope = "function" if node.lineno in func_lines else "module"
                    imports.append({
                        "module": node.module,
                        "line": node.lineno,
                        "scope": scope,
                    })

        rel = str(abs_path.relative_to(root_p))
        files.append({
            "path": str(abs_path),
            "rel": rel,
            "dir": rel.replace("\\", "/").split("/")[0],
            "imports": imports,
        })
    return files


def _classify_import(module: str, project_root: str) -> dict:
    """Classify an import into a boundary category.

    Returns {"domain": ..., "name": ...} where domain is one of
    stdlib, vinfty, kernel, tools, tests, browser, rtl, research,
    plugins, unknown.
    """
    top = module.split(".")[0]

    if top in STDLIB_MODULES or top.startswith("_"):
        return {"domain": "stdlib", "name": module}

    if top == "vinfty":
        return {"domain": "vinfty", "name": module}

    # Check if it's a local module by scanning project root
    local_dirs = {
        "kernel", "tools", "tests", "browser", "rtl",
        "research", "plugins", "docs",
    }
    if top in local_dirs:
        return {"domain": top, "name": module}

    # Check if it's a relative import (starts with .)
    if module.startswith("."):
        return {"domain": "relative", "name": module}

    # Third-party package
    return {"domain": "third_party", "name": module}


def _load_config(project_root: str) -> dict:
    """Load .vinfty.yml or return defaults."""
    candidates = [
        os.path.join(project_root, ".vinfty.yml"),
        os.path.join(project_root, ".vinfty.yaml"),
        os.path.join(project_root, "vinfty.yml"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                if cfg and "boundaries" in cfg:
                    return cfg
            except Exception:
                pass
    return DEFAULT_CONFIG


def check_boundaries(
    project_root: str,
    verbose: bool = False,
) -> dict[str, Any]:
    """Scan all Python files in project_root and check boundary rules.

    Returns
    -------
    dict with keys:
        total_files, total_imports, violations (list), by_domain (dict),
        boundary_clarity_score
    """
    config = _load_config(project_root)
    boundaries = config.get("boundaries", DEFAULT_CONFIG["boundaries"])
    skip_dirs = config.get("import_analysis", {}).get(
        "skip_dirs", DEFAULT_CONFIG["import_analysis"]["skip_dirs"]
    )

    if verbose:
        print(f"[vinfty.boundaries] Scanning {project_root} ...")

    files = _discover_py_files(project_root, skip_dirs)
    if verbose:
        print(f"[vinfty.boundaries]  Found {len(files)} Python files.")

    violations = []
    total_imports = 0
    by_domain: dict[str, dict] = defaultdict(lambda: {"files": 0, "imports": 0, "violations": 0})

    for f in files:
        source_domain = f["dir"]
        by_domain[source_domain]["files"] += 1
        by_domain[source_domain]["imports"] += len(f["imports"])

        for imp_rec in f["imports"]:
            total_imports += 1
            module_name = imp_rec["module"]
            scope = imp_rec.get("scope", "module")
            classified = _classify_import(module_name, project_root)
            target_domain = classified["domain"]

            # Same-domain imports are always allowed (intra-module)
            if source_domain == target_domain:
                continue

            # stdlib and relative imports are always allowed
            if target_domain in ("stdlib", "relative", "third_party"):
                continue

            # vinfty is always allowed
            if target_domain == "vinfty":
                continue

            # Check if source domain has rules
            if source_domain not in boundaries:
                continue  # unknown domain, no rules

            rule = boundaries[source_domain]
            allowed = rule.get("allowed_imports", [])

            # Check if target domain is allowed
            if target_domain not in allowed and "*" not in allowed:
                # Function-level imports are less severe (lazy evaluation)
                sev = "info" if scope == "function" else "warning"
                violations.append({
                    "source": f["rel"],
                    "source_domain": source_domain,
                    "target": module_name,
                    "target_domain": target_domain,
                    "line": imp_rec.get("line", 0),
                    "scope": scope,
                    "severity": sev,
                    "rule": f"'{source_domain}' cannot import '{target_domain}'",
                })
                by_domain[source_domain]["violations"] += 1

    # Weighted clarity: function-level violations count 0.3×
    weighted_violations = sum(
        0.3 if v.get("scope") == "function" else 1.0
        for v in violations
    )
    violation_rate = (weighted_violations / max(total_imports, 1))
    clarity = round(1.0 - violation_rate, 4)

    # Sort violations by source domain
    violations.sort(key=lambda v: (v["source_domain"], v["source"]))

    return {
        "total_py_files": len(files),
        "total_imports": total_imports,
        "violations": violations[:50],  # cap at 50 for readability
        "violations_total": len(violations),
        "violation_rate": round(violation_rate, 4),
        "boundary_clarity": clarity,
        "by_domain": {
            k: dict(v) for k, v in sorted(by_domain.items())
        },
    }
