"""vinfty — AI consistency auxiliary auditor.

Vinfty does NOT train models, fit labels, or make predictions.
It audits: self-consistency (ont_self), coupling coherence (C_ij),
cognitive drift (HMM s0/s1), and silence collapse risk.

For AI engineers who need to know *whether their system is healthy*,
not just whether it returned valid JSON.
"""

from . import barrier
from . import judge
from .core import V9Orchestrator

__version__ = "0.2.0"
