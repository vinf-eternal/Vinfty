"""LambdaEstimator — map hardware metrics to λ.

``λ = cognitive_cost_per_tick / compute_cost_per_tick``

Higher λ means compute is expensive relative to cognition.
Hardware-rich environments (GPU cluster) → λ ≈ 1e-5
Personal PC                     → λ ≈ 0.001
Edge / MCU                      → λ ≈ 0.1 .. 100
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class LambdaEstimator:
    """Estimate λ from hardware metrics.

    Reference: personal PC ≈ 0.001 at 3.0 GHz, 50 GB/s, 8 MB cache.
    """

    base_lambda: float = 0.001
    ref_freq_ghz: float = 3.0
    ref_bw_gbps: float = 50.0
    ref_cache_mb: float = 8.0

    def estimate(
        self,
        cpu_freq_ghz: Optional[float] = None,
        mem_bw_gbps: Optional[float] = None,
        cache_per_core_mb: Optional[float] = None,
        cores_available: int = 8,
        task_idle_ratio: float = 0.3,
        network_latency_ms: float = 0.0,
    ) -> float:
        """Estimate λ given hardware characteristics.

        Returns λ in [1e-6, 1e3].
        """
        freq = cpu_freq_ghz if cpu_freq_ghz is not None else self.ref_freq_ghz
        bw = mem_bw_gbps if mem_bw_gbps is not None else self.ref_bw_gbps
        cache = cache_per_core_mb if cache_per_core_mb is not None else self.ref_cache_mb

        lam = self.base_lambda

        # CPU frequency: slower → higher λ
        lam *= self.ref_freq_ghz / max(0.1, freq)

        # Memory bandwidth: lower → higher λ
        lam *= self.ref_bw_gbps / max(0.1, bw)

        # Cache: smaller → higher λ
        lam *= self.ref_cache_mb / max(0.1, cache)

        # Cores: fewer → higher λ
        lam *= 8.0 / max(1, cores_available)

        # Idle ratio: more idle → lower λ
        lam *= (1.0 + task_idle_ratio) / 1.3

        # Network latency: high → penalizes distributed compute
        if network_latency_ms > 100:
            lam *= 1.0 + min(1.0, network_latency_ms / 1000)

        return round(lam, 8)


def auto_detect_lambda() -> float:
    """Auto-detect λ from platform hardware.

    Uses ``platform`` and ``os`` only. Falls back to 0.001 (PC).
    """
    import os
    import platform

    system = platform.system()

    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                cpuinfo = f.read()
            freq_lines = [l for l in cpuinfo.splitlines() if "cpu MHz" in l]
            freq_mhz = float(freq_lines[0].split(":")[1].strip()) if freq_lines else 3000.0
            cpu_freq_ghz = freq_mhz / 1000.0
        except Exception:
            cpu_freq_ghz = 3.0

        try:
            cores = os.cpu_count() or 8
        except Exception:
            cores = 8

        try:
            with open("/proc/meminfo") as f:
                meminfo = f.read()
            total_line = [l for l in meminfo.splitlines() if "MemTotal" in l]
            total_kb = float(total_line[0].split()[1]) if total_line else 16000000.0
            mem_bw_gbps = max(10.0, total_kb / 1000000.0 * 3.0)  # rough: 3x scaling
        except Exception:
            mem_bw_gbps = 50.0

        cache_mb = 8.0  # best-effort default

    elif system == "Windows":
        try:
            import subprocess
            result = subprocess.run(
                ["wmic", "cpu", "get", "MaxClockSpeed"],
                capture_output=True, text=True, timeout=5
            )
            lines = [l.strip() for l in result.stdout.splitlines() if l.strip().isdigit()]
            max_mhz = float(lines[0]) if lines else 3000.0
            cpu_freq_ghz = max_mhz / 1000.0
        except Exception:
            cpu_freq_ghz = 3.0

        try:
            cores = os.cpu_count() or 8
        except Exception:
            cores = 8

        mem_bw_gbps = 50.0
        cache_mb = 8.0

    elif system == "Darwin":
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.cpufrequency_max"],
                capture_output=True, text=True, timeout=5
            )
            freq_hz = float(result.stdout.strip()) if result.stdout.strip() else 3000000000.0
            cpu_freq_ghz = freq_hz / 1e9
        except Exception:
            cpu_freq_ghz = 3.0

        try:
            cores = os.cpu_count() or 8
        except Exception:
            cores = 8

        mem_bw_gbps = 50.0
        cache_mb = 8.0

    else:
        cpu_freq_ghz = 3.0
        cores = os.cpu_count() or 8
        mem_bw_gbps = 50.0
        cache_mb = 8.0

    return LambdaEstimator().estimate(
        cpu_freq_ghz=cpu_freq_ghz,
        mem_bw_gbps=mem_bw_gbps,
        cache_per_core_mb=cache_mb,
        cores_available=cores,
    )
