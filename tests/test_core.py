import sys
sys.path.insert(0, "..")

from vinfty.core import V9Orchestrator, _HMM
from vinfty.barrier.table import (
    OPERATOR_TABLE, OPERATOR_NAMES,
    compute_l2_star, compute_l1_star, compute_l_total, compute_p_trans,
    compute_operator_optimal,
)
from vinfty.barrier.estimator import LambdaEstimator
from vinfty.barrier.scheduler import LambdaScheduler, CollapseDetector
from vinfty.judge import audit_dataset


# ═══════════════════════════════════════════════════════════════
# Existing V9 core tests (regression)
# ═══════════════════════════════════════════════════════════════

def test_empty_ont_self():
    engine = V9Orchestrator(active_k=50)
    assert engine.ont_self() == 0.0, "empty → ont_self = 0"
    print("  [OK] empty ont_self")


def test_single_memory():
    engine = V9Orchestrator(active_k=50)
    engine.step("hello", palace="P_test")
    assert engine.ont_self() == 0.0, "single memory -> ont_self = 0 (no pairs)"
    print("  [OK] single memory ont_self")


def test_same_palace():
    engine = V9Orchestrator(active_k=50)
    for i in range(5):
        engine.step(f"item {i}", palace="P_fixed")
    assert engine.ont_self() >= 0.5, f"same palace -> high ont_self (got {engine.ont_self():.4f})"
    print(f"  [OK] same palace ont_self = {engine.ont_self():.4f}")


def test_diff_palace():
    engine = V9Orchestrator(active_k=50)
    for i in range(5):
        engine.step(f"item {i}", palace=f"P_{i}")
    assert engine.ont_self() < 0.1, "diff palaces -> near 0 ont_self"
    print(f"  [OK] diff palace ont_self = {engine.ont_self():.4f}")


def test_hmm_initial():
    hmm = _HMM()
    assert hmm.state == "s0", "init state = s0"
    assert hmm.posterior_s1 == 0.0, "init posterior = 0"
    print("  [OK] HMM initial state")


def test_hmm_update():
    hmm = _HMM()
    hmm.update(0.9)  # large delta ratio -> s1
    assert hmm.state == "s1", "high delta ratio -> s1"
    hmm.update(0.0)  # small delta ratio -> s0
    assert hmm.state == "s0", "low delta ratio -> s0"
    print(f"  [OK] HMM state transition {hmm.state}")


def test_register_decorator():
    engine = V9Orchestrator()
    @engine.register(palace="P_test")
    def foo(): pass
    assert "foo" in engine._palace_registry
    assert engine._palace_registry["foo"]["palace"] == "P_test"
    print("  [OK] register decorator")


def test_report():
    engine = V9Orchestrator(active_k=50)
    for i in range(5):
        engine.step(f"item {i}", palace="P_fixed")
    r = engine.report()
    assert "ont_self" in r
    assert "hmm_state" in r
    assert "c_ij_density" in r
    assert "barrier" in r, "report must include barrier section"
    assert r["memory_count"] == 5
    print(f"  [OK] report keys={list(r.keys())}")


def test_trace():
    engine = V9Orchestrator(active_k=50)
    engine.step("hello", palace="P_test")
    t = engine.trace()
    assert len(t) == 1
    assert t[0]["palace"] == "P_test"
    print("  [OK] trace")


def test_active_pool_pruning():
    engine = V9Orchestrator(active_k=10)
    for i in range(20):
        engine.step(f"item {i}", palace="P_fixed")
    assert len(engine.memories) == 10
    assert len(engine.archive) == 10
    print("  [OK] active pool pruning")


# ═══════════════════════════════════════════════════════════════
# Barrier economics — pure math (P0)
# ═══════════════════════════════════════════════════════════════

def test_operator_table_shape():
    assert len(OPERATOR_TABLE) == 5, "exactly 5 operators"
    for name in ("ODE", "Fractal", "exp_log", "PRNG", "Series"):
        assert name in OPERATOR_TABLE, f"missing {name}"
        p = OPERATOR_TABLE[name]
        for k in ("L2_min", "L1_inf", "delta_L1", "L1_max", "tau"):
            assert k in p, f"{name} missing {k}"
    print("  [OK] operator table shape")


def test_ode_at_pc_lam():
    """PC λ=0.001 → ODE optimal, L_total*=22.17 (SiliconLifeOS benchmark)."""
    r = compute_operator_optimal(0.001, OPERATOR_TABLE["ODE"])
    assert abs(r["L_total_star"] - 22.17) < 0.05, f"got {r['L_total_star']}"
    print(f"  [OK] ODE @ λ=0.001: L_total*={r['L_total_star']}")


def test_pc_scenario_ode_is_best():
    """λ=0.001 → ODE is the cheapest operator."""
    best_name = None
    best_cost = float("inf")
    for name, params in OPERATOR_TABLE.items():
        r = compute_operator_optimal(0.001, params)
        if r["L_total_star"] < best_cost:
            best_cost = r["L_total_star"]
            best_name = name
    assert best_name == "ODE", f"expected ODE, got {best_name} ({best_cost})"
    print(f"  [OK] PC λ=0.001 best={best_name} cost={best_cost:.2f}")


def test_inflection_boundary():
    """When lambda*tau/delta_L1 >= 1, no inflection - uses L2_min, L1_max."""
    r = compute_operator_optimal(100.0, OPERATOR_TABLE["PRNG"])
    assert r["has_inflection"] == False, "PRNG at high λ has no inflection"
    assert r["L2_star"] == OPERATOR_TABLE["PRNG"]["L2_min"]
    assert r["L1_star"] == OPERATOR_TABLE["PRNG"]["L1_max"]
    print(f"  [OK] inflection boundary: L2=L2_min={r['L2_star']}")


def test_p_trans_range():
    """P_trans ∈ (0, 1]."""
    p = compute_p_trans(0, 0)
    assert p == 1.0, "no barriers → P=1"
    p2 = compute_p_trans(100, 50000)
    assert 0 < p2 < 0.1, "thick barriers → P≈0"
    print(f"  [OK] P_trans range (0,1]")


def test_fractal_cheapest_at_low_lam():
    """λ=1e-5 → Fractal is cheapest (deep nesting GPU)."""
    best_name = None
    best_cost = float("inf")
    for name, params in OPERATOR_TABLE.items():
        r = compute_operator_optimal(1e-5, params)
        if r["L_total_star"] < best_cost:
            best_cost = r["L_total_star"]
            best_name = name
    assert best_name == "Fractal", f"expected Fractal, got {best_name}"
    print(f"  [OK] GPU λ=1e-5 best={best_name} cost={best_cost:.2f}")


# ═══════════════════════════════════════════════════════════════
# LambdaEstimator (P1)
# ═══════════════════════════════════════════════════════════════

def test_estimator_default_is_pc():
    """Default params → λ ≈ 0.001."""
    est = LambdaEstimator()
    lam = est.estimate()
    assert 0.0005 < lam < 0.002, f"expected ~0.001, got {lam}"
    print(f"  [OK] estimator default λ={lam}")


def test_estimator_scales_down():
    """Better hardware → lower λ."""
    est = LambdaEstimator()
    base = est.estimate()
    gpu = est.estimate(cpu_freq_ghz=4.0, mem_bw_gbps=2000, cache_per_core_mb=64, cores_available=64)
    assert gpu < base, f"GPU λ={gpu} should be < base={base}"
    print(f"  [OK] GPU λ={gpu} < base={base}")


def test_estimator_scales_up():
    """Worse hardware → higher λ."""
    est = LambdaEstimator()
    base = est.estimate()
    edge = est.estimate(cpu_freq_ghz=1.2, mem_bw_gbps=10, cache_per_core_mb=1, cores_available=4)
    assert edge > base, f"Edge λ={edge} should be > base={base}"
    print(f"  [OK] Edge λ={edge} > base={base}")


# ═══════════════════════════════════════════════════════════════
# LambdaScheduler (P2)
# ═══════════════════════════════════════════════════════════════

def test_scheduler_evaluate():
    sched = LambdaScheduler(lam=0.001)
    r = sched.evaluate()
    assert r["recommended_operator"] == "ODE"
    assert "scenario" in r
    assert "operators" in r
    assert len(r["operators"]) == 5
    assert "best" in r
    print(f"  [OK] scheduler evaluate: {r['recommended_operator']} @ {r['scenario']}")


def test_scheduler_recommend():
    sched = LambdaScheduler(lam=0.001)
    rec = sched.recommend()
    assert "operation" in rec
    assert "target_operator" in rec
    assert "collapse_risk" in rec
    assert rec["target_operator"] == "ODE"
    print(f"  [OK] scheduler recommend: {rec['operation']} → {rec['target_operator']}")


def test_collapse_detector():
    cd = CollapseDetector(epsilon=0.01, history_window=3)
    # Feed stagnant data
    for _ in range(5):
        r = cd.update(l1=50, l2=5000, i_struct=0.5, depth=1)
    assert cd.collapse_count > 0, "should have detected collapse"
    print(f"  [OK] collapse detected ({cd.collapse_count}x)")


def test_scheduler_scenario_table():
    sched = LambdaScheduler()
    table = sched.scenario_table()
    assert "pc_benchmark" in table
    assert "datacenter_gpu" in table
    assert "ODE" in table
    print("  [OK] scenario table generated")


# ═══════════════════════════════════════════════════════════════
# V9Orchestrator integration (P3)
# ═══════════════════════════════════════════════════════════════

def test_orchestrator_default_mode():
    """No lambda_estimate → auto-detect → mode is some valid operator."""
    engine = V9Orchestrator()
    assert engine._mode in OPERATOR_NAMES, f"unknown mode {engine._mode}"
    print(f"  [OK] auto-detect mode: {engine._mode}")


def test_orchestrator_edge_mode():
    """lambda_estimate=0.1 → mode=exp_log."""
    engine = V9Orchestrator(lambda_estimate=0.1)
    assert engine._mode == "exp_log", f"expected exp_log, got {engine._mode}"
    print(f"  [OK] edge mode: {engine._mode}")


def test_orchestrator_gpu_mode():
    """lambda_estimate=1e-5 → mode=Fractal."""
    engine = V9Orchestrator(lambda_estimate=1e-5)
    assert engine._mode == "Fractal", f"expected Fractal, got {engine._mode}"
    print(f"  [OK] GPU mode: {engine._mode}")


def test_orchestrator_adapt():
    """adapt() switches mode at runtime."""
    engine = V9Orchestrator(lambda_estimate=0.001)
    assert engine._mode == "ODE"
    engine.adapt(lambda_estimate=1e-5)
    assert engine._mode == "Fractal", f"adapt failed, got {engine._mode}"
    print(f"  [OK] adapt: ODE → Fractal")


def test_report_includes_barrier():
    engine = V9Orchestrator(lambda_estimate=0.001)
    engine.step("hello", palace="P_test")
    r = engine.report()
    assert "barrier" in r
    b = r["barrier"]
    assert "lambda" in b
    assert "mode" in b
    assert "scenario" in b
    assert b["mode"] == "ODE"
    print(f"  [OK] report.barrier: λ={b['lambda']}, mode={b['mode']}")


# ═══════════════════════════════════════════════════════════════
# judge (P4)
# ═══════════════════════════════════════════════════════════════

def test_judge_empty():
    report = audit_dataset([], lambda_estimate=0.001)
    assert report["n_samples"] == 0
    assert report["ont_self_mean"] == 0.0
    print("  [OK] judge empty dataset")


def test_judge_basic():
    texts = ["hello", "world", "foo", "bar", "baz"]
    report = audit_dataset(texts, lambda_estimate=0.001)
    assert report["n_samples"] == 5
    assert isinstance(report["ont_self_mean"], float)
    assert "contradictions_n" in report
    assert "palace_distribution" in report
    assert "silent_collapse_risk" in report
    assert "barrier" in report
    print(f"  [OK] judge basic: ont={report['ont_self_mean']}, palaces={report['n_palaces']}")


def test_judge_with_labels():
    texts = ["cat", "dog", "cnn", "transformer", "paris", "beijing"]
    labels = ["animal", "animal", "ML", "ML", "geo", "geo"]
    report = audit_dataset(texts, labels=labels, lambda_estimate=0.001)
    assert "label_conflicts" in report
    # With hash-based palace assignment, labels should conflict
    print(f"  [OK] judge with labels: conflicts={len(report['label_conflicts'])}")


def test_judge_all_same_palace():
    texts = ["a", "b", "c", "d", "e"]
    palaces = ["P_same"] * 5
    report = audit_dataset(texts, palaces=palaces, lambda_estimate=0.001)
    assert report["ont_self_mean"] >= 0.5, f"same palace -> high ont_self ({report['ont_self_mean']})"
    assert report["c_ij_density"] == 1.0, "all same -> full coupling"
    print(f"  [OK] judge all-same-palace: ont={report['ont_self_mean']}")


def test_judge_all_diff_palace():
    texts = ["a", "b", "c", "d", "e"]
    palaces = [f"P_{i}" for i in range(5)]
    report = audit_dataset(texts, palaces=palaces, lambda_estimate=0.001)
    assert report["ont_self_mean"] < 0.1, f"diff palaces -> low ont_self ({report['ont_self_mean']})"
    assert report["contradictions_n"] >= 0
    print(f"  [OK] judge all-diff-palace: ont={report['ont_self_mean']}, contradictions={report['contradictions_n']}")


def test_judge_lambda_control():
    texts = ["a", "b", "c"]
    r1 = audit_dataset(texts, lambda_estimate=0.001)
    r2 = audit_dataset(texts, lambda_estimate=100.0)
    assert r1["n_samples"] == r2["n_samples"] == 3
    assert r1["barrier"]["lambda"] < r2["barrier"]["lambda"]
    print(f"  [OK] judge lambda control: 0.001 vs 100.0")


# ═══════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═══ Core regression (10) ═══")
    test_empty_ont_self()
    test_single_memory()
    test_same_palace()
    test_diff_palace()
    test_hmm_initial()
    test_hmm_update()
    test_register_decorator()
    test_report()
    test_trace()
    test_active_pool_pruning()

    print("\n═══ Barrier table (P0) 6 ═══")
    test_operator_table_shape()
    test_ode_at_pc_lam()
    test_pc_scenario_ode_is_best()
    test_inflection_boundary()
    test_p_trans_range()
    test_fractal_cheapest_at_low_lam()

    print("\n═══ Estimator (P1) 3 ═══")
    test_estimator_default_is_pc()
    test_estimator_scales_down()
    test_estimator_scales_up()

    print("\n═══ Scheduler (P2) 4 ═══")
    test_scheduler_evaluate()
    test_scheduler_recommend()
    test_collapse_detector()
    test_scheduler_scenario_table()

    print("\n═══ Integration (P3) 5 ═══")
    test_orchestrator_default_mode()
    test_orchestrator_edge_mode()
    test_orchestrator_gpu_mode()
    test_orchestrator_adapt()
    test_report_includes_barrier()

    print("\n═══ Judge (P4) 6 ═══")
    test_judge_empty()
    test_judge_basic()
    test_judge_with_labels()
    test_judge_all_same_palace()
    test_judge_all_diff_palace()
    test_judge_lambda_control()

    print(f"\n==> 34/34 tests passed")
