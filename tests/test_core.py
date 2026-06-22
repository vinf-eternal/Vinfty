import sys
sys.path.insert(0, "..")

from vinfty.core import V9Orchestrator, _HMM


def test_empty_ont_self():
    engine = V9Orchestrator(active_k=50)
    assert engine.ont_self() == 0.0, "empty → ont_self = 0"
    print("  ✅ empty ont_self")


def test_single_memory():
    engine = V9Orchestrator(active_k=50)
    engine.step("hello", palace="P_test")
    assert engine.ont_self() == 0.0, "single memory → ont_self = 0 (no pairs)"
    print("  ✅ single memory ont_self")


def test_same_palace():
    engine = V9Orchestrator(active_k=50)
    for i in range(5):
        engine.step(f"item {i}", palace="P_fixed")
    assert engine.ont_self() >= 0.5, f"same palace → high ont_self (got {engine.ont_self():.4f})"
    print(f"  ✅ same palace ont_self = {engine.ont_self():.4f}")


def test_diff_palace():
    engine = V9Orchestrator(active_k=50)
    for i in range(5):
        engine.step(f"item {i}", palace=f"P_{i}")
    assert engine.ont_self() < 0.1, "diff palaces → near 0 ont_self"
    print(f"  ✅ diff palace ont_self = {engine.ont_self():.4f}")


def test_hmm_initial():
    hmm = _HMM()
    assert hmm.state == "s0", "init state = s0"
    assert hmm.posterior_s1 == 0.0, "init posterior = 0"
    print("  ✅ HMM initial state")


def test_hmm_update():
    hmm = _HMM()
    hmm.update(0.9)  # large delta ratio → s1
    assert hmm.state == "s1", "high delta ratio → s1"
    hmm.update(0.0)  # small delta ratio → s0
    assert hmm.state == "s0", "low delta ratio → s0"
    print(f"  ✅ HMM state transition {hmm.state}")


def test_register_decorator():
    engine = V9Orchestrator()
    @engine.register(palace="P_test")
    def foo(): pass
    assert "foo" in engine._palace_registry
    assert engine._palace_registry["foo"]["palace"] == "P_test"
    print("  ✅ register decorator")


def test_report():
    engine = V9Orchestrator(active_k=50)
    for i in range(5):
        engine.step(f"item {i}", palace="P_fixed")
    r = engine.report()
    assert "ont_self" in r
    assert "hmm_state" in r
    assert "c_ij_density" in r
    assert r["memory_count"] == 5
    print(f"  ✅ report keys={list(r.keys())}")


def test_trace():
    engine = V9Orchestrator(active_k=50)
    engine.step("hello", palace="P_test")
    t = engine.trace()
    assert len(t) == 1
    assert t[0]["palace"] == "P_test"
    print("  ✅ trace")


def test_active_pool_pruning():
    engine = V9Orchestrator(active_k=10)
    for i in range(20):
        engine.step(f"item {i}", palace="P_fixed")
    assert len(engine.memories) == 10
    assert len(engine.archive) == 10
    print("  ✅ active pool pruning")


if __name__ == "__main__":
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
    print("\n🎯 10/10 tests passed")
