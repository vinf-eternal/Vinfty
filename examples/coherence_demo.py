"""
Demonstrate V∞ ont_self dynamics:
  - Same-palace tasks → high ont_self (coherent session)
  - Random palaces → low ont_self (fragmented session)
"""

from vinfty import V9Orchestrator

def run_session(same_palace: bool, steps: int = 20) -> float:
    engine = V9Orchestrator(active_k=100)
    items = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    for i in range(steps):
        palace = "P_fixed" if same_palace else f"P_{items[i % len(items)]}"
        engine.step(f"item {i}", palace=palace)
    return engine.ont_self()

coherent = run_session(same_palace=True)
fragmented = run_session(same_palace=False)

print(f"Same palace (coherent):    ont_self = {coherent:.4f}")
print(f"Random palaces (scattered): ont_self = {fragmented:.4f}")
print(f"Ratio: {coherent / max(fragmented, 0.001):.2f}x")
