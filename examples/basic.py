"""
Quick start example: V∞ cognitive observability without any LLM dependency.

Demonstrates ont_self tracking, HMM state detection, and Palace routing
using pure symbolic computation.
"""

from vinfty import V9Orchestrator

engine = V9Orchestrator(active_k=50)

# Register tools with Palace domains
@engine.register(palace="P_search")
def search_web(q: str) -> str:
    return f"mock results for {q}"

@engine.register(palace="P_analysis")
def calculate(expr: str) -> float:
    return eval(expr)

@engine.register(palace="P_memory")
def save_note(text: str) -> None:
    pass

# Simulate a cognitive session — same-palace tasks boost ont_self
tasks = [
    ("search latest ai papers", search_web),
    ("find transformer tutorials", search_web),
    ("analyze attention formula", calculate),
    ("2 + 2 = ?", calculate),
    ("note: transformer is encoder-decoder", save_note),
    ("search gpt-4 benchmarks", search_web),
    ("analyze benchmark results", calculate),
]

# Feed tasks into the engine
for content, tool in tasks:
    engine.step(content, tool=tool)

# Report
r = engine.report()
print("=== V∞ Cognitive Report ===")
for k, v in r.items():
    print(f"  {k}: {v}")

# Trace
print("\n=== Memory Trace ===")
for m in engine.trace():
    print(f"  [{m['palace']:>14}] {m['content']}")
