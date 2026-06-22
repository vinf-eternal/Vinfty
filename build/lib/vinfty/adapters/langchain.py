"""
vinfty.adapters.langchain — wrap a LangChain agent with cognitive observability.

Usage:
    from vinfty.adapters.langchain import LangChainAdapter

    adapter = LangChainAdapter(agent)
    result, report = adapter.run("query")
    print(report["ont_self"])
"""

from typing import Any, Optional

from vinfty.core import V9Orchestrator


class LangChainAdapter:
    """Wraps a LangChain agent and adds V9 cognitive observability."""

    def __init__(self, agent: Any, active_k: int = 100):
        self.agent = agent
        self.cog = V9Orchestrator(active_k=active_k)

    def run(self, query: str, **kwargs) -> tuple[str, dict]:
        """Run query through agent, return (result, cognitive_report)."""
        self.cog.step(query, palace="P_query")
        result = self.agent.run(query, **kwargs)
        self.cog.step(result, palace="P_response")
        return result, self.cog.report()

    def run_with_observability(self, query: str, **kwargs) -> str:
        """Run and print cognitive report alongside result."""
        result, report = self.run(query, **kwargs)
        print(f"[V∞] ont_self={report['ont_self']}  state={report['hmm_state']}  "
              f"palaces={report['palace_count']}  c_density={report['c_ij_density']}")
        return result
