"""Compatibility re-export for subagent execution.

Prefer importing SubagentExecutor from engine.agents.execution.executor.
"""

from engine.agents.execution.executor import SubagentExecutor

__all__ = ["SubagentExecutor"]
