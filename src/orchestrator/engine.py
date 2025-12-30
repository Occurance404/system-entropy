"""
Compatibility wrapper.

The orchestrator implementation lives under `src/orchestrator/core/` to keep the
surface module small and stable for imports like:
    from src.orchestrator.engine import Orchestrator
"""

from .core.orchestrator import Orchestrator

__all__ = ["Orchestrator"]

