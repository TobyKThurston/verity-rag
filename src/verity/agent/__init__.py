"""The tool-calling agent that turns retrieved context into a grounded answer."""

from verity.agent.loop import RagAgent
from verity.agent.tools import SEARCH_TOOL_SPEC

__all__ = ["RagAgent", "SEARCH_TOOL_SPEC"]
