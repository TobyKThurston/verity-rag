"""The single tool the agent gets: search_knowledge_base.

Giving it only this tool forces it to ground answers in the notes rather than
its own training data.
"""

from __future__ import annotations

SEARCH_TOOL_SPEC: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "Search the user's personal notes for passages relevant to a query. "
            "Always call this before answering; never answer from prior knowledge."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A focused search query derived from the user's question.",
                }
            },
            "required": ["query"],
        },
    },
}

SYSTEM_PROMPT = (
    "You are Verity, a careful study assistant that answers strictly from the user's "
    "own notes. Use the search_knowledge_base tool to find relevant passages, then "
    "answer using ONLY that retrieved context. Cite the supporting note ids in square "
    "brackets, e.g. [04-mughal-empire.md::1]. If the notes do not contain the answer, "
    "say you don't know. Never invent facts."
)
