"""Bounded tool-calling loop: search the notes, then answer with citations."""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from verity.agent.tools import SEARCH_TOOL_SPEC, SYSTEM_PROMPT
from verity.interfaces import ChatModel
from verity.models import Answer, Citation, ScoredChunk
from verity.telemetry import span

RetrieveFn = Callable[[str], list[ScoredChunk]]

_CITATION_RE = re.compile(r"\[([^\[\]]+?::\d+)\]")


class RagAgent:
    def __init__(self, model: ChatModel, retrieve: RetrieveFn, max_steps: int = 4) -> None:
        self._model = model
        self._retrieve = retrieve
        self._max_steps = max_steps

    def answer(self, question: str) -> Answer:
        with span("agent.answer", question=question) as root:
            messages: list[dict[str, object]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
            seen: dict[str, ScoredChunk] = {}

            for step in range(self._max_steps):
                with span("agent.step", step=step):
                    msg = self._model.chat(messages, tools=[SEARCH_TOOL_SPEC])
                    messages.append(msg)
                    raw_calls = msg.get("tool_calls")
                    tool_calls = raw_calls if isinstance(raw_calls, list) else []
                    if not tool_calls:
                        return self._finalize(question, str(msg.get("content", "")), seen, root)
                    for call in tool_calls:
                        contexts = self._run_tool(call)
                        for c in contexts:
                            seen[c.id] = c
                        messages.append(
                            {
                                "role": "tool",
                                "content": _format_contexts(contexts),
                            }
                        )

            # out of steps: force a final answer with no more tools
            final = self._model.chat(messages, tools=None)
            return self._finalize(question, str(final.get("content", "")), seen, root)

    def _run_tool(self, call: dict[str, object]) -> list[ScoredChunk]:
        fn = call.get("function", {}) if isinstance(call, dict) else {}
        args = fn.get("arguments", {}) if isinstance(fn, dict) else {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"query": args}
        query = args.get("query", "") if isinstance(args, dict) else ""
        with span("agent.tool.search", query=query):
            return self._retrieve(query)

    def _finalize(
        self,
        question: str,
        text: str,
        seen: dict[str, ScoredChunk],
        root: object,
    ) -> Answer:
        cited_ids = set(_CITATION_RE.findall(text))
        citations = [
            Citation(
                chunk_id=cid,
                doc_id=seen[cid].chunk.doc_id,
                snippet=seen[cid].chunk.text[:200],
            )
            for cid in cited_ids
            if cid in seen
        ]
        trace_id = None
        ctx = getattr(root, "get_span_context", None)
        if callable(ctx):
            trace_id = format(ctx().trace_id, "032x")
        return Answer(
            query=question,
            text=text,
            citations=citations,
            contexts=sorted(seen.values(), key=lambda s: s.score, reverse=True),
            trace_id=trace_id,
        )


def _format_contexts(contexts: list[ScoredChunk]) -> str:
    if not contexts:
        return "No matching notes found."
    return "\n\n".join(f"[{c.id}] {c.chunk.text}" for c in contexts)
