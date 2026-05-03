"""Offline chat model used when no LLM is running (--no-models, tests, CI).

Drives the same agent loop as the real model: first turn asks to search, second
turn answers by picking the context sentences that overlap the question, or
abstains. No language understanding, but it runs the whole pipeline end to end.
"""

from __future__ import annotations

import re

from verity.store.bm25 import tokenize

_SENT_RE = re.compile(r"(?<=[.!?])\s+")
# tool messages look like "[chunk::id] sentence text" blocks
_CTX_RE = re.compile(r"\[([^\[\]]+?::\d+)\]\s*(.*?)(?=\n\n\[|\Z)", re.S)

# stopwords so abstention keys on content words, not "the"/"is"/"of"
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from",
        "how", "in", "into", "is", "it", "its", "of", "on", "or", "so", "that",
        "the", "to", "under", "we", "what", "when", "which", "while", "with",
        "without", "you", "your",
    }
)


def _content_terms(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in _STOPWORDS}


class ExtractiveChatModel:
    def __init__(self, max_sentences: int = 3) -> None:
        self._max_sentences = max_sentences

    def chat(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        question = self._first_user(messages)
        tool_context = self._latest_tool_content(messages)

        # No search has run yet and a tool is offered -> ask to search.
        if tool_context is None and tools:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_knowledge_base",
                            "arguments": {"query": question},
                        }
                    }
                ],
            }

        return {"role": "assistant", "content": self._answer(question, tool_context or "")}

    @staticmethod
    def _first_user(messages: list[dict[str, object]]) -> str:
        for m in messages:
            if m.get("role") == "user":
                return str(m.get("content", ""))
        return ""

    @staticmethod
    def _latest_tool_content(messages: list[dict[str, object]]) -> str | None:
        for m in reversed(messages):
            if m.get("role") == "tool":
                return str(m.get("content", ""))
        return None

    def _answer(self, question: str, context: str) -> str:
        passages = _CTX_RE.findall(context)
        if not passages or "No matching notes" in context:
            return "I don't know based on these notes."

        q_terms = _content_terms(question)
        scored: list[tuple[int, str, str]] = []
        for cid, text in passages:
            for sent in _SENT_RE.split(text.strip()):
                sent = sent.strip()
                if not sent:
                    continue
                overlap = len(q_terms & _content_terms(sent))
                scored.append((overlap, sent, cid))

        scored.sort(key=lambda t: t[0], reverse=True)
        # only keep sentences that share a content word; otherwise abstain
        top = [s for s in scored[: self._max_sentences] if s[0] > 0]
        if not top:
            return "I don't know based on these notes."

        sentences = " ".join(sent for _, sent, _ in top)
        cited = list(dict.fromkeys(cid for _, _, cid in top))
        return sentences + " " + " ".join(f"[{cid}]" for cid in cited)
