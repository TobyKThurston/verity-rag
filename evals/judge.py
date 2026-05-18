"""Score answers on faithfulness (grounded in context?) and relevancy.

LlmJudge asks a model to score with a JSON rubric. HeuristicJudge approximates
the same with token overlap so CI stays fast and offline. Both implement Judge.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from verity.interfaces import ChatModel
from verity.models import ScoredChunk
from verity.store.bm25 import tokenize

_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_DONT_KNOW = re.compile(
    r"\b(don'?t know|cannot find|no (?:relevant )?information|not in (?:the |my )?notes)\b",
    re.I,
)
_CITATION_RE = re.compile(r"\[[^\[\]]+?::\d+\]")


def _strip_citations(text: str) -> str:
    # citation markers are references, not claims, so drop them before scoring
    return _CITATION_RE.sub("", text)


class Judge(Protocol):
    def faithfulness(self, answer: str, contexts: list[ScoredChunk]) -> float: ...

    def answer_relevancy(self, question: str, answer: str) -> float: ...


class HeuristicJudge:
    def faithfulness(self, answer: str, contexts: list[ScoredChunk]) -> float:
        answer = _strip_citations(answer)
        # a correct "I don't know" with no context is faithful
        if _DONT_KNOW.search(answer):
            return 1.0 if not contexts else 0.8
        context_terms: set[str] = set()
        for c in contexts:
            context_terms |= set(tokenize(c.chunk.text))
        if not context_terms:
            return 0.0
        sentences = [s for s in _SENT_RE.split(answer.strip()) if s.strip()]
        if not sentences:
            return 0.0
        supported = 0
        for sent in sentences:
            terms = set(tokenize(sent))
            if not terms:
                supported += 1
                continue
            coverage = len(terms & context_terms) / len(terms)
            if coverage >= 0.6:
                supported += 1
        return supported / len(sentences)

    def answer_relevancy(self, question: str, answer: str) -> float:
        q = set(tokenize(question))
        a = set(tokenize(_strip_citations(answer)))
        if not q or not a:
            return 0.0
        return len(q & a) / len(q)


class LlmJudge:
    _RUBRIC = (
        "You are a strict evaluator. Given a QUESTION, an ANSWER, and CONTEXT, "
        "score two properties from 0.0 to 1.0:\n"
        "- faithfulness: is every claim in the ANSWER supported by CONTEXT?\n"
        "- answer_relevancy: does the ANSWER address the QUESTION?\n"
        'Respond with ONLY JSON: {"faithfulness": <float>, "answer_relevancy": <float>}.'
    )

    def __init__(self, model: ChatModel) -> None:
        self._model = model

    def _score(self, question: str, answer: str, contexts: list[ScoredChunk]) -> dict[str, float]:
        ctx = "\n\n".join(c.chunk.text for c in contexts) or "(no context)"
        prompt = f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nCONTEXT:\n{ctx}"
        msg = self._model.chat(
            [
                {"role": "system", "content": self._RUBRIC},
                {"role": "user", "content": prompt},
            ]
        )
        content = str(msg.get("content", "{}"))
        match = re.search(r"\{.*\}", content, re.S)
        try:
            data = json.loads(match.group(0)) if match else {}
        except json.JSONDecodeError:
            data = {}
        return {
            "faithfulness": float(data.get("faithfulness", 0.0)),
            "answer_relevancy": float(data.get("answer_relevancy", 0.0)),
        }

    def faithfulness(self, answer: str, contexts: list[ScoredChunk]) -> float:
        return self._score("", answer, contexts)["faithfulness"]

    def answer_relevancy(self, question: str, answer: str) -> float:
        return self._score(question, answer, [])["answer_relevancy"]
