"""Deterministic answerer for the offline eval suite.

Picks the most query-relevant sentences from the retrieved context and attaches
citations. Every sentence is copied from context, so the answer stays grounded.
Used instead of a real LLM so eval runs are fast and reproducible.
"""

from __future__ import annotations

import re

from verity.models import Answer, Citation, ScoredChunk
from verity.store.bm25 import tokenize

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def extractive_answer(question: str, contexts: list[ScoredChunk], max_sentences: int = 3) -> Answer:
    if not contexts:
        return Answer(query=question, text="I don't know based on these notes.")

    q_terms = set(tokenize(question))
    scored: list[tuple[float, str, ScoredChunk]] = []
    for c in contexts:
        for sent in _SENT_RE.split(c.chunk.text):
            sent = sent.strip()
            if not sent:
                continue
            terms = set(tokenize(sent))
            overlap = len(q_terms & terms) / (len(q_terms) or 1)
            scored.append((overlap, sent, c))

    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:max_sentences]
    text = " ".join(sent for _, sent, _ in top)
    seen: dict[str, ScoredChunk] = {}
    for _, _, c in top:
        seen[c.id] = c
    text += " " + " ".join(f"[{cid}]" for cid in seen)

    citations = [
        Citation(chunk_id=cid, doc_id=c.chunk.doc_id, snippet=c.chunk.text[:200])
        for cid, c in seen.items()
    ]
    return Answer(query=question, text=text.strip(), citations=citations, contexts=contexts)
