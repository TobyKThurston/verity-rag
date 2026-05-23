from evals.judge import HeuristicJudge
from evals.metrics import context_precision, context_recall, hit_rate, mrr
from verity.models import Chunk, ScoredChunk


def _ctx(*doc_ids: str) -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk=Chunk(id=f"{d}::0", doc_id=d, text=d), score=1.0, source="reranked")
        for d in doc_ids
    ]


def test_recall_and_precision():
    retrieved = _ctx("a", "b", "x")
    relevant = {"a", "b"}
    assert context_recall(retrieved, relevant) == 1.0
    assert context_precision(retrieved, relevant) == 2 / 3


def test_recall_empty_relevant_is_one():
    assert context_recall(_ctx("a"), set()) == 1.0


def test_mrr_and_hit_rate():
    retrieved = _ctx("x", "y", "a")
    assert mrr(retrieved, {"a"}) == 1 / 3
    assert hit_rate(retrieved, {"a"}) == 1.0
    assert hit_rate(retrieved, {"z"}) == 0.0


def test_heuristic_faithfulness_rewards_grounded_answer():
    judge = HeuristicJudge()
    ctx = _ctx_text("The access token is valid for fifteen minutes.")
    grounded = "The access token is valid for fifteen minutes [a::0]."
    hallucinated = "The access token lasts seven days and never expires anywhere."
    assert judge.faithfulness(grounded, ctx) == 1.0
    assert judge.faithfulness(hallucinated, ctx) < 0.5


def test_heuristic_abstention_is_faithful_when_no_context():
    judge = HeuristicJudge()
    assert judge.faithfulness("I don't know based on these notes.", []) == 1.0


def test_answer_relevancy():
    judge = HeuristicJudge()
    score = judge.answer_relevancy(
        "how long is the token valid", "the token is valid fifteen minutes"
    )
    assert score > 0.4


def _ctx_text(text: str) -> list[ScoredChunk]:
    chunk = Chunk(id="a::0", doc_id="a", text=text)
    return [ScoredChunk(chunk=chunk, score=1.0, source="reranked")]
