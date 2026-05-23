from verity.llm.fake import ScriptedChatModel
from verity.models import Document
from verity.pipeline import Pipeline


def test_agent_runs_tool_then_answers_with_citation(pipeline: Pipeline, docs: list[Document]):
    pipeline.ingest(docs)
    answer = pipeline.ask("Who founded the Mughal Empire?")
    assert "Babur" in answer.text
    # The scripted model emitted [04-mughal-empire.md::0]; it should resolve to a citation.
    assert any(c.chunk_id == "04-mughal-empire.md::0" for c in answer.citations)
    assert answer.contexts  # retrieved context is attached for transparency


def test_pipeline_retrieve_finds_relevant_doc(pipeline: Pipeline, docs: list[Document]):
    pipeline.ingest(docs)
    results = pipeline.retrieve("Harappa Mohenjo-daro Indus Valley cities")
    assert results[0].chunk.doc_id == "02-indus-valley-civilization.md"
    # fused + reranked source labelling survives the pipeline
    assert results[0].source == "reranked"


def test_agent_respects_step_budget(pipeline: Pipeline, docs: list[Document]):
    # A model that always asks to search would loop forever without the cap.
    looping = ScriptedChatModel(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "search_knowledge_base", "arguments": {"query": "x"}}}
                ],
            }
        ]
    )
    pipeline.chat_model = looping
    pipeline.__post_init__()  # rebuild agent with the new model
    pipeline.ingest(docs)
    # Should terminate (force a final answer) rather than hang.
    answer = pipeline.ask("anything")
    assert answer.query == "anything"
