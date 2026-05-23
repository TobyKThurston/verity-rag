from verity.llm.extractive import ExtractiveChatModel

SEARCH_QUERY = {"role": "user", "content": "how does autoscaling work"}
CTX = {
    "role": "tool",
    "content": (
        "[k8s.md::0] A Horizontal Pod Autoscaler targets seventy percent CPU.\n\n"
        "[k8s.md::1] Rolling updates use a maxSurge of one."
    ),
}


def test_first_turn_requests_a_search():
    model = ExtractiveChatModel()
    msg = model.chat([SEARCH_QUERY], tools=[{"type": "function"}])
    assert msg["tool_calls"][0]["function"]["name"] == "search_knowledge_base"
    assert msg["tool_calls"][0]["function"]["arguments"]["query"] == "how does autoscaling work"


def test_second_turn_answers_with_citation():
    model = ExtractiveChatModel()
    msg = model.chat([{"role": "user", "content": "what does the autoscaler target"}, CTX])
    assert "Horizontal Pod Autoscaler" in str(msg["content"])
    assert "[k8s.md::0]" in str(msg["content"])


def test_abstains_when_no_content_overlap():
    model = ExtractiveChatModel()
    msg = model.chat([{"role": "user", "content": "capital of France"}, CTX])
    assert "don't know" in str(msg["content"])


def test_abstains_on_empty_context():
    model = ExtractiveChatModel()
    empty = {"role": "tool", "content": "No matching notes found."}
    msg = model.chat([{"role": "user", "content": "anything"}, empty])
    assert "don't know" in str(msg["content"])
