from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from verity.api import create_app
from verity.corpus import load_directory
from verity.llm.fake import ScriptedChatModel
from verity.pipeline import build_pipeline


def test_load_directory_reads_markdown(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.md").write_text("alpha content", encoding="utf-8")
    (tmp_path / "sub" / "b.txt").write_text("beta content", encoding="utf-8")
    (tmp_path / "empty.md").write_text("   ", encoding="utf-8")
    (tmp_path / "ignore.json").write_text("{}", encoding="utf-8")

    docs = load_directory(tmp_path)
    ids = {d.id for d in docs}
    assert ids == {"a.md", "sub/b.txt"}  # posix relative paths, empties skipped


def test_load_directory_rejects_non_directory(tmp_path: Path):
    f = tmp_path / "x.md"
    f.write_text("hi", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        load_directory(f)


def test_api_health_ingest_and_ask(tmp_path: Path):
    (tmp_path / "mughals.md").write_text(
        "The Mughal Empire was founded in 1526 by Babur.", encoding="utf-8"
    )
    model = ScriptedChatModel(
        [{"role": "assistant", "content": "Founded in 1526 by Babur [mughals.md::0]."}]
    )
    pipeline = build_pipeline(use_models=False, chat_model=model)
    client = TestClient(create_app(pipeline))

    with client:
        assert client.get("/health").json()["status"] == "ok"

        ingest = client.post("/ingest", json={"directory": str(tmp_path)})
        assert ingest.status_code == 200
        assert ingest.json() == {"documents": 1, "chunks": 1}

        ask = client.post("/ask", json={"question": "who founded the Mughals?"})
        assert ask.status_code == 200
        assert "Babur" in ask.json()["text"]
