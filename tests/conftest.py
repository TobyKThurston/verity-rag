import os

# Silence OpenTelemetry console export during tests.
os.environ.setdefault("VERITY_OTEL_CONSOLE_EXPORT", "false")

import pytest

from verity.config import Settings
from verity.llm.fake import ScriptedChatModel
from verity.models import Document
from verity.pipeline import build_pipeline


@pytest.fixture
def docs() -> list[Document]:
    return [
        Document(
            id="04-mughal-empire.md",
            text=(
                "The Mughal Empire was founded in 1526 when Babur won the First "
                "Battle of Panipat. The emperor Akbar abolished the jizya tax."
            ),
        ),
        Document(
            id="02-indus-valley-civilization.md",
            text=(
                "The Indus Valley Civilization had two great cities, Harappa and "
                "Mohenjo-daro, famed for grid street planning and drainage."
            ),
        ),
        Document(
            id="03-religions.md",
            text=(
                "Buddhism was founded by Siddhartha Gautama, the Buddha, who taught "
                "the Four Noble Truths and the Eightfold Path."
            ),
        ),
    ]


@pytest.fixture
def scripted_answer_model() -> ScriptedChatModel:
    """A model that searches once, then answers with a citation."""

    return ScriptedChatModel(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_knowledge_base",
                            "arguments": {"query": "mughal empire founded babur panipat"},
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "content": (
                    "The Mughal Empire was founded in 1526 when Babur won the First "
                    "Battle of Panipat [04-mughal-empire.md::0]."
                ),
            },
        ]
    )


@pytest.fixture
def pipeline(scripted_answer_model: ScriptedChatModel):
    settings = Settings(otel_console_export=False)
    return build_pipeline(settings, use_models=False, chat_model=scripted_answer_model)
