"""Local chat-model client and a deterministic fake for tests."""

from verity.llm.extractive import ExtractiveChatModel
from verity.llm.fake import ScriptedChatModel
from verity.llm.ollama import OllamaChatModel

__all__ = ["ExtractiveChatModel", "OllamaChatModel", "ScriptedChatModel"]
