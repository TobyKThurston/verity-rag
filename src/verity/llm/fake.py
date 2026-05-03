"""Chat model that replays a fixed list of turns. For tests."""

from __future__ import annotations


class ScriptedChatModel:
    def __init__(self, turns: list[dict[str, object]]) -> None:
        self._turns = list(turns)
        self._i = 0
        self.calls: list[list[dict[str, object]]] = []

    def chat(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        self.calls.append(messages)
        turn = self._turns[min(self._i, len(self._turns) - 1)]
        self._i += 1
        return turn
