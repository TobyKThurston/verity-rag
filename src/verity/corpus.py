"""Read a directory of notes into Documents."""

from __future__ import annotations

from pathlib import Path

from verity.models import Document

_DEFAULT_GLOBS = ("*.md", "*.markdown", "*.txt")


def load_directory(root: str | Path, globs: tuple[str, ...] = _DEFAULT_GLOBS) -> list[Document]:
    # The doc id is the relative path so citations point at a real file.
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"corpus root is not a directory: {root_path}")

    docs: list[Document] = []
    for pattern in globs:
        for path in sorted(root_path.rglob(pattern)):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                continue
            rel = path.relative_to(root_path).as_posix()
            docs.append(
                Document(
                    id=rel,
                    text=text,
                    metadata={"path": str(path), "title": path.stem},
                )
            )
    return docs
