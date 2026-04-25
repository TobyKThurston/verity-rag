"""Postgres + pgvector store, HNSW index over cosine distance.

psycopg is imported lazily so the package installs without a database.
"""

from __future__ import annotations

from verity.models import Chunk, ScoredChunk


class PgVectorStore:
    def __init__(self, dsn: str, dim: int, table: str = "verity_chunks") -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self._dim = dim
        self._table = table
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id        TEXT PRIMARY KEY,
                doc_id    TEXT NOT NULL,
                text      TEXT NOT NULL,
                metadata  JSONB NOT NULL DEFAULT '{{}}',
                embedding vector({dim}) NOT NULL
            )
            """
        )
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS {table}_hnsw "
            f"ON {table} USING hnsw (embedding vector_cosine_ops)"
        )

    def add(self, chunks: list[Chunk]) -> None:
        import json

        with self._conn.cursor() as cur:
            for c in chunks:
                if c.embedding is None:
                    continue
                cur.execute(
                    f"""
                    INSERT INTO {self._table} (id, doc_id, text, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                      SET text = EXCLUDED.text, embedding = EXCLUDED.embedding
                    """,
                    (c.id, c.doc_id, c.text, json.dumps(c.metadata), c.embedding),
                )

    def search(self, query_vector: list[float], top_k: int) -> list[ScoredChunk]:
        rows = self._conn.execute(
            f"""
            SELECT id, doc_id, text, metadata, 1 - (embedding <=> %s::vector) AS score
            FROM {self._table}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_vector, query_vector, top_k),
        ).fetchall()
        return [
            ScoredChunk(
                chunk=Chunk(id=r[0], doc_id=r[1], text=r[2], metadata=r[3]),
                score=float(r[4]),
                source="dense",
            )
            for r in rows
        ]

    def all_chunks(self) -> list[Chunk]:
        rows = self._conn.execute(
            f"SELECT id, doc_id, text, metadata FROM {self._table}"
        ).fetchall()
        return [Chunk(id=r[0], doc_id=r[1], text=r[2], metadata=r[3]) for r in rows]
