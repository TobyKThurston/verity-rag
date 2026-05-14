"""FastAPI app over the pipeline: /health, /ingest, /ask."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from verity.config import get_settings
from verity.corpus import load_directory
from verity.models import Answer
from verity.pipeline import Pipeline, build_pipeline


class IngestRequest(BaseModel):
    directory: str


class IngestResponse(BaseModel):
    documents: int
    chunks: int


class AskRequest(BaseModel):
    question: str


def create_app(pipeline: Pipeline | None = None, *, use_models: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.pipeline = pipeline or build_pipeline(use_models=use_models)
        yield

    app = FastAPI(title="Verity", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": get_settings().service_name}

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(req: IngestRequest) -> IngestResponse:
        pl: Pipeline = app.state.pipeline
        docs = load_directory(req.directory)
        chunks = pl.ingest(docs)
        return IngestResponse(documents=len(docs), chunks=len(chunks))

    @app.post("/ask", response_model=Answer)
    async def ask(req: AskRequest) -> Answer:
        pl: Pipeline = app.state.pipeline
        return pl.ask(req.question)

    return app
