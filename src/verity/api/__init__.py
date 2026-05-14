"""FastAPI application exposing the pipeline over HTTP."""

from verity.api.app import create_app

__all__ = ["create_app"]
