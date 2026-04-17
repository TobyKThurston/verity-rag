"""OpenTelemetry tracing. One trace per query, one span per pipeline stage."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Span, Tracer

from verity.config import get_settings

_configured = False


def configure_tracing() -> Tracer:
    # Swap ConsoleSpanExporter for an OTLP exporter to ship to a collector.
    global _configured
    settings = get_settings()
    if not _configured:
        provider = TracerProvider(
            resource=Resource.create({"service.name": settings.service_name})
        )
        if settings.otel_console_export:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        _configured = True
    return trace.get_tracer(settings.service_name)


@contextmanager
def span(name: str, **attributes: object) -> Iterator[Span]:
    tracer = configure_tracing()
    with tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            current.set_attribute(key, value)  # type: ignore[arg-type]
        yield current
