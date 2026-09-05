"""Tracing configuration read from environment variables.

The base install is designed to Just Work with three env vars:
``TRACING_ENABLED``, ``TRACING_URL`` and ``TRACING_PROJECT``. Everything
else is opt-in tuning knob for integrations."""

import os
from typing import Optional


class TracingConfig:
    """Tracing configuration from environment variables."""

    @staticmethod
    def is_enabled() -> bool:
        return os.getenv("TRACING_ENABLED", "false").lower() == "true"

    @staticmethod
    def get_url() -> str:
        return os.getenv("TRACING_URL", "")

    @staticmethod
    def get_project() -> str:
        return os.getenv("TRACING_PROJECT", "default")

    @staticmethod
    def get_api_key() -> Optional[str]:
        return os.getenv("TRACING_API_KEY")

    @staticmethod
    def get_sink_kind() -> str:
        """One of ``http`` (default), ``memory``, ``file``.

        Used by :class:`~eval_lib.tracing.sender.TraceSender` to pick the
        default sink when no explicit sink instance is passed. Ignored
        when a caller passes ``sink=...`` at construction time.
        """
        return os.getenv("TRACING_SINK", "http").lower()

    @staticmethod
    def get_sink_path() -> str:
        """Path used by ``FileSink`` (JSONL append). Default: ``traces.jsonl``."""
        return os.getenv("TRACING_SINK_PATH", "traces.jsonl")

    @staticmethod
    def is_strict() -> bool:
        """When true, send failures raise instead of being logged.

        Useful in CI / tests where a silent tracing outage would hide a
        misconfigured receiver.
        """
        return os.getenv("TRACING_STRICT", "false").lower() == "true"

    @staticmethod
    def is_stream() -> bool:
        """When true, every :meth:`AgentTracer.end_span` immediately
        flushes that span to the receiver as a ``partial_span`` — so a
        long-running session that crashes still has its spans on record.
        """
        return os.getenv("TRACING_STREAM", "false").lower() == "true"

    @staticmethod
    def get_max_field_length() -> Optional[int]:
        """Max characters kept per captured trace field (span input/output).

        Framework callbacks used to hard-truncate span input/output to a
        few hundred–thousand characters, silently dropping data before it
        ever reached the collector. The SDK now preserves fields in full by
        default; set ``TRACING_MAX_FIELD_LENGTH`` to a positive integer to
        cap oversized fields (e.g. to bound payload size). Unset, empty,
        ``0`` or a non-integer all mean *no limit*. When a cap applies the
        truncation is marked explicitly, never silent.
        """
        raw = os.getenv("TRACING_MAX_FIELD_LENGTH")
        if raw is None or not raw.strip():
            return None
        try:
            value = int(raw)
        except ValueError:
            return None
        return value if value > 0 else None
