# eval_lib/tracing/usage.py
"""Token-usage extraction from heterogeneous provider payloads.

Every framework reports token counts in its own shape. This module
normalises the ones seen in practice into a single record:

* LangChain — ``llm_output.token_usage.{prompt,completion}_tokens``
* OpenAI — ``usage.{prompt,completion}_tokens`` plus the
  ``prompt_tokens_details.cached_tokens`` / ``completion_tokens_details.
  reasoning_tokens`` sub-objects
* Anthropic — ``input_tokens``/``output_tokens`` with ``cache_read_input_tokens``
* LlamaIndex — ``ChatResponse.additional_kwargs``, which
  ``_get_response_token_counts()`` fills in with the prompt/completion counts

Kept dependency-free (only :mod:`.types`) so both the sender and the
framework callbacks can import it without an import cycle.
"""

from typing import Any, Dict, Optional

from .types import TraceSpan

_INPUT_TOKEN_KEYS = ("prompt_tokens", "input_tokens")
_OUTPUT_TOKEN_KEYS = ("completion_tokens", "output_tokens")
# OpenAI / Anthropic / agno spellings of "prompt tokens served from cache".
_CACHED_TOKEN_KEYS = ("cached_tokens", "cache_read_input_tokens", "cache_read_tokens", "cache_read")
_REASONING_TOKEN_KEYS = ("reasoning_tokens", "reasoning")


def as_mapping(value: Any) -> Optional[Dict[str, Any]]:
    """Return ``value`` as a plain dict when it looks dict-like."""
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return None
    if isinstance(value, dict):
        return value
    for attr in ("model_dump", "dict"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                result = fn()
                if isinstance(result, dict):
                    return result
            except Exception:
                pass
    if hasattr(value, "__dict__"):
        try:
            return {k: v for k, v in vars(value).items() if not k.startswith("_")}
        except Exception:
            return None
    return None


def _first_int(mapping: Dict[str, Any], keys: tuple) -> int:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def usage_from_mapping(mapping: Optional[Dict[str, Any]]) -> Optional[Dict[str, int]]:
    """Pull a token-usage record out of one dict-like payload.

    Returns ``{"input_tokens", "output_tokens", "cached_tokens",
    "reasoning_tokens"}``, or ``None`` when the payload carries no counts.
    """
    if not mapping:
        return None

    # Unwrap the known nesting layers first — innermost match wins.
    for key in ("token_usage", "usage", "llm_output"):
        nested = as_mapping(mapping.get(key))
        if nested:
            found = usage_from_mapping(nested)
            if found:
                return found

    input_tokens = _first_int(mapping, _INPUT_TOKEN_KEYS)
    output_tokens = _first_int(mapping, _OUTPUT_TOKEN_KEYS)
    cached = _first_int(mapping, _CACHED_TOKEN_KEYS)
    reasoning = _first_int(mapping, _REASONING_TOKEN_KEYS)

    # OpenAI Chat: prompt_tokens_details / completion_tokens_details.
    # OpenAI Responses: input_tokens_details / output_tokens_details.
    # LangChain usage_metadata: input_token_details / output_token_details.
    for key in ("prompt_tokens_details", "input_tokens_details", "input_token_details"):
        details = as_mapping(mapping.get(key))
        if details:
            cached = cached or _first_int(details, _CACHED_TOKEN_KEYS)
    for key in ("completion_tokens_details", "output_tokens_details", "output_token_details"):
        details = as_mapping(mapping.get(key))
        if details:
            reasoning = reasoning or _first_int(details, _REASONING_TOKEN_KEYS)

    if not any((input_tokens, output_tokens, cached, reasoning)):
        return None

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached,
        "reasoning_tokens": reasoning,
    }


def usage_from_response(response: Any) -> Optional[Dict[str, int]]:
    """Token usage for an LLM response object.

    Checks ``additional_kwargs`` first: LlamaIndex populates it on every
    ``ChatResponse`` via ``_get_response_token_counts()``, so the counts are
    already there and don't need to be parsed back out of ``raw``.
    """
    if response is None:
        return None
    for attr in ("additional_kwargs", "raw", "usage"):
        found = usage_from_mapping(as_mapping(getattr(response, attr, None)))
        if found:
            return found
    return usage_from_mapping(as_mapping(response))


def span_token_usage(span: TraceSpan) -> Optional[Dict[str, int]]:
    """Best-effort token usage for one span, from its output or metadata."""
    for candidate in (span.output, span.metadata):
        found = usage_from_mapping(as_mapping(candidate))
        if found:
            return found
    return None
