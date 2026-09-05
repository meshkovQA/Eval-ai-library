# Eval AI Library

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI](https://img.shields.io/pypi/v/eval-ai-library)](https://pypi.org/project/eval-ai-library/)

> Based on [firstlinesoftware/eval-ai-library](https://github.com/firstlinesoftware/eval-ai-library). This is an independently maintained version with additional features and PyPI distribution.

Comprehensive AI model evaluation framework for RAG systems and AI agents. Supports 35+ evaluation metrics, 30+ LLM providers (OpenAI, Anthropic, Google, Azure, AWS Bedrock, Vertex AI, Mistral, Groq, xAI, DeepSeek, Cohere, OpenRouter, Together, Fireworks, Ollama, MLX, and more via LiteLLM), built-in test data generation from documents, an interactive web dashboard for visualization and analysis, and a first-class tracing subsystem with framework integrations for LangChain, LlamaIndex, CrewAI, AutoGen, Haystack, Semantic Kernel, Claude Agent SDK, smolagents, phidata, and OpenAI Assistants (plus an OpenTelemetry exporter). Implements advanced techniques including G-Eval probability-weighted scoring and Temperature-Controlled Verdict Aggregation via Generalized Power Mean.

## Installation

```bash
pip install eval-ai-library
```

Full version with document parsing and OCR support:

```bash
pip install eval-ai-library[full]
```

Lite version (core evaluation only):

```bash
pip install eval-ai-library[lite]
```

## Quick Start

```python
from eval_lib import EvalAI

evaluator = EvalAI(model="gpt-4o")

result = evaluator.evaluate(
    input="What is Python?",
    actual_output="Python is a programming language.",
    expected_output="Python is a high-level programming language.",
    metrics=["answer_relevancy", "faithfulness"]
)

print(result.score)
```

## Tracing

Capture agent runs (LLM calls, tool calls, steps) in your application and ship them to a collector for runtime evaluation. Three environment variables are enough to start:

```bash
TRACING_ENABLED=true
TRACING_URL=https://your-collector/api/traces/ingest
TRACING_PROJECT=my-agent
```

```python
from eval_lib.tracing import tracer, trace_llm, trace_tool, SpanType

@trace_llm(name="answer")
async def answer(prompt: str) -> str:
    ...

trace_id = tracer.start_trace("chat")
result = await answer("hello")
tracer.add_trace_usage(input_tokens=120, output_tokens=30, cost_usd=0.0004)  # per LLM call — accumulates
tracer.set_trace_metadata(model="gpt-4o", input="hello", output=result,      # facts — overwrites
                          session_id="sess-1", customer="acme")
tracer.end_trace()
await tracer.aclose()   # on shutdown: awaits in-flight deliveries, closes the HTTP pool
```

**Two ways to attach data — don't mix them up.** `add_trace_usage(...)` *adds* to running totals and is the right call once per LLM call. `set_trace_metadata(...)` *declares* a fact and overwrites; calling it per LLM call with that call's tokens keeps only the last call. Declared totals, when given, take precedence over accumulated ones.

**Payload.** Every trace is sent as `{"project": ..., "trace": {...}}` where the trace carries, alongside the span tree (`spans`, nested `children`):

| Field | What it is |
|---|---|
| `usage` | Accumulated counters: `input_tokens`, `output_tokens`, `total_tokens`, `cached_tokens`, `reasoning_tokens`, `cost_usd`, `llm_calls`, and `source` (`accumulated` / `spans` / `declared`) |
| `metadata` | Everything passed to `set_trace_metadata`, verbatim, as one object — map it straight onto your own metadata column |
| `started_at` / `ended_at` | ISO-8601 UTC (`start_time` / `end_time` epoch floats are kept for compatibility) |
| `model`, `input`, `output`, `session_id`, `user_id`, `cost_usd`, `input_tokens`, … | Promoted to top level for older consumers |

**Configuration.**

| Variable | Default | Purpose |
|---|---|---|
| `TRACING_ENABLED` | `false` | Master switch |
| `TRACING_URL` | — | Collector endpoint (POST) |
| `TRACING_PROJECT` | `default` | Project name sent with every payload |
| `TRACING_API_KEY` | — | Sent as `Authorization: Bearer …` |
| `TRACING_SINK` | `http` | `http` / `file` (JSONL at `TRACING_SINK_PATH`, default `traces.jsonl`) / `memory` |
| `TRACING_STREAM` | `false` | Ship each span as soon as it ends (`partial_span`), then the full trace at `end_trace()` — a crash mid-run still leaves spans on record |
| `TRACING_STRICT` | `false` | Raise on delivery failure instead of logging a warning |
| `TRACING_MAX_RETRIES` / `TRACING_RETRY_BACKOFF` | `2` / `0.5` | Retries for transient failures (5xx, 408, 429, timeouts); 4xx fails fast |
| `TRACING_MAX_FIELD_LENGTH` | unlimited | Optional cap on captured span input/output; truncation is marked, never silent |
| `TRACING_REDACT` | `true` | Redact credential-looking keys and token-shaped values in captured data |

Delivery is never silent: failures are logged under `eval_lib.tracing`, and `tracer.stats` exposes `sent` / `failed` / `retried` / `dropped` counters for health checks. Buffered traces are flushed at interpreter exit; in an async service call `await tracer.aflush()` (or `aclose()`) before shutdown so scheduled sends are not abandoned.

**Framework integrations** (`eval_lib.tracing`, each lazy-loaded so a slim install still imports): LangChain / LangGraph (`EvalLibCallbackHandler`), LlamaIndex (`install_llamaindex_tracing`, workflow and legacy APIs), CrewAI, AutoGen, Claude Agent SDK, OpenAI — Responses API and Chat Completions via `trace_openai_client(OpenAI())` (function-call tool loops are paired across requests) plus the legacy Assistants API — Haystack, Semantic Kernel, smolagents, phidata / agno, and an OpenTelemetry `SpanExporter` (`EvalLibSpanExporter`). Manual tracing works from any code via the decorators or `with tracer.trace("step"):`. Note that a plain `threading.Thread` / thread-pool worker does not inherit the trace context — submit `tracer.wrap(fn)` instead of `fn` to carry it over, and end such a trace with `tracer.end_trace(trace_id=...)` if the worker finishes it.

**Receiver.** The bundled Flask receiver (`eval_lib.connector.trace_routes`) accepts both full traces and streamed `partial_span` payloads, stores `metadata` / `usage` unchanged, and flattens nested spans for the reliability metrics. Set `TRACE_RECEIVER_ADMIN_KEY` to protect project create/list when you are not supplying your own `auth_verifier`; a verifier returning `{"projects": [...]}` scopes the caller to those projects.

## Documentation

Full documentation is available at [library.eval-ai.com](https://library.eval-ai.com).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:
```bibtex
@software{eval_ai_library,
  author = {Meshkov, Aleksandr},
  title = {Eval AI Library: Comprehensive AI Model Evaluation Framework},
  year = {2025},
  url = {https://github.com/meshkovQA/Eval-ai-library.git}
}
```

### References

This library implements techniques from:
```bibtex
@inproceedings{liu2023geval,
  title={G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment},
  author={Liu, Yang and Iter, Dan and Xu, Yichong and Wang, Shuohang and Xu, Ruochen and Zhu, Chenguang},
  booktitle={Proceedings of EMNLP},
  year={2023}
}
```

## Support

- Issues: [GitHub Issues](https://github.com/meshkovQA/Eval-ai-library/issues)
- Documentation: [library.eval-ai.com](https://library.eval-ai.com)
