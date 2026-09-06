# eval_lib/tracing/spans.py
"""Structural rules over spans shared by the sender, ``trace_utils`` and the
agent metrics.

One tool invocation can be recorded more than once: a framework integration
records it (a LlamaIndex ``call_tool`` step, a CrewAI ``ToolUsageFinished``
event, an OpenAI ``function_call`` item) *and* a ``@trace_tool``-decorated
function or a hand-written tracer records the same execution one layer
deeper. ``top_level_tool_calls`` keeps exactly one record per invocation
using two rules that never look at arguments:

1. **Parent is a tool call** — the span is nested under another tool span,
   so it is the same invocation instrumented deeper. Dropped regardless of
   its name (the inner layer often has a different label).
2. **Same name, contained in time** — the span's ``[start, end]`` lies inside
   another tool span's ``[start, end]`` of the same name. Integrations that
   record calls after the fact cannot link a parent, but they backdate the
   span to the real execution window, so containment identifies the pair.

Genuine repeats — the agent calling the same tool again with the same
arguments — run one after another, never inside each other, so they are
always kept. That is what ``repeated_failure`` detection depends on.

Stdlib only, so it can be imported from anywhere without an import cycle.
"""

from typing import Any, Callable, List, Optional, Sequence, TypeVar

T = TypeVar("T")

# Identical intervals are ambiguous: two zero-length spans recorded at the
# same instant may be different calls, while two identical *positive*
# intervals are almost certainly one call copied by two layers.
_EPS = 1e-9


def _interval(start: Optional[float], end: Optional[float]):
    if start is None or end is None:
        return None
    try:
        s, e = float(start), float(end)
    except (TypeError, ValueError):
        return None
    if e < s:
        return None
    return s, e


def _contained(inner, outer) -> bool:
    """``inner`` lies within ``outer`` and is not the very same interval,
    unless both are strictly positive in length (a copied window)."""
    (s1, e1), (s2, e2) = inner, outer
    if not (s2 - _EPS <= s1 and e1 <= e2 + _EPS):
        return False
    identical = abs(s1 - s2) <= _EPS and abs(e1 - e2) <= _EPS
    if identical:
        return (e1 - s1) > _EPS
    return True


def top_level_tool_calls(
    items: Sequence[T],
    *,
    get_id: Callable[[T], Any],
    get_parent: Callable[[T], Any],
    get_name: Callable[[T], Any],
    get_start: Callable[[T], Optional[float]] = lambda _: None,
    get_end: Callable[[T], Optional[float]] = lambda _: None,
) -> List[T]:
    """The subset of ``items`` (all tool calls) that are distinct invocations.

    Order is preserved. When two same-named items carry identical positive
    intervals the first one is kept.
    """
    items = list(items)
    if not items:
        return []

    ids = {get_id(it) for it in items if get_id(it) is not None}

    kept: List[T] = []
    for idx, item in enumerate(items):
        # Rule 1 — nested under another tool call.
        parent = get_parent(item)
        if parent is not None and parent in ids and parent != get_id(item):
            continue

        # Rule 2 — same name, contained in another tool call's window.
        window = _interval(get_start(item), get_end(item))
        if window is not None:
            name = get_name(item)
            folded = False
            for j, other in enumerate(items):
                if other is item or get_name(other) != name:
                    continue
                other_window = _interval(get_start(other), get_end(other))
                if other_window is None:
                    continue
                if _contained(window, other_window):
                    identical = (
                        abs(window[0] - other_window[0]) <= _EPS
                        and abs(window[1] - other_window[1]) <= _EPS
                    )
                    # For identical windows keep the earlier item only.
                    if identical and j > idx:
                        continue
                    folded = True
                    break
            if folded:
                continue

        kept.append(item)
    return kept
