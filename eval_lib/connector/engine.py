import asyncio
import re
import json
import time
import threading
import uuid
from typing import List, Dict, Any, Optional

import aiohttp

from eval_lib.connector.models import (
    EvalJobConfig, JobProgress, JobStatus,
    ApiConnectionConfig, ResponseMapping, DatasetColumnMapping,
)
from eval_lib.connector.metric_registry import instantiate_metric
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.evaluate import evaluate
from eval_lib.dashboard_server import save_results_to_cache


def extract_path(obj: Any, path: str) -> Any:
    """Extract value from nested object using JSONPath-like notation.

    Supports: dot notation, [index], [-1], [*] wildcard, [?(@.field==value)] filters, $. prefix.

    Examples:
        extract_path(data, "choices[0].message.content")
        extract_path(data, "$.sources[*].content")
        extract_path(data, "items[-1].text")
    """
    if not path:
        return obj
    if path.startswith('$.'):
        path = path[2:]
    elif path == '$':
        return obj
    if not path:
        return obj
    return _extract_recursive(obj, path)


def _extract_recursive(data: Any, path_str: str) -> Any:
    if data is None or not path_str:
        return data

    segments = re.findall(r'[^.\[\]]+|\[[^\]]*\]', path_str)
    result = data

    i = 0
    while i < len(segments):
        seg = segments[i]
        if result is None:
            return None

        if seg.startswith('[') and seg.endswith(']'):
            inner = seg[1:-1]

            # Wildcard [*]
            if inner == '*':
                if not isinstance(result, list):
                    return None
                rest = segments[i + 1:]
                if rest:
                    rest_path = '.'.join(
                        s if not s.startswith('[') else s
                        for s in rest
                    ).replace('.[', '[')
                    return [v for v in (_extract_recursive(item, rest_path) for item in result) if v is not None]
                return result

            # Filter [?(@.field==value)]
            filter_match = re.match(r'\?\(@\.([^=!<>]+)\s*([=!<>]+)\s*[\'"]([^\'"]*)[\'\"]\)', inner)
            if filter_match:
                if not isinstance(result, list):
                    return None
                field_path, op, val = filter_match.group(1), filter_match.group(2), filter_match.group(3)
                filtered = []
                for item in result:
                    if not isinstance(item, dict):
                        continue
                    fv = item
                    for p in field_path.split('.'):
                        fv = fv.get(p) if isinstance(fv, dict) else None
                    if op in ('==', '=') and str(fv) == val:
                        filtered.append(item)
                    elif op == '!=' and str(fv) != val:
                        filtered.append(item)
                rest = segments[i + 1:]
                if rest:
                    rest_path = '.'.join(
                        s if not s.startswith('[') else s
                        for s in rest
                    ).replace('.[', '[')
                    return [v for v in (_extract_recursive(item, rest_path) for item in filtered) if v is not None]
                return filtered

            # Slice [-1:]
            if inner.endswith(':'):
                si = int(inner[:-1])
                if not isinstance(result, list):
                    return None
                ai = si if si >= 0 else len(result) + si
                result = result[ai] if 0 <= ai < len(result) else None
                i += 1
                continue

            # Numeric index (positive or negative)
            try:
                idx = int(inner)
            except ValueError:
                return None
            if not isinstance(result, list):
                return None
            ai = idx if idx >= 0 else len(result) + idx
            result = result[ai] if 0 <= ai < len(result) else None
        else:
            # Object field — also handle key[idx] like "choices[0]"
            m = re.match(r'^(\w+)\[(-?\d+)\]$', seg)
            if m:
                key, idx = m.group(1), int(m.group(2))
                if not isinstance(result, dict) or key not in result:
                    return None
                arr = result[key]
                if not isinstance(arr, list):
                    return None
                ai = idx if idx >= 0 else len(arr) + idx
                result = arr[ai] if 0 <= ai < len(arr) else None
            else:
                if not isinstance(result, dict) or seg not in result:
                    return None
                result = result[seg]

        i += 1

    return result


def _normalize_quotes(text: str) -> str:
    """Replace typographic/smart quotes with straight ASCII quotes."""
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # " "
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # ' '
    text = text.replace('\u00ab', '"').replace('\u00bb', '"')  # « »
    return text


def substitute_template(template: str, row: Dict[str, Any], variable_map: Dict[str, str]) -> str:
    """Replace {{variable}} placeholders in template with values from dataset row."""
    template = _normalize_quotes(template)

    def replacer(match):
        var_name = match.group(1)
        col_name = variable_map.get(var_name, var_name)
        value = row.get(col_name, "")
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    return re.sub(r'\{\{(\w+)\}\}', replacer, template)


class ConnectorEngine:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._jobs: Dict[str, JobProgress] = {}
                    cls._instance._running_job_id: Optional[str] = None
        return cls._instance

    def start_job(
        self,
        config: EvalJobConfig,
        dataset_rows: List[Dict[str, Any]],
        cache_dir: str = ".eval_cache",
    ) -> str:
        with self._lock:
            if self._running_job_id and self._jobs.get(self._running_job_id, JobProgress(job_id="")).status == JobStatus.RUNNING:
                raise RuntimeError("A job is already running")

            job_id = str(uuid.uuid4())[:8]
            progress = JobProgress(
                job_id=job_id,
                status=JobStatus.RUNNING,
                total_rows=len(dataset_rows),
                current_phase="api_calls",
            )
            self._jobs[job_id] = progress
            self._running_job_id = job_id

        thread = threading.Thread(
            target=self._run_in_thread,
            args=(job_id, config, dataset_rows, cache_dir),
            daemon=True,
        )
        thread.start()
        return job_id

    def get_progress(self, job_id: str) -> Optional[JobProgress]:
        return self._jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        progress = self._jobs.get(job_id)
        if progress and progress.status == JobStatus.RUNNING:
            progress.status = JobStatus.CANCELLED
            return True
        return False

    def _run_in_thread(self, job_id, config, dataset_rows, cache_dir):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                self._run_job(job_id, config, dataset_rows, cache_dir)
            )
        except Exception as e:
            progress = self._jobs[job_id]
            progress.status = JobStatus.FAILED
            progress.errors.append(str(e))
        finally:
            loop.close()
            with self._lock:
                if self._running_job_id == job_id:
                    self._running_job_id = None

    async def _run_job(self, job_id, config: EvalJobConfig, rows, cache_dir):
        progress = self._jobs[job_id]
        api = config.api_config
        mapping = config.response_mapping
        col_map = config.dataset_column_mapping

        # Phase 1: API calls
        test_cases: List[EvalTestCase] = []

        timeout = aiohttp.ClientTimeout(total=api.timeout_seconds)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            for i, row in enumerate(rows):
                if progress.status == JobStatus.CANCELLED:
                    return

                try:
                    t0 = time.time()
                    tc = await self._process_row(
                        session, api, mapping, col_map, row
                    )
                    elapsed = int((time.time() - t0) * 1000)
                    progress.response_times_ms.append(elapsed)
                    # Attach response_time to metadata
                    if not hasattr(tc, '_meta') or tc._meta is None:
                        tc._meta = {}
                    tc._meta["response_time_ms"] = elapsed
                    # Calculate cost if token_usage and cost_per_1m_tokens set
                    tokens = tc._meta.get("token_usage")
                    if tokens and config.cost_per_1m_tokens > 0:
                        tc._meta["cost"] = round(tokens * config.cost_per_1m_tokens / 1_000_000, 6)
                    test_cases.append(tc)
                except Exception as e:
                    progress.errors.append(f"Row {i+1}: {str(e)}")

                progress.completed_rows = i + 1
                if progress.response_times_ms:
                    progress.avg_response_time_ms = sum(progress.response_times_ms) // len(progress.response_times_ms)

                if api.delay_between_requests_ms > 0 and i < len(rows) - 1:
                    await asyncio.sleep(api.delay_between_requests_ms / 1000)

        if progress.status == JobStatus.CANCELLED:
            return

        if not test_cases:
            progress.status = JobStatus.FAILED
            progress.errors.append("No successful API responses to evaluate")
            return

        # Phase 2: Evaluation
        progress.current_phase = "evaluation"
        progress.completed_rows = 0
        progress.total_rows = len(test_cases)

        if not config.metrics:
            # No metrics selected — just save test cases as results
            progress.current_phase = "done"
            progress.status = JobStatus.COMPLETED
            return

        metrics = []
        for mc in config.metrics:
            try:
                m = instantiate_metric(mc.metric_class, config.eval_model, mc.params)
                metrics.append(m)
            except Exception as e:
                progress.errors.append(f"Metric {mc.metric_class}: {str(e)}")

        if not metrics:
            progress.status = JobStatus.FAILED
            progress.errors.append("No metrics could be instantiated")
            return

        try:
            session_name = config.name or f"connector_{job_id}"
            results = await evaluate(
                test_cases,
                metrics,
                verbose=False,
                show_dashboard=True,
                session_name=session_name,
            )
            progress.result_session_id = session_name

            # Inject connector metadata (response_time, tokens, cost) into cached results
            self._inject_metadata(session_name, test_cases, progress, cache_dir)
        except Exception as e:
            progress.status = JobStatus.FAILED
            progress.errors.append(f"Evaluation error: {str(e)}")
            return

        progress.current_phase = "done"
        progress.status = JobStatus.COMPLETED
        progress.completed_rows = len(test_cases)

    def _inject_metadata(self, session_name, test_cases, progress, cache_dir):
        """Inject connector-specific metadata into cached dashboard results."""
        try:
            from eval_lib.dashboard_server import DashboardCache
            cache = DashboardCache(cache_dir)
            session = cache.get_by_session(session_name)
            if not session or 'data' not in session:
                return

            data = session['data']
            tc_list = data.get('test_cases', [])

            total_tokens = 0
            total_cost = 0.0
            perf_data = []

            for i, tc_data in enumerate(tc_list):
                meta = getattr(test_cases[i], '_meta', {}) if i < len(test_cases) else {}
                rt = meta.get("response_time_ms")
                tokens = meta.get("token_usage")
                cost = meta.get("cost")
                sys_prompt = meta.get("system_prompt")

                tc_data["response_time_ms"] = rt
                tc_data["token_usage"] = tokens
                tc_data["api_cost"] = cost
                tc_data["system_prompt"] = sys_prompt

                if tokens:
                    total_tokens += tokens
                if cost:
                    total_cost += cost

                perf_data.append({
                    "response_time_ms": rt,
                    "token_usage": tokens,
                    "api_cost": cost,
                })

            # Add performance summary
            rts = [p["response_time_ms"] for p in perf_data if p["response_time_ms"] is not None]
            data["performance"] = {
                "response_times_ms": rts,
                "avg_response_time_ms": sum(rts) // len(rts) if rts else 0,
                "min_response_time_ms": min(rts) if rts else 0,
                "max_response_time_ms": max(rts) if rts else 0,
                "total_tokens": total_tokens,
                "total_api_cost": round(total_cost, 6),
                "per_row": perf_data,
            }

            cache._save_cache()
        except Exception:
            pass

    async def _process_row(self, session, api, mapping, col_map, row):
        """Send HTTP request for a single dataset row and build EvalTestCase."""
        body_str = substitute_template(
            api.body_template, row, col_map.template_variable_map
        )

        url = substitute_template(api.base_url, row, col_map.template_variable_map)

        headers = {}
        for h in api.headers:
            if h.enabled:
                headers[h.key] = substitute_template(h.value, row, col_map.template_variable_map)

        params = {}
        for k, v in api.query_params.items():
            params[k] = substitute_template(v, row, col_map.template_variable_map)

        kwargs = {"params": params, "headers": headers}
        if api.method.value in ("POST", "PUT") and body_str.strip():
            kwargs["data"] = body_str.encode("utf-8")
            kwargs["headers"]["Content-Type"] = "application/json"

        for attempt in range(max(1, api.max_retries)):
            try:
                async with session.request(api.method.value, url, **kwargs) as resp:
                    resp_text = await resp.text()
                    resp_data = json.loads(resp_text)
                    break
            except Exception as e:
                if attempt == api.max_retries - 1:
                    raise
                await asyncio.sleep(1)

        # Extract fields from response
        actual_output = str(extract_path(resp_data, mapping.actual_output_path))

        retrieval_context = None
        if mapping.retrieval_context_path:
            ctx = extract_path(resp_data, mapping.retrieval_context_path)
            if isinstance(ctx, list):
                retrieval_context = [str(c) for c in ctx]
            else:
                retrieval_context = [str(ctx)]

        # Extract token_usage from API response
        token_usage = None
        if mapping.token_usage_path:
            tu = extract_path(resp_data, mapping.token_usage_path)
            if isinstance(tu, (int, float)):
                token_usage = int(tu)

        # Extract system_prompt from API response
        system_prompt = None
        if mapping.system_prompt_path:
            sp = extract_path(resp_data, mapping.system_prompt_path)
            if sp is not None:
                system_prompt = str(sp)

        # Extract tools_called from API response
        tools_called_from_resp = None
        if mapping.tools_called_path:
            tc = extract_path(resp_data, mapping.tools_called_path)
            if isinstance(tc, list):
                tools_called_from_resp = [str(t) for t in tc]
            elif tc is not None:
                tools_called_from_resp = [str(tc)]

        # Build EvalTestCase
        input_val = str(row.get(col_map.input_column, ""))
        expected = str(row.get(col_map.expected_output_column, "")) if col_map.expected_output_column else None
        context_from_dataset = None
        if col_map.context_column:
            ctx_val = row.get(col_map.context_column, "")
            if isinstance(ctx_val, list):
                context_from_dataset = [str(c) for c in ctx_val]
            elif isinstance(ctx_val, str) and ctx_val:
                context_from_dataset = [ctx_val]

        # Merge retrieval context: from response takes priority, fallback to dataset
        final_context = retrieval_context or context_from_dataset

        # tools_called: from response takes priority, fallback to dataset
        tools_called = tools_called_from_resp
        if not tools_called and col_map.tools_called_column:
            tc_val = row.get(col_map.tools_called_column)
            if isinstance(tc_val, list):
                tools_called = tc_val
            elif isinstance(tc_val, str) and tc_val:
                tools_called = [t.strip() for t in tc_val.split(",")]

        expected_tools = None
        if col_map.expected_tools_column:
            et_val = row.get(col_map.expected_tools_column)
            if isinstance(et_val, list):
                expected_tools = et_val
            elif isinstance(et_val, str) and et_val:
                expected_tools = [t.strip() for t in et_val.split(",")]

        tc = EvalTestCase(
            input=input_val,
            actual_output=actual_output,
            expected_output=expected,
            retrieval_context=final_context,
            tools_called=tools_called,
            expected_tools=expected_tools,
        )
        # Attach extra metadata for dashboard
        tc._meta = {
            "token_usage": token_usage,
            "system_prompt": system_prompt,
        }
        return tc


async def test_api_connection(api_config: ApiConnectionConfig, sample_row: Dict[str, Any], variable_map: Dict[str, str]) -> Dict[str, Any]:
    """Send a single test request and return the response."""
    body_str = substitute_template(
        api_config.body_template, sample_row, variable_map
    )

    headers = {}
    for h in api_config.headers:
        if h.enabled:
            headers[h.key] = substitute_template(h.value, sample_row, variable_map)

    url = substitute_template(api_config.base_url, sample_row, variable_map)

    params = {}
    for k, v in api_config.query_params.items():
        params[k] = substitute_template(v, sample_row, variable_map)

    timeout = aiohttp.ClientTimeout(total=api_config.timeout_seconds)

    start = time.time()
    try:
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            kwargs = {"params": params}
            if api_config.method.value in ("POST", "PUT") and body_str.strip():
                kwargs["data"] = body_str.encode("utf-8")
                kwargs["headers"] = {"Content-Type": "application/json"}

            async with session.request(api_config.method.value, url, **kwargs) as resp:
                elapsed = int((time.time() - start) * 1000)
                resp_text = await resp.text()
                try:
                    resp_json = json.loads(resp_text)
                except json.JSONDecodeError:
                    resp_json = None

                return {
                    "status_code": resp.status,
                    "response_body": resp_json if resp_json is not None else resp_text,
                    "elapsed_ms": elapsed,
                    "sent_body": body_str,
                }
    except Exception as e:
        elapsed = int((time.time() - start) * 1000)
        return {
            "status_code": 0,
            "response_body": {"error": str(e)},
            "elapsed_ms": elapsed,
        }
