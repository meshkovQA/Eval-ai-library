# eval_lib/dashboard_server.py

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class DashboardCache:
    """Cache to store evaluation results for the dashboard"""

    def __init__(self, cache_dir: str = ".eval_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "results.json"
        self.results_history = []
        self._load_cache()

    def _load_cache(self):
        """Load cache from file"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.results_history = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")
                self.results_history = []

    def _save_cache(self):
        """Save cache to file"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.results_history, f,
                          indent=2, ensure_ascii=False, sort_keys=False)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    def add_results(self, results: List[tuple], session_name: Optional[str] = None) -> str:
        """Add new results to the cache"""
        import time
        session_id = session_name or f"session_{int(time.time())}"
        parsed_data = self._parse_results(results)

        session_data = {
            'session_id': session_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data': parsed_data
        }

        self.results_history.append(session_data)
        self._save_cache()

        return session_id

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Get latest results"""
        if self.results_history:
            return self.results_history[-1]
        return None

    def get_all(self) -> List[Dict[str, Any]]:
        """Get all results"""
        return self.results_history

    def get_by_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get results by session_id"""
        for session in self.results_history:
            if session['session_id'] == session_id:
                return session
        return None

    def clear(self):
        """Clear the cache"""
        self.results_history = []
        self._save_cache()

    def _parse_results(self, results: List[tuple]) -> Dict[str, Any]:
        """Parse raw results into structured format for dashboard"""

        test_cases = []
        metrics_summary = {}
        total_cost = 0.0

        for test_idx, test_results in results:
            for result in test_results:
                test_case_data = {
                    'test_index': test_idx,
                    'input': result.input[:100] + '...' if len(result.input) > 100 else result.input,
                    'input_full': result.input,
                    'actual_output': result.actual_output[:200] if result.actual_output else '',
                    'actual_output_full': result.actual_output,
                    'expected_output': result.expected_output[:200] if result.expected_output else '',
                    'expected_output_full': result.expected_output,
                    'retrieval_context': result.retrieval_context if result.retrieval_context else [],
                    'metrics': []
                }

                for metric_data in result.metrics_data:
                    # Determine model name
                    if isinstance(metric_data.evaluation_model, str):
                        model_name = metric_data.evaluation_model
                    else:
                        # For CustomLLMClient
                        try:
                            model_name = metric_data.evaluation_model.get_model_name()
                        except:
                            model_name = str(
                                type(metric_data.evaluation_model).__name__)

                    test_case_data['metrics'].append({
                        'name': metric_data.name,
                        'score': round(metric_data.score, 3),
                        'success': metric_data.success,
                        'threshold': metric_data.threshold,
                        'reason': metric_data.reason[:300] if metric_data.reason else '',
                        'reason_full': metric_data.reason,
                        'evaluation_model': model_name,
                        'evaluation_cost': metric_data.evaluation_cost,
                        'evaluation_log': metric_data.evaluation_log
                    })

                    if metric_data.name not in metrics_summary:
                        metrics_summary[metric_data.name] = {
                            'scores': [],
                            'passed': 0,
                            'failed': 0,
                            'threshold': metric_data.threshold,
                            'total_cost': 0.0,
                            'model': model_name
                        }

                    metrics_summary[metric_data.name]['scores'].append(
                        metric_data.score)
                    if metric_data.success:
                        metrics_summary[metric_data.name]['passed'] += 1
                    else:
                        metrics_summary[metric_data.name]['failed'] += 1

                    if metric_data.evaluation_cost:
                        total_cost += metric_data.evaluation_cost
                        metrics_summary[metric_data.name]['total_cost'] += metric_data.evaluation_cost

                test_cases.append(test_case_data)

        for metric_name, data in metrics_summary.items():
            data['avg_score'] = sum(data['scores']) / \
                len(data['scores']) if data['scores'] else 0
            data['success_rate'] = (data['passed'] / (data['passed'] + data['failed'])
                                    * 100) if (data['passed'] + data['failed']) > 0 else 0

        return {
            'test_cases': test_cases,
            'metrics_summary': metrics_summary,
            'total_cost': total_cost,
            'total_tests': len(test_cases)
        }


def save_results_to_cache(results: List[tuple], session_name: Optional[str] = None) -> str:
    """
    Save evaluation results to cache for dashboard viewing.
    Cache is always saved to .eval_cache/ in current directory.

    Args:
        results: Evaluation results from evaluate()
        session_name: Optional name for the session

    Returns:
        Session ID
    """
    cache = DashboardCache()
    return cache.add_results(results, session_name)
