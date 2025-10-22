# eval_lib/metrics/bias_metric/bias.py

from eval_lib.metric_pattern import MetricPattern
from .bias_template import BiasTemplate


class BiasMetric(MetricPattern):
    name = "biasMetric"
    template_cls = BiasTemplate
