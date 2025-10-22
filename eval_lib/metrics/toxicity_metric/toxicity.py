# toxicity.py

from eval_lib.metric_pattern import MetricPattern
from .toxicity_template import ToxicityTemplate


class ToxicityMetric(MetricPattern):
    name = "toxicityMetric"
    template_cls = ToxicityTemplate
