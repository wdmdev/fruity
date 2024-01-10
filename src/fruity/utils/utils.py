"""Utility functions for Fruity."""

from typing import Mapping, Any, Optional


def get_metric_value(metric_dict: Mapping[str, Any], metric_name: str) -> Optional[Any]:
    """Safely retrieves value of the metric logged in LightningModule.

    Args:
    ----
        metric_dict (Mapping[str, Any]): Dict with metrics logged in LightningModule.
        metric_name (str): Name of the metric to retrieve.

    Returns:
    -------
        float: Value of the metric.
    """
    if not metric_name:
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()

    return metric_value
