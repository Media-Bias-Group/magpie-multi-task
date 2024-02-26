"""Enum for Aggregation method."""

from enum import Enum


class AggregationMethod(Enum):
    """Aggregation method."""

    MEAN = "mean"
    PCGRAD = "pcgrad"
    PCGRAD_ONLINE = "pcgrad_online"
