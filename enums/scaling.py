"""Enum for loss-scaling-strategy"""

from enum import Enum


class LossScaling(Enum):
    """Training splits."""

    UNIFORM = "uniform"
    STATIC = "static"
