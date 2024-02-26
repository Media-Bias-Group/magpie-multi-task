"""Enum for training-splits"""

from enum import Enum


class Split(Enum):
    """Training splits."""

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"
    EVAL = "eval"
