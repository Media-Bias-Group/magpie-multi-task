"""This module contains utils."""

import functools
import random

import numpy as np
import torch

from config import RANDOM_SEED


def integer_formatter(i):
    """Format integers.

    Example:
        1827028 -> 1,827,028
    """
    if isinstance(i, str):
        return i
    return f"{i:,d}"


def float_formatter(i):
    """Format floats.

    Example:
        0.8982232 -> 0.898
    """
    if isinstance(i, str):
        return i
    return f"{i:0.3f}"


def get_class_weights(y, method="ins"):
    """Compute the weights for vector of counts of each label.

    ins = inverse number of samples
    isns = inverse squared number of samples
    esns = effective sampling number of samples
    """
    counts = y.unique(return_counts=True)[1]

    if method == "ins":
        weights = 1.0 / counts
        weights = weights / sum(weights)
    if method == "isns":
        weights = 1.0 / torch.pow(counts, 0.5)
        weights = weights / sum(weights)
    if method == "esns":
        beta = 0.999
        weights = (1.0 - beta) / (1.0 - torch.pow(beta, counts))
        weights = weights / sum(weights)

    return weights


def set_random_seed(seed=RANDOM_SEED):
    """Random seed for comparable results."""
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.cuda.manual_seed_all(seed)


def rsetattr(obj, attr, val):
    """Set an attribute recursively.

    Inspired by
    https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties.
    """
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    """Get an attribute recursively.

    Inspired by
    https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties.
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))
