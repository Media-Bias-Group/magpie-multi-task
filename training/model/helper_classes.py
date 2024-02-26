"""This module contains helper classes for model training."""

import copy
import logging
import math
import os
from enum import Enum
from typing import Dict, List

import numpy as np
import torch

import wandb
from enums.splits import Split


class Logger:
    """Logger to keep track of metrics, losses and artifacts.

    This logger is used as an abstraction. If we want to integrate with third party providers (wandb, GCS, ...),
    use this logger.
    """

    def __init__(self, experiment_name: str):
        """Initialize a Logger."""
        PATH = "logging/" + experiment_name
        os.makedirs(PATH, exist_ok=True)

        self.experiment_logfilename = PATH + "/train_data.log"
        experiment_logfile_handler = logging.FileHandler(filename=self.experiment_logfilename)
        experiment_logfile_formatter = logging.Formatter(fmt="%(message)s")
        experiment_logfile_handler.setFormatter(experiment_logfile_formatter)

        self.experiment_logger = logging.getLogger("experiment_logger")
        self.experiment_logger.addHandler(experiment_logfile_handler)
        self.experiment_logger.setLevel("INFO")

    def log(self, out):
        """Log."""
        self.experiment_logger.info(out)
        wandb.log(out)


class EarlyStopperSingle:
    """
    EarlyStopper for a single branch of the model.

    Inspired by .https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch.
    """

    def __init__(self, patience: int, min_delta: int, resurrection: bool):
        """Initialize an EarlyStopperSingle."""
        self.patience = patience
        self.patience_zombie = 10
        self.min_delta = min_delta
        self.counter = 0
        self.counter_zombie = 0
        self.min_dev_loss = np.inf
        self.min_dev_loss_zombie = np.inf
        self.resurrection = resurrection

    def early_stop(self, dev_loss):
        """Return True if dev_loss is steadily increasing."""
        if math.isnan(dev_loss):
            return False
        if dev_loss < self.min_dev_loss:
            self.min_dev_loss = dev_loss
            self.counter = 0
        elif dev_loss > (self.min_dev_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def resurrect(self, dev_loss):
        """Return True if dev_loss is steadily increasing and a dead task should resurrect."""
        if math.isnan(dev_loss) or not self.resurrection:
            return False
        if dev_loss < self.min_dev_loss_zombie:
            self.min_dev_loss_zombie = dev_loss
            self.counter_zombie = 0
        elif dev_loss > self.min_dev_loss_zombie:
            self.counter_zombie += 1
            if self.counter_zombie >= self.patience_zombie:
                return True
        return False

    def reset_early_stopper(self):
        """Reset the state of an early stopper.

        As zombies can resurrect and die multiple times, we have to reset their internal variables,
        counter and the min_dev_loss each time a zombie resurrects or dies.
        """
        self.counter_zombie = 0
        self.counter = 0
        self.min_dev_loss_zombie = np.inf
        self.min_dev_loss = np.inf


class EarlyStoppingMode(Enum):
    """Enum for early stopping mode."""

    HEADS = "heads"  # Only stop heads
    BACKBONE = "backbone"  # Also stop backbone
    NONE = "none"


class EarlyStopper:
    """EarlyStopper container for all heads."""

    def __init__(self, st_ids: List[str], mode: EarlyStoppingMode, patience, resurrection: bool, min_delta=0):
        """Initialize an EarlyStopper."""
        self.mode = mode
        self.early_stoppers = {
            st_id: EarlyStopperSingle(patience=patience[st_id], min_delta=min_delta, resurrection=resurrection)
            for st_id in st_ids
        }

    def early_stop(self, st_id, dev_loss):
        """Return True if dev_loss is steadily increasing."""
        return (
            False if self.mode == EarlyStoppingMode.NONE else self.early_stoppers[st_id].early_stop(dev_loss=dev_loss)
        )

    def resurrect(self, st_id, dev_loss):
        """Return True if dev_loss is steadily increasing and a dead task should resurrect."""
        return (
            False if self.mode == EarlyStoppingMode.NONE else self.early_stoppers[st_id].resurrect(dev_loss=dev_loss)
        )

    def reset_early_stopper(self, st_id):
        """Reset the state of an early stopper."""
        self.early_stoppers[st_id].reset_early_stopper()


class Accumulator:
    """Abstract Accumulator."""

    def __init__(self):
        """Raise RuntimeError if this Accumulator is instantiated."""
        if type(self) == Accumulator:
            raise RuntimeError("Abstract class <Accumulator> must not be instantiated.")
        self.gradients = None
        self.n = 0

    def update(self, gradients):
        """Update the values of a gradient.

        Must be overwritten by concrete implementation.
        """
        raise NotImplementedError

    def get_avg_gradients(self):
        """Return the gradients, normalized across 0-axis."""
        out_gradients = copy.deepcopy(self.gradients)
        for k, v in self.gradients.items():
            out_gradients[k] /= self.n
            out_gradients[k] = out_gradients[k].squeeze(dim=0)
        return out_gradients

    def get_gradients(self):
        """Return the gradients.

        Must be overwritten by concrete implementation.
        """
        return self.gradients


class StackedAccumulator(Accumulator):
    """Accumulate the gradients for one SubTask within on Super-Batch."""

    def __init__(self):
        """Initialize a StackedAccumulator."""
        super(StackedAccumulator, self).__init__()

    def update(self, gradients, weight=1.0):
        """Update. Concatenate new set of gradients along 0-axis."""
        if not self.gradients:
            self.gradients = gradients
            # unsqueeze all gradients for later concatenation
            for k, v in self.gradients.items():
                self.gradients[k] = self.gradients[k].unsqueeze(dim=0) * weight
        else:
            for k, v in self.gradients.items():
                new_value = gradients[k].unsqueeze(dim=0) * weight
                self.gradients[k] = torch.cat((v, new_value), dim=0)
        self.n += 1

    def set_gradients(self, gradients: Dict[str, torch.tensor]):
        """Set the gradients."""
        for k, v in self.gradients.items():
            self.gradients[k] = gradients[k].unsqueeze(dim=0)


class RunningSumAccumulator(Accumulator):
    """Keep track of the running sum of gradients."""

    def __init__(self):
        """Initialize a RunningSumAccumulator."""
        super(RunningSumAccumulator, self).__init__()

    def update(self, gradients: Dict[str, torch.tensor], weight=1.0) -> None:
        """Update. Sum the gradients along 0-axis."""
        if not self.gradients:
            self.gradients = gradients
            # unsqueeze all gradients for later concatenation
            for k, v in self.gradients.items():
                self.gradients[k] = self.gradients[k].unsqueeze(dim=0) * weight
        else:
            for k, v in self.gradients.items():
                new_value = gradients[k].unsqueeze(dim=0) * weight
                self.gradients[k] = torch.add(v, new_value)
        self.n += 1


class AverageMeter:
    """The AverageMeter keeps track of a metric."""

    def __init__(self, name):
        """Initialize an AverageMeter."""
        self.values = []
        self.name = name

    def mean_last_k(self, k=10):
        """Return the mean of the last k values."""
        assert 1 <= k
        vals = self.values[-k:]
        if len(vals) < k:
            return float("NaN")

        return np.mean(vals)

    def mean_all(self):
        """Return the mean of all values."""
        return np.mean(self.values)

    def update(self, value=0):
        """Update the Metric by appending a new value."""
        self.values.append(value)

    def reset(self):
        """Reset AverageMeter."""
        self.values.clear()

    def __repr__(self):
        """Print."""
        return f"{self.mean_last_k(1):.2f}"


class Tracker:
    """Keep track of all metrics and losses of an epoch."""

    def __init__(self, heads, logger: Logger):
        """Initialize a Tracker."""
        self.metrics = self.init_metrics(heads=heads)
        self.losses, self.combined_losses = self.init_losses(heads=heads)
        self.logger = logger

    def init_losses(self, heads):
        """Initialize the losses."""
        train_losses = {f"{st_id}": AverageMeter(name=f"{st_id}_train_loss") for st_id, head in heads.items()}
        dev_losses = {f"{st_id}": AverageMeter(name=f"{st_id}_dev_loss") for st_id, head in heads.items()}
        eval_losses = {f"{st_id}": AverageMeter(name=f"{st_id}_eval_loss") for st_id, head in heads.items()}
        test_losses = {f"{st_id}": AverageMeter(name=f"{st_id}_test_loss") for st_id, head in heads.items()}
        combined_losses = {
            Split.TRAIN: AverageMeter(name="combined_train_loss"),
            Split.DEV: AverageMeter(name="combined_dev_loss"),
            Split.TEST: AverageMeter(name="combined_test_loss"),
            Split.EVAL: AverageMeter(name="combined_eval_loss"),
        }
        return {
            Split.TRAIN: train_losses,
            Split.DEV: dev_losses,
            Split.TEST: test_losses,
            Split.EVAL: eval_losses,
        }, combined_losses

    def init_metrics(self, heads=Dict):
        """Initialize the AverageMeters for the metrics."""
        train_metrics = {
            st_id: {m: AverageMeter(name=f"{st_id}_train_{m}") for m in head.metrics.keys()}
            for st_id, head in heads.items()
        }

        dev_metrics = {
            st_id: {m: AverageMeter(name=f"{st_id}_dev_{m}") for m in head.metrics.keys()}
            for st_id, head in heads.items()
        }

        eval_metrics = {
            st_id: {m: AverageMeter(name=f"{st_id}_eval_{m}") for m in head.metrics.keys()}
            for st_id, head in heads.items()
        }

        test_metrics = {
            st_id: {m: AverageMeter(name=f"{st_id}_test_{m}") for m in head.metrics.keys()}
            for st_id, head in heads.items()
        }
        return {Split.TRAIN: train_metrics, Split.DEV: dev_metrics, Split.TEST: test_metrics, Split.EVAL: eval_metrics}

    def update_metric(self, split, st_id, metric, value):
        """Update the metric, given a split and subtask id."""
        self.metrics[split][st_id][metric].update(value=value)

    def update_loss(self, split, st_id, value):
        """Update the loss, given a split and subtask id."""
        self.losses[split][st_id].update(value)

    def update_combined_loss(self, split, value):
        """Update the combined losses, given a split."""
        self.combined_losses[split].update(value)

    def get_last_st_loss(self, split, st_id, k):
        """Get mean of last subtask loss."""
        return self.losses[split][st_id].mean_last_k(k=k)

    def get_last_st_metric(self, split, st_id, k):
        """Get mean of last subtask metric."""
        return self.metrics[split][st_id][next(iter(self.metrics[split][st_id]))].mean_last_k(k=k)

    def __repr__(self):
        """Represent a Tracker."""
        return f"TRAIN LOSS: {self.combined_losses[Split.TRAIN]} - DEV LOSS: {self.combined_losses[Split.DEV]} - EVAL LOSS: {self.combined_losses[Split.EVAL]}"

    def log(self, splits: List[Split], additional_payload: Dict[str, float] = {}):
        """Log the metrics & losses of a list of splits."""
        out: Dict[str, float] = {**additional_payload}
        for split in splits:
            if split in [Split.DEV, Split.TRAIN]:
                metrics = {m.name: m.mean_last_k(1) for d in self.metrics[split].values() for m in d.values()}
                combined_losses = self.combined_losses[split]
                losses = {v.name: v.mean_last_k(1) for v in self.losses[split].values()}
                out = {**out, **metrics, combined_losses.name: combined_losses.mean_last_k(1), **losses}
            else:
                metrics = {m.name: m.mean_all() for d in self.metrics[split].values() for m in d.values()}
                combined_losses = self.combined_losses[split]
                losses = {v.name: v.mean_all() for v in self.losses[split].values()}
                out = {**out, **metrics, combined_losses.name: combined_losses.mean_all(), **losses}

        self.logger.log(out)
