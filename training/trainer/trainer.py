"""This module contains the trainer class."""
import statistics as stats
from typing import Any, Dict, List
from config import MAX_NUMBER_OF_STEPS

import numpy as np
import torch
from tqdm import tqdm
from transformers import get_polynomial_decay_schedule_with_warmup

from enums.aggregation_method import AggregationMethod
from enums.model_checkpoints import ModelCheckpoint
from enums.scaling import LossScaling
from enums.splits import Split
from training.data.task import Task
from training.model.helper_classes import (
    EarlyStopper,
    EarlyStoppingMode,
    Logger,
    Tracker,
)
from training.model.model_factory import ModelFactory
from training.model.optimization.grad_aggregator import GradientAggregator


class Trainer:
    """Trainer class to train and evaluate a model."""

    def __init__(
        self,
        task_list: List[Task],
        LM: ModelCheckpoint,
        initial_lr,
        model_name: str,
        pretrained_path: str,
        sub_batch_size: int,
        eval_batch_size: int,
        early_stopping_mode,
        resurrection: bool,
        aggregation_method: AggregationMethod,
        loss_scaling: LossScaling,
        num_warmup_steps: int,
        head_specific_lr_dict: Dict[str, float],
        head_specific_patience_dict: Dict[str, int],
        head_specific_max_epoch_dict: Dict[str, int],
        logger: Logger,
        *args,
        **kwargs,
    ):
        """Initialize a Trainer."""
        self.early_stopping_mode = early_stopping_mode
        self.logger = logger
        self.loss_scaling = loss_scaling
        self.model, batch_list_train, batch_list_dev, batch_list_eval, batch_list_test = ModelFactory(
            task_list=task_list,
            LM=LM,
            sub_batch_size=sub_batch_size,
            eval_batch_size=eval_batch_size,
            pretrained_path=pretrained_path,
            *args,
            **kwargs,
        )
        self.batch_lists = {
            Split.TRAIN: batch_list_train,
            Split.DEV: batch_list_dev,
            Split.EVAL: batch_list_eval,
            Split.TEST: batch_list_test,
        }

        # shared backbone model optimizer
        self.lm_optimizer = torch.optim.AdamW(self.model.language_model.backbone.parameters(), lr=initial_lr)
        self.lm_lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=self.lm_optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max([len(dl) for dl in self.batch_lists[Split.TRAIN].dataloaders.values()])
            * stats.median(head_specific_max_epoch_dict.values()),
        )

        # task-specifics optimizers
        self.head_optimizers = {
            str(st_id): torch.optim.AdamW(head.parameters(), lr=head_specific_lr_dict[st_id])
            for st_id, head in self.model.heads.items()
        }
        self.head_lr_schedulers = {
            str(st_id): get_polynomial_decay_schedule_with_warmup(
                optimizer=self.head_optimizers[st_id],
                num_warmup_steps=num_warmup_steps,
                num_training_steps=len(self.batch_lists[Split.TRAIN].dataloaders[st_id])
                * head_specific_max_epoch_dict[st_id],
            )
            for st_id in self.model.heads.keys()
        }

        # flags controlling stopping and resurrection
        self.task_alive_flags = {str(st_id): True for st_id in self.model.heads.keys()}
        self.task_zombie_flags = {str(st_id): False for st_id in self.model.heads.keys()}
        self.early_stopper = EarlyStopper(
            st_ids=self.model.heads.keys(),
            mode=self.early_stopping_mode,
            patience=head_specific_patience_dict,
            resurrection=resurrection,
        )

        self.tracker = Tracker(heads=self.model.heads, logger=logger)
        self.GA = GradientAggregator(aggregation_method=aggregation_method)
        self.progress_bar = tqdm(range(len(self.model.heads)))
        self.model_name = model_name
        self.scaling_weights = {str(st.id): st.get_scaling_weight() for t in task_list for st in t.subtasks_list}
        self.MAX_NUMBER_OF_STEPS = MAX_NUMBER_OF_STEPS
        self.k = 50

    def head_specific_optimization(self, st_id: str, lm_grads, scaling_weight):
        """
        Perform the optimization of a task-specific head.

        This method is only called when mode is training.
        @param st_id: The subtask id.
        @param lm_grads: The LM gradients.
        @param scaling_weight: The scaling weight of that subtask.
        @return: A dictionary with additional payload containing the conflicting gradients ratio.
        """
        additional_payload = {}
        last_dev_loss = self.tracker.get_last_st_loss(split=Split.DEV, st_id=st_id, k=self.k)
        should_stop_now = (
            self.early_stopper.early_stop(st_id=st_id, dev_loss=last_dev_loss)
            if (self.task_alive_flags[st_id] or self.task_zombie_flags[st_id])
            else False
        )

        should_resurrect_now = (
            self.early_stopper.resurrect(st_id=st_id, dev_loss=last_dev_loss)
            if (not self.task_zombie_flags[st_id] and not self.task_alive_flags[st_id])
            else False
        )

        should_stay_zombie = not self.task_alive_flags[st_id] and self.task_zombie_flags[st_id] and not should_stop_now

        # Eval + Log task when it DIES
        if should_stop_now and self.task_alive_flags[st_id]:
            print(f"Subtask {st_id} is now DEAD.")
            self.eval_st(split=Split.EVAL, st_id=st_id)
            self.tracker.log(splits=[Split.EVAL], additional_payload={st_id + "_STOPPED": 0})
            self.progress_bar.update()

        # Eval + Log task when it RESURRECTS
        elif should_resurrect_now and not self.task_zombie_flags[st_id]:
            print(f"Subtask {st_id} is now ZOMBIE.")
            additional_payload[st_id + "_ZOMBIE"] = 0
            self.early_stopper.reset_early_stopper(st_id=st_id)

        # Eval + Log task when a ZOMBIE DIES
        elif should_stop_now and self.task_zombie_flags[st_id]:
            print(f"Subtask {st_id} is now DEAD AGAIN.")
            additional_payload[st_id + "_DEAD_ZOMBIE"] = 0
            self.early_stopper.reset_early_stopper(st_id=st_id)

        self.task_alive_flags[st_id] = self.task_alive_flags[st_id] and not (
            should_stop_now or self.tracker.get_last_st_metric(split=Split.DEV, st_id=st_id, k=10) == 1
        )
        self.task_zombie_flags[st_id] = should_resurrect_now or should_stay_zombie

        # We optimize a task if it is alive or zombie
        optimize_task = self.task_alive_flags[str(st_id)] or self.task_zombie_flags[str(st_id)]
        if optimize_task:
            self.head_optimizers[st_id].step()
            self.head_lr_schedulers[st_id].step()

        if self.early_stopping_mode != EarlyStoppingMode.BACKBONE or optimize_task:
            self.GA.update(lm_grads, scaling_weight=scaling_weight)

        return additional_payload

    def backbone_optimization(self) -> Dict[str, Any]:
        """
        Perform the optimization of the backbone.

        This method is only called when mode is training.
        @return: A dictionary with additional payload containing the conflicting gradients ratio.
        """
        # Optimize the LM such that: we aggregate gradients from subtasks and set the final
        # gradient to the LM and subsequently optimize (only the LM)
        additional_payload = {}
        if any(self.task_alive_flags.values()):
            aggregated_gradients = self.GA.aggregate_gradients()
            self.model.language_model.set_grads(aggregated_gradients)
            self.lm_optimizer.step()
            self.lm_lr_scheduler.step()
        if self.GA.aggregation_method in [AggregationMethod.PCGRAD, AggregationMethod.PCGRAD_ONLINE]:
            conflicting_gradients_ratio = self.GA.get_conflicting_gradients_ratio()
            additional_payload["conflicting_gradients_ratio"] = conflicting_gradients_ratio

        return additional_payload

    def handle_batch(self, batch, split: Split = Split.TRAIN) -> Dict[str, Any]:
        """Handle a batch.

         (always) Pass a batch of sub_batches through the network.
         (in train-mode) For each sub_batch, accumulate the gradients of the LM.
         For each sub_batch and each st_id,
            - (in train-mode) accumulate the gradients of the respective head,
            - (always) accumulate the metric of the respective head,
            - (always) accumulate the loss of the respective head.
        (always) Log all metrics and losses to wandb.
         (in train-mode) After all sub_batches are processed, normalize the LM gradients and the head-specific gradients.
         (in train-mode) Then, perform the step of the lr_scheduler and the optimizer.

        @param batch: The batch containing sub-batches.
        @param split: The split (TRAIN, DEV, TEST)
        @return: A dictionary containing additional payload that needs to be logged.
        """
        training = split == Split.TRAIN
        losses = []
        additional_payloads: Dict[str, Any] = {}
        # reset accumulator only if it's a new batch for training, otherwise eval drops accumulated gradients
        if training:
            self.GA.reset_accumulator()

        # sub_batch consists of data of one subtask only
        for sub_batch in batch:
            X, attention_masks, Y, st_id = sub_batch
            loss, metric_values, lm_grads = self._step((X, attention_masks, Y, st_id.unique()), training=training)
            st_id = str(st_id.unique().item())
            scaling_weight = self.scaling_weights[st_id] if self.loss_scaling == LossScaling.STATIC else 1.0

            if training:
                additional_payload = self.head_specific_optimization(
                    st_id=st_id, lm_grads=lm_grads, scaling_weight=scaling_weight
                )
                additional_payloads = {**additional_payload, **additional_payloads}

            # Update losses & metrics
            for metric, value in metric_values.items():
                self.tracker.update_metric(split=split, st_id=st_id, metric=metric, value=value)
            self.tracker.update_loss(split=split, st_id=st_id, value=loss.item())
            losses.append(loss.item())

        if training:
            additional_payload = self.backbone_optimization()
            additional_payloads = {**additional_payload, **additional_payloads}

        self.tracker.update_combined_loss(split=split, value=np.mean(losses))
        return additional_payloads

    def fit(self):
        """Fit a model."""
        step = 0
        
        for i in range(self.MAX_NUMBER_OF_STEPS):
            additional_payload_train, additional_payload_dev = {}, {}
            # Check if any task is still training
            if not any(self.task_alive_flags.values()):
                break
            step += 1
            batch = next(self.batch_lists[Split.TRAIN])
            additional_payload_train = self.handle_batch(batch=batch, split=Split.TRAIN)
            if step % 3 == 0:
                batch = next(self.batch_lists[Split.DEV])
                additional_payload_dev = self.handle_batch(batch=batch, split=Split.DEV)
            self.refresh_pbar()
            self.tracker.log(
                splits=[Split.TRAIN, Split.DEV],
                additional_payload={**additional_payload_train, **additional_payload_dev},
            )

        self.eval(split=Split.EVAL)

    def _step(self, batch, training: bool = True):
        """
        Make one step.

        @param batch: A dictionary containing X, Y, std_ids and attention_masks.
        """
        inputs = {"X": batch[0], "attention_masks": batch[1], "Y": batch[2], "st_id": batch[3]}
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        if training:
            self.model.train()
            loss, heads_metrics_values = self.model(**inputs)
            self.lm_optimizer.zero_grad()
            for st_id, optim in self.head_optimizers.items():
                optim.zero_grad()
            loss.backward()
            lm_gradients = self.model.language_model.get_grads()
        else:
            self.model.eval()
            lm_gradients = None
            with torch.no_grad():
                loss, heads_metrics_values = self.model(**inputs)

        del inputs
        return loss, heads_metrics_values, lm_gradients

    def eval(self, split):
        """Evaluate the model on the entire test or dev set."""
        assert split in [Split.EVAL, Split.TEST]

        for st_id in self.batch_lists[split].iter_dataloaders.keys():
            self.eval_st(split=split, st_id=st_id)

        self.tracker.log(splits=[split])

    def eval_st(self, split, st_id):
        """Evaluate on a subtask, given a certain split."""
        batch_list = self.batch_lists[split]
        batch_list._reset()
        idl = batch_list.iter_dataloaders[st_id]
        for batch in idl:
            _ = self.handle_batch(batch=[batch], split=split)

    def refresh_pbar(self):
        """Update the progress bar."""
        desc = str(self.tracker)
        self.progress_bar.set_description(desc=desc)
        self.progress_bar.refresh()

    def fit_debug(self, k: int):
        """Fit for k iterations only to check if a model can process the data."""
        step = 0
        for _ in range(k):
            step += 1
            batch = next(self.batch_lists[Split.TRAIN])
            self.handle_batch(batch=batch, split=Split.TRAIN)
            # Evaluate on dev-batch
            batch = next(self.batch_lists[Split.DEV])
            self.handle_batch(batch=batch, split=Split.DEV)

    def save_model(self):
        """Save the model."""
        model_files_path = "model_files/" + self.model_name + ".pth"
        torch.save(self.model.state_dict(), model_files_path)