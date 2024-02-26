"""This module contains the GradTS trainer class. It is deprecated now."""

from typing import Dict, List

# import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import torch
from torch.nn.functional import normalize
from tqdm import tqdm
from transformers import get_polynomial_decay_schedule_with_warmup

from enums.model_checkpoints import ModelCheckpoint
from enums.splits import Split
from training.data.task import Task
from training.model.helper_classes import Logger, Tracker
from training.model.model_factory import ModelFactory
from training.trainer.trainer import Trainer



class GradTSTrainer(Trainer):
    """Simplified Trainer class for single-task training with gradient accumulation for the purpose of GradTS experiment."""

    def __init__(
        self,
        task_list: List[Task],
        LM: ModelCheckpoint,
        initial_lr,
        model_name: str,
        pretrained_path: str,
        sub_batch_size: int,
        eval_batch_size: int,
        num_warmup_steps: int,
        head_specific_lr_dict: Dict[str, float],
        logger: Logger,
        number_of_epochs: int,
        st_id: str,
        *args,
        **kwargs,
    ):
        """Initialize a Trainer."""
        self.st_id = st_id
        self.logger = logger
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
            num_training_steps=len(self.batch_lists[Split.TRAIN].dataloaders[st_id]) * number_of_epochs,
        )
        self.head_optimizers = {
            str(st_id): torch.optim.AdamW(head.parameters(), lr=head_specific_lr_dict[st_id])
            for st_id, head in self.model.heads.items()
        }
        self.head_lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=self.head_optimizers[self.st_id],
            num_warmup_steps=num_warmup_steps,
            num_training_steps=len(self.batch_lists[Split.TRAIN].dataloaders[st_id]) * number_of_epochs,
        )

        self.tracker = Tracker(heads=self.model.heads, logger=logger)
        self.progress_bar = tqdm(range(len(self.model.heads)))
        self.model_name = model_name
        self.number_of_epochs = 3

    def handle_batch(self, batch, accumulating):
        """Handle single batch in 'single-task' manner."""
        losses = []
        att_head_mat = None

        X, attention_masks, Y, st_id = batch
        loss, metric_values, lm_grads = self._step((X, attention_masks, Y, st_id.unique()), training=True)
        st_id = str(st_id.unique().item())

        # apply opt steps in first 3 epochs
        if not accumulating:
            self.head_optimizers[self.st_id].step()
            self.head_lr_scheduler.step()
            self.lm_optimizer.step()
            self.lm_lr_scheduler.step()
        else:
            att_head_mat = get_attention_head_matrix(lm_grads)

        # Update losses & metrics
        for metric, value in metric_values.items():
            self.tracker.update_metric(split=Split.TRAIN, st_id=st_id, metric=metric, value=value)
        self.tracker.update_loss(split=Split.TRAIN, st_id=st_id, value=loss.item())
        losses.append(loss.item())

        self.tracker.update_combined_loss(split=Split.TRAIN, value=np.mean(losses))

        return att_head_mat

    def fit(self):
        """Fit a model."""
        batch_list = self.batch_lists[Split.TRAIN]
        batch_list._reset()
        idl = batch_list.iter_dataloaders[self.st_id]

        # forward+backward pass for fixed number of epochs (3)
        for i in range(self.number_of_epochs):
            for batch in idl:
                self.handle_batch(batch=batch, accumulating=False)
                self.refresh_pbar()
                self.tracker.log(splits=[Split.TRAIN])

        att_head_mat = torch.zeros(size=(12, 12))
        batch_list._reset()
        idl = batch_list.iter_dataloaders[self.st_id]
        # one forward pass (with subsequent gradient extraction)
        for batch in idl:
            batch_att_mat = self.handle_batch(batch=batch, accumulating=True)
            att_head_mat += batch_att_mat
            self.refresh_pbar()
            self.tracker.log(splits=[Split.TRAIN])


        def norm(a):
            a = (a - a.min()) / float(a.max() - a.min())
            return a

        # average att_matrix
        att_head_mat = att_head_mat / len(idl)
        att_head_mat = normalize(att_head_mat, p=1.0, dim=1)
        att_head_mat = norm(att_head_mat)

        model_files_path = "model_files/gradts/" + str(self.st_id) + "_att_mat.pt"
        torch.save(att_head_mat,model_files_path)


def get_attention_head_matrix(lm_grads):
    number_of_layers = 12
    number_of_heads = 12
    hidden_size = 768
    head_mat = torch.empty(size=(number_of_layers, number_of_heads))

    for layer_idx in range(number_of_layers):
        k = lm_grads["backbone.encoder.layer." + str(layer_idx) + ".attention.self.key.weight"]
        v = lm_grads["backbone.encoder.layer." + str(layer_idx) + ".attention.self.value.weight"]
        k_v = np.abs(k.detach().cpu()) + np.abs(v.detach().cpu())
        k_v = k_v.view(number_of_heads, hidden_size, int(hidden_size / number_of_heads))
        k_v = k_v.mean(axis=[1, 2])
        head_mat[layer_idx] = k_v

    return head_mat


