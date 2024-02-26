"""Module for creating instantiating the appropriate model defined by the task list only."""

from typing import List

import torch

from enums.model_checkpoints import ModelCheckpoint
from enums.splits import Split
from training.data.multi_subtask_dataset import BatchList, BatchListEvalTest
from training.model.model import Model

import os


def ModelFactory(
    task_list: List, LM: ModelCheckpoint, sub_batch_size, eval_batch_size, pretrained_path, *args, **kwargs
):
    """Create model defined by task list and return the model as well as dataloaders."""
    # subtask preparation
    subtask_list = [st for t in task_list for st in t.subtasks_list]
    for st in subtask_list:
        assert st.processed, "Data must be loaded at this point."

    # model preparation
    model = Model(stl=subtask_list, LM=LM,**kwargs)
    # load_head_initializations(model) # TODO: optional

    if pretrained_path is not None:
        model = load_pretrained_weights(model, pretrained_path=pretrained_path)

    model.to(model.device)
    # DataLoaders preparation
    batch_list_train = BatchList(subtask_list=subtask_list, sub_batch_size=sub_batch_size, split=Split.TRAIN)
    batch_list_dev = BatchList(subtask_list=subtask_list, sub_batch_size=eval_batch_size, split=Split.DEV)
    batch_list_eval = BatchListEvalTest(subtask_list=subtask_list, sub_batch_size=sub_batch_size, split=Split.DEV)
    batch_list_test = BatchListEvalTest(subtask_list=subtask_list, sub_batch_size=sub_batch_size, split=Split.TEST)

    return model, batch_list_train, batch_list_dev, batch_list_eval, batch_list_test


def save_head_initializations(model):
    """Save weight initialization of the head. This method will not be called anymore.
     It's only for the initial saving of weight inits for all tasks."""
    for head_name in model.heads.keys():
        torch.save(model.heads[head_name].state_dict(), 'model_files/heads/' + head_name + '_init.pth')
    
def load_head_initializations(model):
    """Load fixed weight initialization for each head in order to ensure reproducibility."""
    for head_name in model.heads.keys():
        weights_path = 'model_files/heads/' + head_name + '_init.pth'
        head_weights = torch.load(weights_path)
        model.heads[head_name].load_state_dict(head_weights,strict=True)

def load_pretrained_weights(model, pretrained_path):
    """Load the weights of a pretrained model."""
    weight_dict = torch.load(pretrained_path)
    model.load_state_dict(weight_dict, strict=False)
    return model
