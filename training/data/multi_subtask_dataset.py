"""This module contains the SubTaskDataset."""
import random
from typing import List

import numpy as np
from torch.utils.data import DataLoader, Dataset

from enums.splits import Split
from training.data.subtask import SubTask
from utils import set_random_seed


class SubTaskDataset(Dataset):
    """A Datset for a single SubTask."""

    def __init__(self, subtask: SubTask, split: Split):
        """Initialize a SubTaskDataset."""
        self.split = split
        self.subtask = subtask
        self.observations: List = []
        self._reset()

    def __len__(self):
        """Get the length of the Dataset."""
        return len(self.observations)

    def __getitem__(self, item):
        """Get the next observation from the Dataset."""
        if self._counter == len(self.observations):
            self._reset()
        i = self.observations[self._counter]
        x = self.subtask.get_X(split=self.split)[i]
        masks = self.subtask.get_att_mask(split=self.split)[i]
        y = self.subtask.get_Y(split=self.split)[i]
        self._counter += 1
        return x, masks, y, self.subtask.id

    def _reset(self):
        self.observations = [i for i in range(len(self.subtask.get_X(split=self.split)))]
        set_random_seed()
        np.random.shuffle(self.observations)  # Not a real 'reshuffling' as it will always arrange same.
        self._counter = 0


class BatchList:
    """A BatchList is a wrapper around dataloaders for each subtask.

    This BatchList will never stop; it will always yield super-batches containing one sub-batch per task.
    """

    def __init__(self, subtask_list: List[SubTask], sub_batch_size, split=Split.TRAIN):
        """Initialize a BatchList."""
        self.sub_batch_size = sub_batch_size
        self.datasets = {f"{st.id}": SubTaskDataset(subtask=st, split=split) for st in subtask_list}
        self.dataloaders = {
            f"{st_id}": DataLoader(ds, batch_size=self.sub_batch_size) for st_id, ds in self.datasets.items()
        }
        self.iter_dataloaders = {f"{st_id}": iter(dl) for st_id, dl in self.dataloaders.items()}

    def __next__(self):
        """Yield a batch of sub-batches."""
        data = []
        items = list(self.iter_dataloaders.items())  # List of tuples of (key,values)
        random.shuffle(items)
        for st_id, dl in items:
            try:
                batch = next(dl)
            except StopIteration:
                self.iter_dataloaders[st_id] = iter(self.dataloaders[st_id])  # Reset the iter_dataloader
                batch = next(self.iter_dataloaders[st_id])
            data.append(batch)
        return data  # Batch contains Sub-batches.

    def _reset(self):
        """Reset this BatchListEvalTest."""
        self.iter_dataloaders = {f"{st_id}": iter(dl) for st_id, dl in self.dataloaders.items()}

class BatchListEvalTest:
    """A BatchListEvalTest is a wrapper around dataloaders for each subtask.

    If one task is exhausted, it will stop yielding sub-batches from this task.
    Instead, it will continue until it has yielded all sub-batches from all tasks.
    """

    def __init__(self, subtask_list: List[SubTask], sub_batch_size, split=Split.TRAIN):
        """Initialize a BatchList."""
        self.sub_batch_size = sub_batch_size
        self.datasets = {f"{st.id}": SubTaskDataset(subtask=st, split=split) for st in subtask_list}
        self.dataloaders = {
            f"{st_id}": DataLoader(ds, batch_size=self.sub_batch_size) for st_id, ds in self.datasets.items()
        }
        self.iter_dataloaders = {f"{st_id}": iter(dl) for st_id, dl in self.dataloaders.items()}

    def __len__(self):
        """Return the length of this BatchListEvalTest.

        The length is the maximum length of all subtask-datadloaders.
        """
        return sum([len(dl) for dl in self.dataloaders.values()])

    def _reset(self):
        """Reset this BatchListEvalTest."""
        self.iter_dataloaders = {f"{st_id}": iter(dl) for st_id, dl in self.dataloaders.items()}
