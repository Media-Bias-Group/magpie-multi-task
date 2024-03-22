"""This module contains the Subtask."""

import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from config import DEV_RATIO, MAX_LENGTH, REGRESSION_SCALAR, TRAIN_RATIO
from enums.splits import Split
from training.tokenizer import tokenizer
from utils import get_class_weights


def get_pos_idxs(pos: str, text: str):
    """
    Get the correct idxs of the pos for a given text.

    @param pos: A pattern as text.
    @param text: The text to search trough.
    @return: The ids of the tokens in the text that match the pattern.
    """
    if pos == text:
        mask = np.array(np.ones((len(text))), dtype="int")
    else:
        pos = pos.replace("[", "\[")
        pos = pos.replace("$", "\$")
        pos = pos.replace("?", "\?")
        pos = pos.replace(")", "\)")
        pos = pos.replace("(", "\(")
        pos = pos.replace("*", "\*")
        pos = pos.replace("+", "\+")
        start, end = re.search(pos, text).span()

        mask = np.zeros((len(text)), dtype=int)
        mask[start:end] = 1
    c, idx_list = 0, []
    for t in text.split():
        idx_list.append(c)
        c += len(t) + 1
    mask_idxs = [mask[i] for i in idx_list]
    return mask_idxs


def align_labels_with_tokens(labels: List[int], word_ids: List[int]):
    """Align labels with tokens.

    C/p from https://huggingface.co/course/chapter7/2
    """
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = (
                -100 if word_id is None else labels[word_id]
            )  # -100 is an index that will be ignored by cross entropy
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def get_tokens_and_labels(pos_list_list, text_list, labels):
    """Get tokens and labels for scattered POS.

    In this objective, we have a list of consecutive spans.
    For each of these consecutive spans, find the correct index of the corresponding tokens in the text_list.
    Returns the bitwise or ('union') of this ids.
    """
    mask_idxs_list = []
    for i, pos_list in enumerate(pos_list_list):
        label = labels[i]
        text = text_list[i]
        observation_mask_idxs = []
        for pos in pos_list:
            if len(pos) == 0:
                # If there is no POS, we just return zeros
                observation_mask_idxs.append(get_pos_idxs("", text))
            else:
                for pos in pos_list:
                    if label == 0:  # In that case, the label is the neutral class
                        observation_mask_idxs.append(get_pos_idxs(pos, text))
                    else:
                        pos_idxs = get_pos_idxs(pos, text)
                        pos_idxs = [label if idx == 1 else 0 for idx in pos_idxs]
                        observation_mask_idxs.append(pos_idxs)

        # reduce observation_mask_idxs
        observation_mask_idxs = np.bitwise_or.reduce(observation_mask_idxs, axis=0)
        mask_idxs_list.append(observation_mask_idxs)

    return [t.split() for t in text_list], mask_idxs_list


class SubTask:
    """A Subtask."""

    def __init__(self, id, task_id, filename, src_col="text", tgt_cols_list=["label"], *args, **kwargs):
        """Raise RuntimeError if this SubTask is instantiated."""
        if type(self) == SubTask:
            raise RuntimeError("Abstract class <SubTask> must not be instantiated.")
        self.attention_masks = None
        self.Y = None
        self.X = None
        self.class_weights = None
        self.id = id
        self.src_col = src_col
        self.tgt_cols_list = tgt_cols_list
        self.task_id = task_id
        self.filename = os.path.join("datasets", filename)
        self.processed = False

    def process(self, force_download: bool = False):
        """Process a SubTask.

        Load the data for this subtask, set properties X, Y and attention_mask.
        """
        X, Y, attention_masks = self.load_data()

        train_split = int(len(X) * TRAIN_RATIO)
        dev_split = train_split + int(len(X) * DEV_RATIO)

        self.X = {Split.TRAIN: X[:train_split], Split.DEV: X[train_split:dev_split], Split.TEST: X[dev_split:]}
        self.attention_masks = {
            Split.TRAIN: attention_masks[:train_split],
            Split.DEV: attention_masks[train_split:dev_split],
            Split.TEST: attention_masks[dev_split:],
        }
        self.Y = {Split.TRAIN: Y[:train_split], Split.DEV: Y[train_split:dev_split], Split.TEST: Y[dev_split:]}
        self.create_class_weights()
        self.processed = True

    def load_data(self) -> Tuple:
        """Load the data of a SubTask.

        Must be implemented for inherited.
        """
        raise NotImplementedError

    def create_class_weights(self):
        """Compute the weights for imbalanced classes."""
        pass

    def get_scaling_weight(self):
        """Get the scaling weight of a Subtask.

        Needs to be overwritten.
        """
        raise NotImplementedError

    def get_X(self, split: Split):
        """Get all X of a given split."""
        return self.X[split]

    def get_att_mask(self, split: Split):
        """Get attention_masks for inputs of a given split."""
        return self.attention_masks[split]

    def get_Y(self, split: Split):
        """Get all Y of a given split."""
        return self.Y[split]

    def __str__(self) -> str:
        return str(self.id)


# a[43485:43500]
class ClassificationSubTask(SubTask):
    """A ClassificationSubTask."""

    def __init__(self, num_classes=2, *args, **kwargs):
        """Initialize a ClassificationSubTask."""
        super(ClassificationSubTask, self).__init__(num_classes=num_classes, *args, **kwargs)
        self.num_classes = num_classes

    def load_data(self) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """Load the data of a ClassificationSubTask."""
        df = pd.read_csv(self.filename)

        X, Y = df[self.src_col], df[self.tgt_cols_list]
        tokenized_inputs = tokenizer(X.to_list(), padding="max_length", truncation=True, max_length=MAX_LENGTH)
        X = tokenized_inputs.get("input_ids")
        attention_masks = tokenized_inputs.get("attention_mask")
        assert Y.nunique().squeeze() == self.num_classes
        assert Y[self.tgt_cols_list[0]].min(axis=0) == 0
        if self.num_classes == 2:  # if it's binary classification
            Y = Y.to_numpy()
        else:
            Y = Y[self.tgt_cols_list].to_numpy()
        return torch.LongTensor(X), torch.LongTensor(Y), torch.LongTensor(attention_masks)

    def __repr__(self):
        """Represent a Classification Subtask."""
        return f"{'Multi-class' if self.num_classes != 2 else 'Binary'} Classification"

    def create_class_weights(self):
        """Compute the weights."""
        self.class_weights = get_class_weights(self.Y[Split.TRAIN], method="isns")

    def get_scaling_weight(self):
        """Get the weight of a Classification Subtask.

        As with the other tasks, we normalize by the natural logarithm of the domain size.
        """
        return 1 / np.log(self.num_classes)



class SoftClassificationSubTask(SubTask):
    """A ClassificationSubTask."""

    def __init__(self, num_classes=2, *args, **kwargs):
        """Initialize a ClassificationSubTask."""
        super(SoftClassificationSubTask, self).__init__(num_classes=num_classes, *args, **kwargs)
        self.num_classes = num_classes

    def load_data(self) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """Load the data of a ClassificationSubTask."""
        df = pd.read_csv(self.filename)

        X, Y = df[self.src_col], df[self.tgt_cols_list]
        tokenized_inputs = tokenizer(X.to_list(), padding="max_length", truncation=True, max_length=MAX_LENGTH)
        X = tokenized_inputs.get("input_ids")
        attention_masks = tokenized_inputs.get("attention_mask")
        # assert Y.nunique().squeeze() == self.num_classes
        # assert Y[self.tgt_cols_list[0]].min(axis=0) == 0

        Y = Y[self.tgt_cols_list].to_numpy()
        return torch.LongTensor(X), torch.Tensor(Y), torch.LongTensor(attention_masks)

    def __repr__(self):
        """Represent a Classification Subtask."""
        return f"{'Multi-class' if self.num_classes != 2 else 'Binary'} Classification"

    def create_class_weights(self):
        """Compute the weights."""
        self.class_weights = get_class_weights(self.Y[Split.TRAIN], method="isns")

    def get_scaling_weight(self):
        """Get the weight of a Classification Subtask.

        As with the other tasks, we normalize by the natural logarithm of the domain size.
        """
        return 1 / np.log(self.num_classes)


class RegressionSubTask(SubTask):
    """A RegressionSubTask."""

    def __init__(self, *args, **kwargs):
        """Initialize a RegressionSubTask."""
        super(RegressionSubTask, self).__init__(*args, **kwargs)

    def load_data(self) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.LongTensor]:
        """Load the data of a RegressionSubTask."""
        df = pd.read_csv(self.filename)
        X, Y = df[self.src_col], df[self.tgt_cols_list]
        tokenized_inputs = tokenizer(X.to_list(), padding="max_length", truncation=True, max_length=MAX_LENGTH)
        X = tokenized_inputs.get("input_ids")
        attention_masks = tokenized_inputs.get("attention_mask")
        Y = (((Y - Y.min()) / (Y.max() - Y.min())).to_numpy()).astype("float32")  # scale from 0 to 1

        return torch.LongTensor(X), torch.FloatTensor(Y), torch.LongTensor(attention_masks)

    def __repr__(self):
        """Represent a Regression Subtask."""
        return "Regression"

    def get_scaling_weight(self):
        """Get the scaling weight of a Regression Subtask.

        As of now, this scaling weight is a simple scalar and is a mere heuristic-based approximation (ie. we eyeballed it).
        """
        return REGRESSION_SCALAR


class MultiLabelClassificationSubTask(SubTask):
    """A MultiLabelClassificationSubTask."""

    def __init__(self, num_classes=2, num_labels=2, *args, **kwargs):
        """Initialize a MultiLabelClassificationSubTask."""
        super(MultiLabelClassificationSubTask, self).__init__(num_classes=2, num_labels=2, *args, **kwargs)
        self.num_classes = num_classes
        self.num_labels = num_labels

    def load_data(self) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """Load the data of a MultiLabelClassificationSubTask."""
        df = pd.read_csv(self.filename)
        X, Y = df[self.src_col], df[self.tgt_cols_list]
        tokenized_inputs = tokenizer(X.to_list(), padding="max_length", truncation=True, max_length=MAX_LENGTH)
        X = tokenized_inputs.get("input_ids")
        attention_masks = tokenized_inputs.get("attention_mask")
        assert Y.max(axis=0).to_numpy().max() == 1
        Y = Y.to_numpy()
        return torch.LongTensor(X), torch.LongTensor(Y), torch.LongTensor(attention_masks)

    def __repr__(self):
        """Represent a Multi-label Classification Subtask."""
        return "Multi-label Classification"

    def get_scaling_weight(self):
        """Get the weight of a Multi-label Classification Subtask.

        As with the other tasks, we normalize by the natural logarithm of the domain size.
        """
        return 1 / np.log(self.num_classes * self.num_labels)


class POSSubTask(SubTask):
    """A POSSubTask.

    Each POSSubTask can be either binary classification or multiclass classification.
    If it is binary classification, zero (0) must be the neutral class.
    This neutral class is also applied to all other, 'normal' tokens.
    """

    def __init__(self, tgt_cols_list, label_col=None, *args, **kwargs):
        """Initialize a POSSubTask.

        Normally, we have 3 classes: (0=no-tag, 1=tag-start, 2=tag-continue)
        However, we have POS-tasks where we have more than just 'binary token level classification'.
        In these scenarios, each class has two tags: 'tag-start' and 'tag-continue'.
        The 'no-class' tag has no 'tag-continue'.
        """
        super(POSSubTask, self).__init__(tgt_cols_list=tgt_cols_list, *args, **kwargs)
        # self.num_classes = 3   # The default num_classes is 2 or 3 (0=no-tag, 1=tag-start, 2=tag-continue)
        self.label_col = label_col
        assert len(tgt_cols_list) == 1

    def load_data(self) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """Load the data of a POSSubTask."""
        df = pd.read_csv(self.filename)

        df[self.tgt_cols_list] = df[self.tgt_cols_list].fillna("")
        mask = df.apply(
            lambda row: all([p in row[self.src_col] for p in row[self.tgt_cols_list[0]].split(";")]), axis=1
        )
        df = df[mask].reset_index(drop=True)
        assert sum(mask) == len(df[self.tgt_cols_list]), "At least one POS is not contained in the source column."

        pos_list_list = df[self.tgt_cols_list[0]].apply(lambda x: x.split(";")).to_list()
        X = df[self.src_col].values
        # If we do not provide a labels column, we assume that, whenever a pos is present, that is the non-neutral class
        labels = (
            df[self.label_col]
            if self.label_col
            else [1 if len(pos) > 0 else 0 for pos in df[self.tgt_cols_list[0]].to_list()]
        )
        tokens, labels = get_tokens_and_labels(pos_list_list=pos_list_list, text_list=X, labels=labels)
        tokenized_inputs = tokenizer(
            tokens, padding="max_length", is_split_into_words=True, truncation=True, max_length=MAX_LENGTH
        )
        new_labels = []
        for i, labels in enumerate(labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(align_labels_with_tokens(labels, word_ids))
        Y = np.array(new_labels)
        # This should in most cases not alter self.num_classes, as we only use binary tags (+ tag-continue = 3 classes).
        # However, we leave this generic implementation for future tasks.
        self.num_classes = len(np.unique(Y)) - 1
        X = tokenized_inputs.get("input_ids")
        attention_masks = tokenized_inputs.get("attention_mask")
        return torch.LongTensor(X), torch.LongTensor(Y), torch.LongTensor(attention_masks)

    def __repr__(self):
        """Represent a Token-level classification Subtask."""
        return "Token-level classification"

    def create_class_weights(self):
        """Compute the weights."""
        labels = self.Y[Split.TRAIN]
        only_class_labels = labels[labels != -100]
        self.class_weights = get_class_weights(only_class_labels, method="isns")

    def get_scaling_weight(self):
        """Get the weight of a POS Subtask.

        As with the other tasks, we normalize by the natural logarithm of the domain size.
        In case of POS subtask, the domain size equals the vocab size.
        """
        return 1 / np.log(self.num_classes)


class MLMSubTask(SubTask):
    """A Masked Language Modelling Subtask."""

    def __init__(self, *args, **kwargs):
        """Initialize a MLMSubTask."""
        super(MLMSubTask, self).__init__(*args, **kwargs)

    def load_data(self) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """Load the data of a MLMSubTask."""
        df = pd.read_csv(self.filename)
        X = df[self.src_col]
        tokenized_inputs = tokenizer(X.to_list(), padding="max_length", truncation=True, max_length=MAX_LENGTH)
        X = torch.LongTensor(tokenized_inputs.get("input_ids"))
        attention_masks = tokenized_inputs.get("attention_mask")

        MASK_TOKEN = tokenizer.mask_token_id
        SEP_TOKEN = tokenizer.sep_token_id
        CLS_TOKEN = tokenizer.cls_token_id
        PAD_TOKEN = tokenizer.pad_token_id

        Y = X.clone()
        rand = torch.rand(X.shape)
        masking_mask = (rand < 0.15) * (X != SEP_TOKEN) * (X != CLS_TOKEN) * (X != PAD_TOKEN)
        X[masking_mask] = MASK_TOKEN
        Y[~masking_mask] = -100

        return torch.LongTensor(X), torch.LongTensor(Y), torch.LongTensor(attention_masks)

    def __repr__(self):
        """Represent a MLM Subtask."""
        return "Masked Language Modelling"

    def get_scaling_weight(self):
        """Get the weights for imbalanced classes."""
        return 1 / np.log(len(tokenizer))
