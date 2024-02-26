"""This module contains the implementation of the heads for specific tasks as well a factory-method for deciding which head to use."""

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torchmetrics import Accuracy, F1Score, MeanSquaredError, Perplexity, R2Score

from training.data.subtask import (
    ClassificationSubTask,
    MLMSubTask,
    MultiLabelClassificationSubTask,
    POSSubTask,
    RegressionSubTask,
    SubTask,
)
from training.model.GradsWrapper import GradsWrapper
from training.tokenizer import tokenizer


def HeadFactory(st: SubTask, *args, **kwargs):
    """Decide which head to use for the specific task type
       st: subtask"""
    if isinstance(st, ClassificationSubTask):
        return ClassificationHead(num_classes=st.num_classes, class_weights=st.class_weights, *args, **kwargs)
    elif isinstance(st, MultiLabelClassificationSubTask):
        return ClassificationHead(
            num_classes=st.num_classes, num_labels=st.num_labels, class_weights=st.class_weights, *args, **kwargs)
    elif isinstance(st, POSSubTask):
        return TokenClassificationHead(num_classes=st.num_classes, class_weights=st.class_weights, *args, **kwargs)
    elif isinstance(st, RegressionSubTask):
        return RegressionHead(*args, **kwargs)
    elif isinstance(st, MLMSubTask):
        return LanguageModellingHead(*args, **kwargs)


class ClassificationHead(GradsWrapper):
    """Classifier inspired by one used in RoBERTa."""

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        dropout_prob: float,
        num_classes=2,
        num_labels=1,
        class_weights=None,
    ):
        """Initialize the head."""
        super().__init__()
        self.dense = nn.Linear(input_dimension, hidden_dimension)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.out_proj = nn.Linear(hidden_dimension, num_classes * num_labels)
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.loss = CrossEntropyLoss(weight=class_weights)
        self.metrics = {
            "f1": F1Score(num_classes=num_classes, mdmc_reduce="global", average="macro"),
            "acc": Accuracy(mdmc_reduce="global"),
        }

    def forward(self, X, y):
        """Feed the data through head accordingly to RoBERTa approach and compute loss."""
        batch_size = y.shape[0]  # size of data in this subbatch

        x = X[:, 0, :]  # take <s> token (equiv. to [CLS])

        # pass CLS through classifier
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.out_proj(x)

        loss = self.loss(logits.view(-1, self.num_classes), y.view(-1))
        logits = logits.view(batch_size, self.num_classes, self.num_labels)  # reshape logits into prediction
        metrics_values = {k: metric(logits.cpu(), y.cpu()) for k, metric in self.metrics.items()}
        return logits, loss, metrics_values


class TokenClassificationHead(GradsWrapper):
    """TokenClassificationHead inspired by one used in RoBERTa."""

    def __init__(self, num_classes: int, class_weights, hidden_dimension: int, dropout_prob: float, *args, **kwargs):
        """Initialize the TokenClassificationHead."""
        super(TokenClassificationHead, self).__init__(*args, **kwargs)
        self.dropout_LM = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(hidden_dimension, num_classes)
        self.loss = CrossEntropyLoss(weight=class_weights)
        self.num_classes = num_classes
        self.metrics = {
            "f1": F1Score(num_classes=num_classes, mdmc_reduce="global", average="macro"),
            "acc": Accuracy(mdmc_reduce="global"),
        }

    def forward(self, X, y):
        """Feed the data through head accordingly to RoBERTa approach and compute loss."""
        sequence_output = self.dropout_LM(X)
        logits = self.classifier(sequence_output)
        loss = self.loss(logits.view(-1, self.num_classes), y.view(-1))

        # Ignore class -100 when computing metrics
        mask = torch.where(y != -100, 1, 0)
        logits = torch.masked_select(logits, (mask.unsqueeze(-1).expand(logits.size()) == 1))

        y = torch.masked_select(y, (mask == 1))
        logits = logits.view(y.shape[0], self.num_classes)
        metrics_values = {k: metric(logits.cpu(), y.cpu()) for k, metric in self.metrics.items()}

        return logits, loss, metrics_values


class RegressionHead(GradsWrapper):
    """Regression head inspired by one used in RoBERTa."""

    def __init__(self, input_dimension: int, hidden_dimension: int, dropout_prob: float):
        """Initialize the RegressionHead."""
        super().__init__()
        self.dense = nn.Linear(input_dimension, hidden_dimension)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.out_proj = nn.Linear(hidden_dimension, 1)
        self.loss = MSELoss()
        self.metrics = {"R2": R2Score(), "MSE": MeanSquaredError()}  # Needs at least 2 samples

    def forward(self, X, y):
        """Feed the data through head accordingly to RoBERTa approach and compute loss."""
        x = X[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.out_proj(x)

        loss = self.loss(logits.squeeze(), y.squeeze())

        metrics_values = {k: metric(logits.cpu(), y.cpu()).detach() for k, metric in self.metrics.items()}

        return logits, loss, metrics_values


class LanguageModellingHead(GradsWrapper):
    """Roberta Head for masked language modeling."""

    def __init__(self, input_dimension: int, hidden_dimension: int, dropout_prob: float):
        """Initialize LM head."""
        super().__init__()
        self.dense = nn.Linear(input_dimension, hidden_dimension)
        self.layer_norm = nn.LayerNorm(hidden_dimension, eps=1e-5)
        self.gelu = torch.nn.GELU()
        self.loss = CrossEntropyLoss()

        # output dimension is of size of all possible tokens
        self.decoder = nn.Linear(hidden_dimension, tokenizer.vocab_size)
        self.bias = nn.Parameter(torch.zeros(tokenizer.vocab_size))
        self.decoder.bias = self.bias
        self.metrics = {"perplexity": Perplexity()}

    def forward(self, X, y):
        """Feed the data through one layer and then project to vocab size."""
        x = self.dense(X)
        x = self.gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary
        logits = self.decoder(x)
        loss = self.loss(logits.view(-1, tokenizer.vocab_size), y.view(-1))

        metrics_values = {k: metric(logits.cpu(), y.cpu()) for k, metric in self.metrics.items()}

        return logits, loss, metrics_values
