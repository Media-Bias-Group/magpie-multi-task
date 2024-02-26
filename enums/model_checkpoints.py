"""Enum type for pre-trained model checkpoints we use."""

from enum import Enum


class ModelCheckpoint(Enum):
    """Model checkpoints from huggingface."""

    BART = "facebook/bart-base"
    ROBERTA = "roberta-base"
    MUPPET = "facebook/muppet-roberta-base"
    ROBERTA_DUMMY = "roberta-dummy"
