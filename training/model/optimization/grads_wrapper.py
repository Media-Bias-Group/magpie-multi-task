"""This module contains a Wrapper used to get/ set gradients."""
from typing import Dict

from torch import nn

from utils import rsetattr


class GradsWrapper(nn.Module):
    """Abstract class for a GradsWrapper.

    This class must be extended and not instantiated.
    """

    def __init__(self, *args, **kwargs):
        """Raise RuntimeError if this class is instantiated."""
        if type(self) == GradsWrapper:
            raise RuntimeError("Abstract class <GradsWrapper> must not be instantiated.")
        super(GradsWrapper, self).__init__()

    def get_grads(self) -> Dict:
        """Get the gradients of the weights and biases of all trainable layers."""
        return {k: v.grad.clone() if v.grad is not None else None for k, v in dict(self.named_parameters()).items()}

    def set_grads(self, grads: Dict):
        """Set the gradients of the weights and biases of all trainable layers."""
        for k, v in grads.items():
            rsetattr(self, f"{k}.grad", v)
