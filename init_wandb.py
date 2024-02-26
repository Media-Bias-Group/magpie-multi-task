"""Initialize Weights & Biases."""

import wandb
from config import WANDB_API_KEY

wandb.login(key=WANDB_API_KEY)
