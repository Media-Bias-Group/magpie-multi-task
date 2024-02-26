"""Script for executing the experiment 1. Run co-training of all families."""
import wandb
from config import head_specific_lr, head_specific_max_epoch, head_specific_patience
from enums.aggregation_method import AggregationMethod
from enums.model_checkpoints import ModelCheckpoint
from enums.scaling import LossScaling
from training.data import all_tasks
from training.model.helper_classes import EarlyStoppingMode, Logger
from training.trainer.trainer import Trainer
from utils import set_random_seed

EXPERIMENT_NAME = "mbib-mtl-pretraining"
tasks = all_tasks

for t in tasks:
    for st in t.subtasks_list:
        st.process()

config = {
    "sub_batch_size": 32,
    "eval_batch_size": 256,
    "initial_lr": 4e-5,
    "dropout_prob": 0.1,
    "hidden_dimension": 768,
    "input_dimension": 768,
    "aggregation_method": AggregationMethod.MEAN,
    "early_stopping_mode": EarlyStoppingMode.HEADS,
    "loss_scaling": LossScaling.STATIC,
    "num_warmup_steps": 10,
    "pretrained_path": None,
    "resurrection": False,
    "model_name": "all_mtl",
    "head_specific_lr_dict": head_specific_lr,
    "head_specific_patience_dict": head_specific_patience,
    "head_specific_max_epoch_dict": head_specific_max_epoch,
    "logger": Logger(EXPERIMENT_NAME)
}

set_random_seed()
wandb.init(project=EXPERIMENT_NAME, name="all-mtl")
trainer = Trainer(task_list=tasks, LM=ModelCheckpoint.ROBERTA, **config)
trainer.fit()
trainer.save_model()
wandb.finish()