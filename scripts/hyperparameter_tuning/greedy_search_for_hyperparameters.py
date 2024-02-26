"""This module can be executed to run a hyperparameter training job for multiple tasks, each task alone."""

import wandb
from config import hyper_param_dict
from enums.aggregation_method import AggregationMethod
from enums.model_checkpoints import ModelCheckpoint
from training.data import all_subtasks
from training.data.task import Task
from training.model.helper_classes import EarlyStoppingMode, Logger
from training.trainer.trainer import Trainer
from utils import set_random_seed
from training.data import all_subtasks
import training.data


def train_wrapper():
    """Execute the wandb hyperparameter tuning job.

    Takes the (globally defined) tasks, instantiates a trainer for them.
    This function is passed as a callback to wandb.
    """
    wandb.init(project="hyperparam-tuning")
    set_random_seed()
    
    our_config = {
        "sub_batch_size": 32,
        "eval_batch_size": 128,
        "initial_lr": 5e-5,
        "dropout_prob": 0.1,
        "hidden_dimension": 768,
        "input_dimension": 768,
        "early_stopping_mode": EarlyStoppingMode("heads"),
        "aggregation_method": AggregationMethod.MEAN,
        "model_name": None,
        "pretrained_path": None,
        "resurrection":False,
        "num_warmup_steps": 10,
        "head_specific_max_epoch_dict": {f"{st.id}": wandb.config.max_epoch},
        "head_specific_lr_dict": {f"{st.id}": wandb.config.lr},
        "head_specific_patience_dict": {f"{st.id}": wandb.config.patience},
        "logger":Logger("hyperparam-tuning")
    }

    trainer = Trainer(task_list=task_wrapper, LM=ModelCheckpoint.ROBERTA, **our_config)
    trainer.fit()


if __name__ == "__main__":
    for st in [training.data.st_1_gwsd_128]:
        st.process()
        task_wrapper = [Task(task_id=st.id, subtasks_list=[st])]

        # wandb sweep
        sweep_config = {"method": "grid"}
        sweep_config["parameters"] = hyper_param_dict
        sweep_config["name"] = str(st)
        sweep_id = wandb.sweep(sweep_config, project="hyperparam-tuning")

        wandb.agent(sweep_id, train_wrapper)
        wandb.finish()