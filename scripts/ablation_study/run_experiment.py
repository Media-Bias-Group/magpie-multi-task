"""Script for executing the experiment 0. Run grid-search on different optimization techniques."""

import random

import wandb
from config import head_specific_lr, head_specific_max_epoch, head_specific_patience
from enums.aggregation_method import AggregationMethod
from enums.model_checkpoints import ModelCheckpoint
from enums.scaling import LossScaling
from training.data import all_tasks, babe_10
from training.model.helper_classes import EarlyStoppingMode, Logger
from training.trainer.trainer import Trainer
from utils import set_random_seed

hyper_param_dict = {
    "grad_agg": {"values": ["pcgrad","mean","pcgrad_online"]},
    "scaling": {"values": ["static","uniform"]},
    "resurrection": {"values": [True, False]},
    "es": {"values": ["backbone","heads"]},
}


def train_wrapper():
    """Execute the wandb hyperparameter tuning job.

    Takes the (globally defined) tasks, instantiates a trainer for them.
    This function is passed as a callback to wandb.
    """
    wandb.init(project="experiment_0")
    set_random_seed()

    our_config = {
        "sub_batch_size": 32,
        "eval_batch_size": 128,
        "initial_lr": 4e-5,
        "dropout_prob": 0.1,
        "hidden_dimension": 768,
        "input_dimension": 768,
        "aggregation_method": AggregationMethod(wandb.config.grad_agg),
        "early_stopping_mode": EarlyStoppingMode(wandb.config.es),
        "loss_scaling": LossScaling(wandb.config.scaling),
        "num_warmup_steps": 10,
        "pretrained_path": None,
        "resurrection": wandb.config.resurrection,
        "model_name": "experiment_0_model",
        "head_specific_lr_dict": head_specific_lr,
        "head_specific_patience_dict": head_specific_patience,
        "head_specific_max_epoch_dict": head_specific_max_epoch,
        "logger": Logger("experiment_0"),
    }

    trainer = Trainer(task_list=task_list, LM=ModelCheckpoint.ROBERTA, **our_config)
    trainer.fit()


if __name__ == "__main__":
    # wandb sweep
    tasks = all_tasks
    random.shuffle(tasks)
    task_list = tasks[:15]
    if babe_10 not in task_list:
        task_list.append(babe_10)


    for t in task_list:
        for st in t.subtasks_list:
            st.process()

    sweep_config = {"method": "grid"}
    sweep_config["parameters"] = hyper_param_dict
    sweep_config["name"] = "random-set"
    sweep_id = wandb.sweep(sweep_config, project="experiment_0")

    wandb.agent(sweep_id, train_wrapper)
    wandb.finish()
