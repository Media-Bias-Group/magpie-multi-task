"""Script for executing the experiment 1. Run co-training of all families."""
import wandb
from config import head_specific_lr, head_specific_max_epoch, head_specific_patience
from enums.aggregation_method import AggregationMethod
from enums.model_checkpoints import ModelCheckpoint
from enums.scaling import LossScaling
from enums.splits import Split
from training.data import st_1_babe_10 as babe
from training.data.task import Task
from training.model.helper_classes import EarlyStoppingMode, Logger
from training.trainer.trainer import Trainer
from utils import set_random_seed

EXPERIMENT_NAME = "babe_gradts_all"
babe = [Task(task_id=babe.id, subtasks_list=[babe])]

for t in babe:
    for st in t.subtasks_list:
        st.process()

config = {
    "sub_batch_size": 32,
    "eval_batch_size": 128,
    "initial_lr": 4e-5,
    "dropout_prob": 0.1,
    "hidden_dimension": 768,
    "input_dimension": 768,
    "aggregation_method": AggregationMethod.PCGRAD,
    "early_stopping_mode": EarlyStoppingMode.HEADS,
    "loss_scaling": LossScaling.STATIC,
    "num_warmup_steps": 10,
    "pretrained_path": None,
    "resurrection": True,
    "model_name": "experiment_2_model",
    "head_specific_lr_dict": head_specific_lr,
    "head_specific_patience_dict": head_specific_patience,
    "head_specific_max_epoch_dict": head_specific_max_epoch,
    "logger": Logger(EXPERIMENT_NAME),
}

for i in range(59):
    set_random_seed()
    wandb.init(project=EXPERIMENT_NAME, name=str(i + 1) + "-finetuning")
    config["pretrained_path"] = "model_files/first_" + str(i+1) + "_tasks.pth"
    trainer = Trainer(task_list=babe, LM=ModelCheckpoint.ROBERTA, **config)
    trainer.fit()
    trainer.eval(split=Split.TEST)
    wandb.finish()
