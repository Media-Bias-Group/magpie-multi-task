"""Script for executing the experiment 1. Run co-training of all families."""
import itertools

import wandb
from config import head_specific_lr, head_specific_max_epoch, head_specific_patience
from enums.aggregation_method import AggregationMethod
from enums.model_checkpoints import ModelCheckpoint
from enums.scaling import LossScaling
from training.data import (
    emotionality,
    fake_news,
    gender_bias,
    group_bias,
    hate_speech,
    media_bias,
    sentiment_analysis,
    stance_detection,
    subjective_bias,
)
from training.model.helper_classes import EarlyStoppingMode, Logger
from training.trainer.trainer import Trainer
from utils import set_random_seed

EXPERIMENT_NAME = "experiment_1"

all_families = {
    "media_bias": media_bias,
    "subjective_bias": subjective_bias,
    "hate_speech": hate_speech,
    "gender_bias": gender_bias,
    "sentiment_analysis": sentiment_analysis,
    "fake_news": fake_news,
    "group_bias": group_bias,
    "emotionality": emotionality,
    "stance_detection": stance_detection,
}

if __name__ == "__main__":
    set_random_seed()
    our_config = {
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
        "model_name": "experiment_0_model",
        "head_specific_lr_dict": head_specific_lr,
        "head_specific_patience_dict": head_specific_patience,
        "head_specific_max_epoch_dict": head_specific_max_epoch,
        "logger": Logger("experiment_1"),
    }

    # first, for each family, train this family with all tasks of said family
    for family_name, task_list in all_families.items():  # family is a list of tasks
        for t in task_list:
            for st in t.subtasks_list:
                st.process()
        wandb.init(project=EXPERIMENT_NAME, name=family_name)
        trainer = Trainer(task_list=task_list, LM=ModelCheckpoint.ROBERTA, **our_config)
        trainer.fit()
        wandb.finish()

    # now, after we have trained each family independently, we want to perform co-training
    for family_1, family_2 in list(itertools.combinations(all_families.keys(), 2)):
        task_list = all_families[family_1] + all_families[family_2]
        wandb.init(project=EXPERIMENT_NAME, name=f"co-training-{family_1}-{family_2}")
        trainer = Trainer(task_list=task_list, LM=ModelCheckpoint.ROBERTA, **our_config)
        trainer.fit()
        wandb.finish()
