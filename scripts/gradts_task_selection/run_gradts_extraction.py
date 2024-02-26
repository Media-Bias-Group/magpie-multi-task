"""Script for executing the experiment 1. Run co-training of all families."""
import wandb
from config import head_specific_lr
from enums.model_checkpoints import ModelCheckpoint
from training.data import all_subtasks
from training.data.task import Task
from training.experiments.GradTSTrainer import GradTSTrainer
from training.model.helper_classes import Logger
from utils import set_random_seed

EXPERIMENT_NAME = "experiment_2"


if __name__ == "__main__":
    set_random_seed()

    config = {
        "sub_batch_size": 32,
        "eval_batch_size": 128,
        "initial_lr": 4e-5,
        "dropout_prob": 0.1,
        "hidden_dimension": 768,
        "input_dimension": 768,
        "num_warmup_steps": 10,
        "pretrained_path": None,
        "head_specific_lr_dict": head_specific_lr,
        "logger": Logger(EXPERIMENT_NAME),
    }

for st in all_subtasks:
    st.process()
    task_wrapper = [Task(task_id=st.id, subtasks_list=[st])]

    wandb.init(project=EXPERIMENT_NAME, name=str(st.id))
    trainer = GradTSTrainer(
        task_list=task_wrapper,
        LM=ModelCheckpoint.ROBERTA,
        number_of_epochs=3,
        model_name=str(st.id) + "_three_epochs",
        st_id=str(st.id),
        **config
    )
    trainer.fit()
    wandb.finish()
