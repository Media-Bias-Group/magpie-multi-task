"""Script for executing the experiment 1. Run co-training of all families."""
import wandb
from config import head_specific_lr, head_specific_max_epoch, head_specific_patience
from enums.aggregation_method import AggregationMethod
from enums.model_checkpoints import ModelCheckpoint
from enums.scaling import LossScaling
from training.data import st_1_babe_10 as babe
from training.data import st_1_basil_09 as basil
from training.data import st_1_bunemo_96 as bunemo
from training.data import st_1_cw_hard_03 as cw_hard
from training.data import st_1_fake_news_net_25 as fake_news
from training.data import st_1_gwsd_128 as gwsd
from training.data import st_1_mpqa_103 as mpqa
from training.data import st_1_pheme_12 as pheme1
from training.data import st_1_semeval2016_task6_124 as semeval2
from training.data import st_1_semeval2023_task3_120 as semeval
from training.data import st_1_stereoset_64 as stereoset
from training.data import st_1_stereotype_109 as stereotype
from training.data import st_1_subj_31 as subj
from training.data import st_1_trac2_104 as trac2
from training.data import st_2_pheme_12 as pheme2
from training.data import st_2_stereoset_64 as stereoset2
from training.data.task import Task
from training.model.helper_classes import EarlyStoppingMode, Logger
from training.trainer.trainer import Trainer
from utils import set_random_seed

EXPERIMENT_NAME = "experiment_2"
tasks = [
    Task(task_id=babe.id, subtasks_list=[babe]),
    Task(task_id=semeval.id, subtasks_list=[semeval]),
    Task(task_id=basil.id, subtasks_list=[basil]),
    Task(task_id=pheme1.id, subtasks_list=[pheme1]),
    Task(task_id=mpqa.id, subtasks_list=[mpqa]),
    Task(task_id=gwsd.id, subtasks_list=[gwsd]),
    Task(task_id=subj.id, subtasks_list=[subj]),
    Task(task_id=cw_hard.id, subtasks_list=[cw_hard]),
    Task(task_id=pheme2.id, subtasks_list=[pheme2]),
    Task(task_id=trac2.id, subtasks_list=[trac2]),
    Task(task_id=fake_news.id, subtasks_list=[fake_news]),
    Task(task_id=bunemo.id, subtasks_list=[bunemo]),
    Task(task_id=stereoset.id, subtasks_list=[stereoset]),
    Task(task_id=stereotype.id, subtasks_list=[stereotype]),
    Task(task_id=semeval2.id, subtasks_list=[semeval2]),
    Task(task_id=stereoset2.id, subtasks_list=[stereoset2]),
]

for t in tasks:
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

for i in range(len(tasks)):
    set_random_seed()
    current_tasks = tasks[: i + 1]
    wandb.init(project=EXPERIMENT_NAME, name="pre-training-first " + str(i + 1) + " tasks")
    trainer = Trainer(task_list=current_tasks, LM=ModelCheckpoint.ROBERTA, **config)
    trainer.fit()
    wandb.finish()
