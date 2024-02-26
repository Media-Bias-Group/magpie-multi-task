"""Script for executing the experiment 1. Run co-training of all families."""
import wandb


from config import head_specific_lr, head_specific_max_epoch, head_specific_patience
from enums.aggregation_method import AggregationMethod
from enums.model_checkpoints import ModelCheckpoint
from enums.scaling import LossScaling
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
from training.data import st_1_hatespeech_twitter_88 as hatespeech1
from training.data import st_1_online_harassment_dataset_87 as harasment1
from training.data import st_1_multidimnews_19 as multidim1
from training.data import st_2_crowS_pairs_33 as crows2
from training.data import st_1_imdb_101 as imdb1
from training.data import st_1_mdgender_116 as mdgender1
from training.data import st_1_funpedia_117 as fundpedia1
from training.data import st_1_sst2_99 as sst1
from training.data import st_1_liar_72 as liar1
from training.data import st_1_crowS_pairs_33 as crows1
from training.data import st_1_vaccine_lies_127 as vacc1
from training.data import st_2_stereotype_109 as stereotype2
from training.data import st_1_debate_effects_80 as debate1
from training.data import st_1_multitargetstance_125 as multistance1
from training.data import st_1_jigsaw_40 as jigsaw1
from training.data import st_1_wikimadlibs_91 as madlibs1
from training.data import st_2_hateXplain_92 as hatexplain2
from training.data import st_1_me_too_ma_108 as metoo1
from training.data import st_1_rtgender_105 as rtgender1
from training.data import st_2_redditbias_75 as redditbias2
from training.data import st_1_wikidetox_aggression_and_attack_892 as wikidetoxagg1
from training.data import st_1_gap_18 as gap1
from training.data import st_1_WTWT_126 as wtwt1
from training.data import st_2_wikidetox_aggression_and_attack_892 as wikidetoxagg2
from training.data import st_1_offensive_language_86 as offensivelang1
from training.data import st_1_wizards_of_wikipedia_118 as wizards1
from training.data import st_3_hateXplain_92 as hatexplain3
from training.data import st_2_basil_09 as basil2
from training.data import st_1_amazon_reviews_100 as amazon1
from training.data import st_1_newsWCL50_22 as newswcl1
from training.data import st_1_wikidetox_toxicity_891 as wikidetoxtox1
from training.data import st_1_semeval2014_63 as semeval63
from training.data import st_1_semeval2023_task4_119 as semeval4119
from training.data import st_1_redditbias_75 as redditbias1
from training.data import st_1_starbucks_38 as starbucks1
from training.data import st_2_babe_10 as babe2
from training.data import st_1_emotion_tweets_84 as emotions1
from training.data import st_3_crowS_pairs_33 as crows3
from training.data import st_3_redditbias_75 as redditbias3
from training.data import st_1_good_news_everyone_42 as goodnews1
from training.data import st_2_good_news_everyone_42 as goodnews2
from training.data import st_1_neutralizing_bias_26 as wnc
from training.data import st_1_hateXplain_92 as hatexplain1


from training.data.task import Task
from training.model.helper_classes import EarlyStoppingMode, Logger
from training.trainer.trainer import Trainer
from utils import set_random_seed

EXPERIMENT_NAME = "experiment_5"
tasks = [
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
    Task(task_id=hatespeech1.id, subtasks_list=[hatespeech1]),
    Task(task_id=harasment1.id, subtasks_list=[harasment1]),
    Task(task_id=hatespeech1.id, subtasks_list=[hatespeech1]),
    Task(task_id=multidim1.id, subtasks_list=[multidim1]),
    Task(task_id=crows2.id, subtasks_list=[crows2]),
    Task(task_id=imdb1.id, subtasks_list=[imdb1]),
    Task(task_id=mdgender1.id, subtasks_list=[mdgender1]),
    Task(task_id=fundpedia1.id, subtasks_list=[fundpedia1]),
    Task(task_id=sst1.id, subtasks_list=[sst1]),
    Task(task_id=liar1.id, subtasks_list=[liar1]),
    Task(task_id=crows1.id, subtasks_list=[crows1]),
    Task(task_id=vacc1.id, subtasks_list=[vacc1]),
    Task(task_id=stereotype2.id, subtasks_list=[stereotype2]),
    Task(task_id=debate1.id, subtasks_list=[debate1]),
    Task(task_id=multistance1.id, subtasks_list=[multistance1]),
    Task(task_id=jigsaw1.id, subtasks_list=[jigsaw1]),
    Task(task_id=madlibs1.id, subtasks_list=[madlibs1]),
    Task(task_id=hatexplain2.id, subtasks_list=[hatexplain2]),
    Task(task_id=metoo1.id, subtasks_list=[metoo1]),
    Task(task_id=rtgender1.id, subtasks_list=[rtgender1]),
    Task(task_id=redditbias2.id, subtasks_list=[redditbias2]),
    Task(task_id=wikidetoxagg1.id, subtasks_list=[wikidetoxagg1]),
    Task(task_id=gap1.id, subtasks_list=[gap1]),
    Task(task_id=wtwt1.id, subtasks_list=[wtwt1]),
    Task(task_id=wikidetoxagg2.id, subtasks_list=[wikidetoxagg2]),
    Task(task_id=offensivelang1.id, subtasks_list=[offensivelang1]),
    Task(task_id=wizards1.id, subtasks_list=[wizards1]),
    Task(task_id=hatexplain3.id, subtasks_list=[hatexplain3]),
    Task(task_id=basil2.id, subtasks_list=[basil2]),
    Task(task_id=amazon1.id, subtasks_list=[amazon1]),
    Task(task_id=newswcl1.id, subtasks_list=[newswcl1]),
    Task(task_id=wikidetoxtox1.id, subtasks_list=[wikidetoxtox1]),
    Task(task_id=semeval63.id, subtasks_list=[semeval63]),
    Task(task_id=semeval4119.id, subtasks_list=[semeval4119]),
    Task(task_id=redditbias1.id, subtasks_list=[redditbias1]),
    Task(task_id=starbucks1.id, subtasks_list=[starbucks1]),
    Task(task_id=babe2.id, subtasks_list=[babe2]),
    Task(task_id=emotions1.id, subtasks_list=[emotions1]),
    Task(task_id=crows3.id, subtasks_list=[crows3]),
    Task(task_id=redditbias3.id, subtasks_list=[redditbias3]),
    Task(task_id=goodnews1.id, subtasks_list=[goodnews1]),
    Task(task_id=goodnews2.id, subtasks_list=[goodnews2]),
    Task(task_id=wnc.id, subtasks_list=[wnc]),
    Task(task_id=hatexplain1.id, subtasks_list=[hatexplain1])
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
    "aggregation_method": AggregationMethod.MEAN,
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
    wandb.init(project=EXPERIMENT_NAME, name="first " + str(i + 1) + " tasks")
    config["model_name"] = "first_" + str(i + 1) + "_tasks"
    trainer = Trainer(task_list=current_tasks, LM=ModelCheckpoint.ROBERTA, **config)
    trainer.fit()
    trainer.save_model()
    wandb.finish()
