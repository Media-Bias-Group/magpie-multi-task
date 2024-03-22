"""Initialize the Subtasks."""


import itertools

from training.data.subtask import (
    ClassificationSubTask,
    SoftClassificationSubTask,
    MLMSubTask,
    MultiLabelClassificationSubTask,
    POSSubTask,
    RegressionSubTask,
)
from training.data.task import Task

# subtasks
st_1_cc_news_mlm = MLMSubTask(
    tgt_cols_list=[""],
    task_id=0,
    filename="00_MLM-CC-News/preprocessed.csv",
    id=0,
)

st_1_multidimnews_19 = MultiLabelClassificationSubTask(
    tgt_cols_list=["label_bias", "label_subj", "label_framing", "label_hidden_assumpt"],
    task_id=19,
    num_classes=2,
    num_labels=4,
    filename="19_MultiDimNews/preprocessed.csv",
    id=19001,
)
st_1_subj_31 = ClassificationSubTask(task_id=31, num_classes=2, filename="31_SUBJ/preprocessed.csv", id=31001)
st_1_basil_09 = ClassificationSubTask(task_id=9, num_classes=3, filename="9_BASIL/preprocessed.csv", id=90001)
st_2_basil_09 = POSSubTask(task_id=9, tgt_cols_list=["pos"], filename="9_BASIL/preprocessed.csv", id=90002)
st_1_neutralizing_bias_26 = POSSubTask(
    tgt_cols_list=["pos"],
    task_id=26,
    filename="26_neutralizing-bias/preprocessed.csv",
    id=26001,
)
st_1_newsWCL50_22 = RegressionSubTask(task_id=22, filename="22_NewsWCL50/preprocessed.csv", id=22001)
st_1_starbucks_38 = RegressionSubTask(task_id=38, filename="38_starbucks/preprocessed.csv", id=38001)
st_1_cw_hard_03 = ClassificationSubTask(task_id=3, filename="03_CW_HARD/preprocessed.csv", id=300001)
st_1_offensive_language_86 = ClassificationSubTask(
    task_id=86, filename="86_OffensiveLanguage/preprocessed.csv", id=86001, num_classes=3
)
st_1_online_harassment_dataset_87 = ClassificationSubTask(
    task_id=87, filename="87_OnlineHarassmentDataset/preprocessed.csv", id=87001
)
st_1_wikidetox_aggression_and_attack_892 = RegressionSubTask(
    task_id=892,
    tgt_cols_list=["label_aggression"],
    filename="892_WikiDetoxAggressionAndAttack/preprocessed.csv",
    id=89201,
)
st_2_wikidetox_aggression_and_attack_892 = ClassificationSubTask(
    task_id=892,
    tgt_cols_list=["label_attack"],
    filename="892_WikiDetoxAggressionAndAttack/preprocessed.csv",
    id=89202,
)
st_1_wikidetox_toxicity_891 = RegressionSubTask(
    task_id=891, filename="891_WikiDetoxToxicity/preprocessed.csv", id=89101
)
st_1_jigsaw_40 = ClassificationSubTask(task_id=40, filename="40_JIGSAW/preprocessed.csv", id=40001)
st_1_me_too_ma_108 = MultiLabelClassificationSubTask(
    num_classes=2,
    num_labels=2,
    task_id=108,
    filename="108_MeTooMA/preprocessed.csv",
    id=10801,
    tgt_cols_list=["hate_speech_label", "sarcasm_label"],
)
st_1_wikimadlibs_91 = ClassificationSubTask(task_id=91, filename="91_WikiMadlibs/preprocessed.csv", id=91001)
st_1_hateXplain_92 = POSSubTask(
    task_id=92, tgt_cols_list=["rationale_pos"], id=92001, filename="92_HateXplain/preprocessed.csv"
)
st_2_hateXplain_92 = ClassificationSubTask(
    task_id=92, id=92002, filename="92_HateXplain/preprocessed.csv", num_classes=3
)
st_3_hateXplain_92 = MultiLabelClassificationSubTask(
    task_id=92,
    id=92003,
    filename="92_HateXplain/preprocessed.csv",
    num_labels=5,
    num_classes=2,
    tgt_cols_list=["label_race", "label_religion", "label_gender", "label_economic", "label_minority"],
)
st_1_hatespeech_twitter_88 = ClassificationSubTask(
    task_id=88, id=88001, filename="88_HatespeechTwitter/preprocessed.csv", num_classes=4
)
st_1_gap_18 = ClassificationSubTask(task_id=18, id=18001, filename="18_GAP/preprocessed.csv", num_classes=3)
st_1_rtgender_105 = ClassificationSubTask(task_id=105, id=10501, filename="105_RtGender/preprocessed.csv")
st_1_mdgender_116 = ClassificationSubTask(
    task_id=116, id=11601, filename="116_MDGender/preprocessed.csv", num_classes=6
)
st_1_trac2_104 = ClassificationSubTask(task_id=104, id=10401, filename="104_TRAC2/preprocessed.csv")
st_1_funpedia_117 = ClassificationSubTask(
    task_id=117, num_classes=3, id=11701, filename="117_Funpedia/preprocessed.csv"
)
st_1_wizards_of_wikipedia_118 = ClassificationSubTask(
    task_id=118, num_classes=3, id=11801, filename="118_WizardsOfWikipedia/preprocessed.csv"
)
st_1_sst2_99 = ClassificationSubTask(task_id=99, id=99001, filename="99_SST2/preprocessed.csv")
st_1_imdb_101 = ClassificationSubTask(task_id=101, id=10101, filename="101_IMDB/preprocessed.csv")
st_1_mpqa_103 = ClassificationSubTask(task_id=103, id=10301, filename="103_MPQA/preprocessed.csv")
st_1_semeval2014_63 = POSSubTask(
    id=63001, task_id=63, filename="63_semeval2014/preprocessed.csv", tgt_cols_list=["pos"], label_col="label"
)
st_1_amazon_reviews_100 = ClassificationSubTask(task_id=100, id=100001, filename="100_Amazon_reviews/preprocessed.csv")
st_1_liar_72 = RegressionSubTask(task_id=72, id=72001, filename="72_LIAR/preprocessed.csv")
st_1_fake_news_net_25 = ClassificationSubTask(task_id=25, id=25001, filename="25_FakeNewsNet/preprocessed.csv")
st_1_pheme_12 = ClassificationSubTask(task_id=12, id=12001, filename="12_PHEME/preprocessed.csv")
st_2_pheme_12 = ClassificationSubTask(
    task_id=12,
    id=12002,
    filename="12_PHEME/preprocessed.csv",
    tgt_cols_list=["veracity_label"],
    num_classes=3,
)
st_1_crowS_pairs_33 = ClassificationSubTask(task_id=33, id=33001, filename="33_CrowSPairs/preprocessed.csv")
st_2_crowS_pairs_33 = ClassificationSubTask(
    task_id=33,
    id=33002,
    filename="33_CrowSPairs/preprocessed.csv",
    tgt_cols_list=["stereotype_label"],
    num_classes=9,
)
st_3_crowS_pairs_33 = POSSubTask(
    task_id=33, id=33003, filename="33_CrowSPairs/preprocessed.csv", tgt_cols_list=["pos"]
)
st_1_stereoset_64 = ClassificationSubTask(task_id=64, id=64001, filename="64_StereoSet/preprocessed.csv")
st_2_stereoset_64 = ClassificationSubTask(
    task_id=64,
    id=64002,
    filename="64_StereoSet/preprocessed.csv",
    num_classes=4,
    tgt_cols_list=["stereotype_label"],
)
st_1_stereotype_109 = ClassificationSubTask(task_id=109, id=10901, filename="109_stereotype/preprocessed.csv")
st_2_stereotype_109 = MultiLabelClassificationSubTask(
    task_id=109,
    id=10902,
    filename="109_stereotype/preprocessed.csv",
    tgt_cols_list=["stereotype_explicit_label", "stereotype_explicit_label"],
    num_classes=2,
    num_labels=2,
)
st_1_redditbias_75 = ClassificationSubTask(task_id=75, id=75001, filename="75_RedditBias/preprocessed.csv")
st_2_redditbias_75 = ClassificationSubTask(
    num_classes=5,
    task_id=75,
    id=75002,
    filename="75_RedditBias/preprocessed.csv",
    tgt_cols_list=["label_group"],
)
st_3_redditbias_75 = POSSubTask(
    tgt_cols_list=["bias_pos"], task_id=75, id=75003, filename="75_RedditBias/preprocessed.csv"
)
st_1_good_news_everyone_42 = POSSubTask(
    tgt_cols_list=["cue_pos"], task_id=42, id=42001, filename="42_GoodNewsEveryone/preprocessed.csv"
)
st_2_good_news_everyone_42 = POSSubTask(
    tgt_cols_list=["experiencer_pos"],
    task_id=42,
    id=42002,
    filename="42_GoodNewsEveryone/preprocessed.csv",
)
st_1_bunemo_96 = ClassificationSubTask(task_id=96, id=96001, filename="96_Bu-NEMO/preprocessed.csv", num_classes=8)
st_1_emotion_tweets_84 = ClassificationSubTask(
    task_id=84, id=84001, filename="84_emotion_tweets/preprocessed.csv", num_classes=8
)
st_1_debate_effects_80 = RegressionSubTask(task_id=80, id=80001, filename="80_DebateEffects/preprocessed.csv")
st_1_semeval2023_task4_119 = ClassificationSubTask(
    id=11901, task_id=119, filename="119_SemEval2023Task4/preprocessed.csv"
)
st_1_semeval2023_task3_120 = ClassificationSubTask(
    task_id=120, id=120001, filename="120_SemEval2023Task3/preprocessed.csv"
)
st_1_vaccine_lies_127 = ClassificationSubTask(
    task_id=127, id=12701, filename="127_VaccineLies/preprocessed.csv", num_classes=4
)
st_1_semeval2016_task6_124 = ClassificationSubTask(
    task_id=124, id=12401, filename="124_SemEval2016Task6/preprocessed.csv", num_classes=3
)
st_1_WTWT_126 = ClassificationSubTask(num_classes=3, task_id=126, id=12601, filename="126_WTWT/preprocessed.csv")
st_1_multitargetstance_125 = ClassificationSubTask(
    num_classes=3, task_id=125, id=12501, filename="125_MultiTargetStance/preprocessed.csv"
)

st_1_babe_10 = ClassificationSubTask(task_id=10, id=10001, filename="10_BABE/preprocessed.csv", num_classes=2)
st_2_babe_10 = POSSubTask(task_id=10, id=10002, filename="10_BABE/preprocessed.csv", tgt_cols_list=["biased_words"])
st_1_gwsd_128 = ClassificationSubTask(task_id=128, num_classes=3, filename="128_GWSD/preprocessed.csv", id=12801)


st_1_babe_sh_129 = ClassificationSubTask(task_id = 129, id=12901, filename = "129_BABE_SH/preprocessed.csv",num_classes=2)
st_2_babe_sh_129 = SoftClassificationSubTask(task_id = 129, id=12902, filename = "129_BABE_SH/preprocessed.csv",num_classes=2,tgt_cols_list=['soft_label'])

# Tasks

babe_sh_129 = Task(task_id= 129, subtasks_list=[st_1_babe_sh_129,st_2_babe_sh_129])
mlm_0 = Task(task_id=0, subtasks_list=[st_1_cc_news_mlm])
multidimnews_19 = Task(task_id=19, subtasks_list=[st_1_multidimnews_19])
subj_31 = Task(task_id=31, subtasks_list=[st_1_subj_31])
babe_10 = Task(task_id=10, subtasks_list=[st_1_babe_10, st_2_babe_10])
basil_09 = Task(task_id=9, subtasks_list=[st_1_basil_09, st_2_basil_09])
neutralizing_bias_26 = Task(task_id=26, subtasks_list=[st_1_neutralizing_bias_26])
newsWCL50_22 = Task(task_id=22, subtasks_list=[st_1_newsWCL50_22])
starbucks_38 = Task(task_id=38, subtasks_list=[st_1_starbucks_38])
cw_hard_03 = Task(task_id=3, subtasks_list=[st_1_cw_hard_03])
offensive_language_86 = Task(task_id=86, subtasks_list=[st_1_offensive_language_86])
online_harassment_dataset_87 = Task(task_id=87, subtasks_list=[st_1_online_harassment_dataset_87])
wikidetox_toxicity_891 = Task(task_id=891, subtasks_list=[st_1_wikidetox_toxicity_891])
wikidetox_aggression_and_attack_892 = Task(
    task_id=892, subtasks_list=[st_1_wikidetox_aggression_and_attack_892, st_2_wikidetox_aggression_and_attack_892]
)
jigsaw_40 = Task(task_id=40, subtasks_list=[st_1_jigsaw_40])
me_too_ma_108 = Task(task_id=108, subtasks_list=[st_1_me_too_ma_108])
wikimadlibs_91 = Task(task_id=91, subtasks_list=[st_1_wikimadlibs_91])
hateXplain_92 = Task(task_id=92, subtasks_list=[st_1_hateXplain_92, st_2_hateXplain_92, st_3_hateXplain_92])
hatespeech_twitter_88 = Task(task_id=88, subtasks_list=[st_1_hatespeech_twitter_88])
gap_18 = Task(task_id=18, subtasks_list=[st_1_gap_18])
rtgender_105 = Task(task_id=105, subtasks_list=[st_1_rtgender_105])
mdgender_116 = Task(task_id=116, subtasks_list=[st_1_mdgender_116])
trac2_104 = Task(task_id=104, subtasks_list=[st_1_trac2_104])
funpedia_117 = Task(task_id=117, subtasks_list=[st_1_funpedia_117])
wizards_of_wikipedia_118 = Task(task_id=118, subtasks_list=[st_1_wizards_of_wikipedia_118])
sst2_99 = Task(task_id=99, subtasks_list=[st_1_sst2_99])
imdb_101 = Task(task_id=101, subtasks_list=[st_1_imdb_101])
mpqa_103 = Task(task_id=103, subtasks_list=[st_1_mpqa_103])
semeval2014_63 = Task(task_id=63, subtasks_list=[st_1_semeval2014_63])
amazon_reviews_100 = Task(task_id=100, subtasks_list=[st_1_amazon_reviews_100])
liar_72 = Task(task_id=72, subtasks_list=[st_1_liar_72])
fake_news_net_25 = Task(task_id=25, subtasks_list=[st_1_fake_news_net_25])
pheme_12 = Task(task_id=12, subtasks_list=[st_2_pheme_12, st_1_pheme_12])
crowSpairs_33 = Task(task_id=33, subtasks_list=[st_1_crowS_pairs_33, st_2_crowS_pairs_33, st_3_crowS_pairs_33])
stereoset_64 = Task(task_id=64, subtasks_list=[st_1_stereoset_64, st_2_stereoset_64])
stereotype_109 = Task(task_id=109, subtasks_list=[st_1_stereotype_109, st_2_stereotype_109])
reddit_bias_75 = Task(task_id=75, subtasks_list=[st_1_redditbias_75, st_2_redditbias_75, st_3_redditbias_75])
good_news_everyone_42 = Task(task_id=42, subtasks_list=[st_1_good_news_everyone_42, st_2_good_news_everyone_42])
bunemo_96 = Task(task_id=96, subtasks_list=[st_1_bunemo_96])
emotion_tweets_84 = Task(task_id=84, subtasks_list=[st_1_emotion_tweets_84])
debate_effects_80 = Task(task_id=80, subtasks_list=[st_1_debate_effects_80])
semeval2023_task4_119 = Task(task_id=119, subtasks_list=[st_1_semeval2023_task4_119])
semeval2023_task_3_120 = Task(task_id=120, subtasks_list=[st_1_semeval2023_task3_120])
vaccine_lies_127 = Task(task_id=127, subtasks_list=[st_1_vaccine_lies_127])
semeval2016_task6_124 = Task(task_id=124, subtasks_list=[st_1_semeval2016_task6_124])
WTWT_126 = Task(task_id=126, subtasks_list=[st_1_WTWT_126])
multitargetstance_125 = Task(task_id=125, subtasks_list=[st_1_multitargetstance_125])
gwsd_128 = Task(task_id=128, subtasks_list=[st_1_gwsd_128])

# MBIB
st_linguistic = ClassificationSubTask(task_id=11111, id=11111, filename="mbib_linguistic/preprocessed.csv", num_classes=2)
mbib_lingustic = Task(task_id=11111, subtasks_list=[st_linguistic])

all_tasks = [
    multidimnews_19,
    subj_31,
    babe_10,
    basil_09,
    neutralizing_bias_26,
    newsWCL50_22,
    starbucks_38,
    cw_hard_03,
    offensive_language_86,
    online_harassment_dataset_87,
    wikidetox_aggression_and_attack_892,
    wikidetox_toxicity_891,
    jigsaw_40,
    me_too_ma_108,
    wikimadlibs_91,
    hateXplain_92,
    hatespeech_twitter_88,
    gap_18,
    rtgender_105,
    mdgender_116,
    trac2_104,
    funpedia_117,
    wizards_of_wikipedia_118,
    sst2_99,
    imdb_101,
    mpqa_103,
    semeval2014_63,
    amazon_reviews_100,
    liar_72,
    fake_news_net_25,
    pheme_12,
    crowSpairs_33,
    stereoset_64,
    stereotype_109,
    reddit_bias_75,
    good_news_everyone_42,
    bunemo_96,
    emotion_tweets_84,
    debate_effects_80,
    semeval2023_task4_119,
    semeval2023_task_3_120,
    vaccine_lies_127,
    semeval2016_task6_124,
    WTWT_126,
    multitargetstance_125,
    gwsd_128,
]
all_subtasks = list(itertools.chain.from_iterable(t.subtasks_list for t in all_tasks))


# Task families
media_bias = [multidimnews_19, basil_09, starbucks_38, semeval2023_task_3_120, babe_10]
subjective_bias = [subj_31, cw_hard_03, neutralizing_bias_26, newsWCL50_22]
hate_speech = [
    offensive_language_86,
    online_harassment_dataset_87,
    wikidetox_toxicity_891,
    wikidetox_aggression_and_attack_892,
    jigsaw_40,
    me_too_ma_108,
    wikimadlibs_91,
    hateXplain_92,
    hatespeech_twitter_88,
]
gender_bias = [gap_18, rtgender_105, mdgender_116, trac2_104, funpedia_117, wizards_of_wikipedia_118]
sentiment_analysis = [sst2_99, imdb_101, mpqa_103, semeval2014_63, amazon_reviews_100]
fake_news = [liar_72, fake_news_net_25, pheme_12]
group_bias = [crowSpairs_33, stereoset_64, stereotype_109, reddit_bias_75]
emotionality = [good_news_everyone_42, bunemo_96, emotion_tweets_84, debate_effects_80]
stance_detection = [
    semeval2023_task4_119,
    vaccine_lies_127,
    semeval2016_task6_124,
    WTWT_126,
    multitargetstance_125,
    gwsd_128,
]
mlm = [mlm_0]
