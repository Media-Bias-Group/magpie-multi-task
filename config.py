"""Config file."""

import os
from enum import Enum

from dotenv import load_dotenv

load_dotenv(dotenv_path="local.env")

# TWITTER API
API_KEY = os.getenv("API_KEY")
API_KEY_SECRET = os.getenv("API_KEY_SECRET")
TOKEN = os.getenv("TOKEN")
TOKEN_SECRET = os.getenv("TOKEN_SECRET")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

# CLUSTER CREDS
SERVER_URL = os.getenv("SERVER_URL")
SERVER_NAME = os.getenv("SERVER_NAME")
VPN_USERNAME = os.getenv("VPN_USERNAME")
VPN_PASSWORD = os.getenv("VPN_PASSWORD")
CLUSTER_PASSWORD = os.getenv("CLUSTER_PASSWORD")

NEW_TOKENS = ["<TAG-A>", "<TAG-B>", "<TAG-P"]

# WANDB
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

dataset_id_to_dataset_name = {
    0: "MLM",
    3: "CW_HARD",
    9: "BASIL",
    10: "BABE",
    12: "PHEME",
    18: "GAP",
    19: "MultiDimNews",
    22: "NewsWCL50",
    25: "FakeNewsNet",
    26: "NeutralizingBias",
    31: "SUBJ",
    33: "CrowSPairs",
    38: "Starbucks",
    40: "JIGSAW",
    42: "GoodNewsEveryone",
    63: "SemEval2014",
    64: "StereoSet",
    72: "LIAR",
    75: "RedditBias",
    80: "DebateEffects",
    84: "EmotionTweets",
    86: "OffensiveLanguage",
    87: "OnlineHarassment",
    88: "HateSpeechTwitter",
    91: "WikiMadlibs",
    92: "HateXplain",
    96: "Bu-NEMO",
    99: "SST2",
    100: "AmazonReviews",
    101: "IMDB",
    103: "MPQA",
    104: "TRAC2",
    105: "RtGender",
    108: "MeTooMA",
    109: "Stereotype",
    116: "MDGender",
    117: "Funpedia",
    118: "WizardsOfWikipedia",
    119: "SemEval2023Task4",
    120: "SemEval2023Task3",
    124: "SemEval2016Task6",
    125: "MultiTargetStance",
    126: "WTWT",
    128: "GWSD",
    127: "VaccineLies",
    891: "WikiDetoxToxicity",
    892: "WikiDetoxAggression",
}


class TaskFamilies(Enum):
    """Task Families."""

    MLM = "Masked Language Modelling"
    SUBJECTIVITY = "Subjectivity"
    MEDIA_BIAS = "Media Bias"
    HATE_SPEECH = "Hate Speech"
    GENDER_BIAS = "Gender Bias"
    SENTIMENT_ANALYSIS = "Sentiment Analysis"
    FAKE_NEWS_DETECTION = "Fake News Detection"
    GROUP_BIAS = "Group Bias"
    EMOTIONALITY = "Emotionality"
    STANCE_DETECTION = "Stance Detection"


dataset_id_to_family = {
    0: TaskFamilies.MLM,
    3: TaskFamilies.SUBJECTIVITY,
    9: TaskFamilies.MEDIA_BIAS,
    10: TaskFamilies.MEDIA_BIAS,
    12: TaskFamilies.FAKE_NEWS_DETECTION,
    18: TaskFamilies.GENDER_BIAS,
    19: TaskFamilies.MEDIA_BIAS,
    22: TaskFamilies.MEDIA_BIAS,  # TODO incorrect
    25: TaskFamilies.FAKE_NEWS_DETECTION,
    26: TaskFamilies.SUBJECTIVITY,
    31: TaskFamilies.SUBJECTIVITY,
    33: TaskFamilies.GROUP_BIAS,
    38: TaskFamilies.MEDIA_BIAS,
    40: TaskFamilies.HATE_SPEECH,
    42: TaskFamilies.EMOTIONALITY,
    63: TaskFamilies.SENTIMENT_ANALYSIS,
    64: TaskFamilies.GROUP_BIAS,
    72: TaskFamilies.FAKE_NEWS_DETECTION,
    75: TaskFamilies.GROUP_BIAS,
    80: TaskFamilies.EMOTIONALITY,
    84: TaskFamilies.EMOTIONALITY,
    86: TaskFamilies.HATE_SPEECH,
    87: TaskFamilies.HATE_SPEECH,
    88: TaskFamilies.HATE_SPEECH,
    91: TaskFamilies.HATE_SPEECH,
    92: TaskFamilies.HATE_SPEECH,
    96: TaskFamilies.EMOTIONALITY,
    99: TaskFamilies.SENTIMENT_ANALYSIS,
    100: TaskFamilies.SENTIMENT_ANALYSIS,
    101: TaskFamilies.SENTIMENT_ANALYSIS,
    103: TaskFamilies.SENTIMENT_ANALYSIS,
    104: TaskFamilies.GENDER_BIAS,
    105: TaskFamilies.GENDER_BIAS,
    108: TaskFamilies.GENDER_BIAS,
    109: TaskFamilies.GROUP_BIAS,
    116: TaskFamilies.GENDER_BIAS,
    117: TaskFamilies.GENDER_BIAS,
    118: TaskFamilies.GENDER_BIAS,
    119: TaskFamilies.STANCE_DETECTION,
    120: TaskFamilies.MEDIA_BIAS,
    124: TaskFamilies.STANCE_DETECTION,
    125: TaskFamilies.STANCE_DETECTION,
    126: TaskFamilies.STANCE_DETECTION,
    127: TaskFamilies.STANCE_DETECTION,
    128: TaskFamilies.STANCE_DETECTION,
    891: TaskFamilies.HATE_SPEECH,
    892: TaskFamilies.HATE_SPEECH,
}

MAX_NUMBER_OF_STEPS = 1000

# Task-configs
MAX_LENGTH = 128

# reproducibility
RANDOM_SEED = 321

# regression constant scalar
REGRESSION_SCALAR = 2.5

# Split ratio for train/ dev/ test
TRAIN_RATIO, DEV_RATIO, TEST_RATIO = 0.8, 0.1, 0.1

# plotting configuration
# TODO find the correct width from the latex document
width, fraction = 570, 1
fig_width_pt = width * fraction
# Convert from pt to inches
inches_per_pt = 1 / 72.27
# Golden ratio to set aesthetic figure height
# https://disq.us/p/2940ij3
golden_ratio = (5 ** 0.5 - 1) / 2
# Figure width in inches
fig_width_in = fig_width_pt * inches_per_pt
# Figure height in inches
fig_height_in = fig_width_in * golden_ratio

FIGSIZE = (fig_width_in, fig_height_in)
TABLE_CONFIG = {"hrules": True, "multicol_align": "l", "multirow_align": "t", "clines": "skip-last;index"}

# hyperparameter ranges
hyper_param_dict = {
    "lr": {"values": [5e-5, 4e-5, 3e-5, 2e-5, 1e-4]},
    "patience": {"values": [25, 50, 75, 100]},
    "max_epoch": {"values": [3, 5, 10]},
}


# TRAINING parameters
head_specific_lr = {
    "100001": 0.0001,
    "10001": 3e-05,
    "10002": 2e-05,
    "10101": 4e-05,
    "10301": 0.0001,
    "10401": 0.0001,
    "10501": 2e-05,
    "10801": 2e-05,
    "10901": 4e-05,
    "10902": 2e-05,
    "11601": 4e-05,
    "11701": 2e-05,
    "11801": 5e-05,
    "11901": 4e-05,
    "120001": 2e-05,
    "12001": 2e-05,
    "12002": 4e-05,
    "12401": 2e-05,
    "12501": 5e-05,
    "12601": 2e-05,
    "12701": 4e-05,
    "18001": 4e-05,
    "19001": 5e-05,
    "25001": 0.0001,
    "26001": 2e-05,
    "300001": 0.0001,
    "31001": 0.0001,
    "33001": 2e-05,
    "33002": 2e-05,
    "33003": 4e-05,
    "40001": 2e-05,
    "42001": 2e-05,
    "42002": 2e-05,
    "63001": 5e-05,
    "64001": 4e-05,
    "64002": 0.0001,
    "75001": 0.0001,
    "75002": 2e-05,
    "75003": 0.0001,
    "84001": 0.0001,
    "86001": 2e-05,
    "87001": 5e-05,
    "88001": 4e-05,
    "89202": 0.0001,
    "90001": 0.0001,
    "90002": 3e-05,
    "91001": 3e-05,
    "92001": 2e-05,
    "92002": 3e-05,
    "92003": 0.0001,
    "96001": 2e-05,
    "99001": 5e-05,
    "38001": 2e-05,
    "72001": 4e-05,
    "80001": 2e-05,
    "89101": 0.0001,
    "89201": 2e-05,
    "22001": 0.0001,
    "12801": 0.0001,
    "12901": 4e-5,
    "12902": 4e-5,
}
head_specific_patience = {
    "100001": 100,
    "10001": 75,
    "10002": 100,
    "10101": 100,
    "10301": 75,
    "10401": 100,
    "10501": 75,
    "10801": 25,
    "10901": 75,
    "10902": 75,
    "11601": 75,
    "11701": 100,
    "11801": 100,
    "11901": 100,
    "120001": 100,
    "12001": 100,
    "12002": 100,
    "12401": 100,
    "12501": 75,
    "12601": 100,
    "12701": 75,
    "18001": 100,
    "19001": 25,
    "25001": 100,
    "26001": 100,
    "300001": 50,
    "31001": 100,
    "33001": 50,
    "33002": 50,
    "33003": 100,
    "40001": 100,
    "42001": 100,
    "42002": 100,
    "63001": 100,
    "64001": 75,
    "64002": 75,
    "75001": 100,
    "75002": 75,
    "75003": 100,
    "84001": 100,
    "86001": 100,
    "87001": 25,
    "88001": 75,
    "89202": 75,
    "90001": 100,
    "90002": 100,
    "91001": 75,
    "92001": 100,
    "92002": 100,
    "92003": 100,
    "96001": 75,
    "99001": 75,
    "38001": 100,
    "72001": 75,
    "80001": 75,
    "89101": 100,
    "89201": 100,
    "22001": 75,
    "12801": 75,
    "12901": 50,
    "12902": 50,
}
head_specific_max_epoch = {
    "100001": 5,
    "10001": 3,
    "10002": 3,
    "10101": 5,
    "10301": 3,
    "10401": 3,
    "10501": 5,
    "10801": 3,
    "10901": 3,
    "10902": 5,
    "11601": 10,
    "11701": 5,
    "11801": 3,
    "11901": 3,
    "120001": 3,
    "12001": 3,
    "12002": 3,
    "12401": 3,
    "12501": 3,
    "12601": 5,
    "12701": 3,
    "18001": 5,
    "19001": 3,
    "25001": 3,
    "26001": 3,
    "300001": 3,
    "31001": 5,
    "33001": 10,
    "33002": 5,
    "33003": 5,
    "40001": 10,
    "42001": 3,
    "42002": 10,
    "63001": 5,
    "64001": 3,
    "64002": 5,
    "75001": 5,
    "75002": 3,
    "75003": 3,
    "84001": 3,
    "86001": 3,
    "87001": 5,
    "88001": 10,
    "89202": 3,
    "90001": 10,
    "90002": 3,
    "91001": 3,
    "92001": 10,
    "92002": 3,
    "92003": 5,
    "96001": 3,
    "99001": 3,
    "38001": 5,
    "72001": 10,
    "80001": 10,
    "89101": 10,
    "89201": 10,
    "22001": 15,
    "12801": 3,
    "12901": 5,
    "12902": 5,
}
