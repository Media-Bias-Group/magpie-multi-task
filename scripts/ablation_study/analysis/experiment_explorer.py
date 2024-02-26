"""Explore the results of Experiment 0."""

import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import wandb
from config import FIGSIZE, TABLE_CONFIG, WANDB_API_KEY
from utils import float_formatter

wandb.login(key=WANDB_API_KEY)
api = wandb.Api(timeout=60)

METRIC_CLS = "_eval_f1"
METRIC_REG = "_eval_R2"


def get_first_eval_from_hist(hist: Dict, st_ids: List):
    out = {}
    for st_id in st_ids:
        tmp = pd.DataFrame.from_dict(
            hist.get("heads+static+pcgrad_online+False").get(f"{st_id}_eval_f1"), orient="index"
        )
        mask = tmp[tmp.columns.to_list()[0]].apply(lambda x: x == x and x != "NaN")
        tmp = tmp[mask].dropna()
        first_eval_f1 = float(tmp[mask].iloc[0])
        out[st_id] = first_eval_f1

    df = pd.DataFrame.from_dict(out, orient="index")
    df.columns = ["first_eval_f1"]
    return df


def get_best_eval_from_hist(hist: Dict, st_ids: List):
    out = {}
    for st_id in st_ids:
        tmp = pd.DataFrame.from_dict(
            hist.get("heads+uniform+pcgrad_online+True").get(f"{st_id}_eval_f1"), orient="index"
        )
        mask = tmp[tmp.columns.to_list()[0]].apply(lambda x: x == x and x != "NaN")
        tmp = tmp[mask].dropna()
        best_eval_f1 = float(tmp[mask].astype(float).max())
        out[st_id] = best_eval_f1

    df = pd.DataFrame.from_dict(out, orient="index")
    df.columns = ["best_eval_f1"]
    return df


class Experiment0Explorer:
    """Explore Experiment 0."""

    def __init__(self, run_path: str, force_download: bool = False):
        """Initialize the Explorer."""
        filename_runs = "logging/experiment_0/final_runs_comparison.csv"
        filename_hists = "logging/experiment_0/hists.json"

        if not os.path.exists(filename_hists) or not os.path.exists(filename_runs) or force_download:
            hist_dict = {}
            runs = api.runs(run_path)
            row_list = []

            for run in tqdm(runs):
                run_dict = {}
                sweep = run.sweep.name
                run_dict.update({"sweep": sweep})
                run_dict.update(run.summary._json_dict)

                run_dict.update(run.config)  # hyperparameters

                row_list.append(run_dict)
                key = "+".join(list(map(lambda x: str(x), run.config.values())))
                hist_dict[key] = run.history().to_dict()

            df = pd.DataFrame(row_list)
            df.to_csv(filename_runs, index=False)
            json_object = json.dumps(hist_dict, indent=4)
            with open(filename_hists, "w") as outfile:
                outfile.write(json_object)
        else:
            df = pd.read_csv(filename_runs)
        self.df = df
        self.df["combined_f1_score"] = self.df.filter(regex="eval_f1").mean(axis=1)
        self.hist = json.load(open(filename_hists))

    def plot_vs_baselines(self):
        plt.close()
        task_specific_baselines = pd.read_csv("logging/hyperparam-tuning/task_specific_var.csv", index_col=0)
        # Join with baselines
        best_mtl_model = self.df["combined_f1_score"].idxmax()  # as it's sorted, the first row is the best
        subtasks = self.df.filter(regex="_eval_f1")
        st_ids = [int(x.split("_")[0]) for x in subtasks.columns.to_list()]
        subtasks.columns = st_ids
        subtasks = subtasks.T
        subtasks = subtasks[[best_mtl_model]]
        subtasks.rename(columns={0: "MTL Model final evaluation"}, inplace=True)

        joined = task_specific_baselines.join(subtasks, how="right")
        first_eval_f1s = get_first_eval_from_hist(hist=self.hist, st_ids=st_ids)
        joined = joined.join(first_eval_f1s, how="left")
        joined.sort_values(by=joined.columns.to_list(), inplace=True)

        joined.reset_index(inplace=True)
        joined.drop(columns="index", inplace=True)
        joined.dropna(inplace=True)
        joined.rename(
            columns={
                "metric": "Single-Task Model",
                "first_eval_f1": "MTL Model first evaluation",
            },
            inplace=True,
        )
        means = joined.mean(axis=0)

        fig, axs = plt.subplots(ncols=2, figsize=FIGSIZE)

        markers = ["o", "s", "v"]
        for i, c in enumerate(joined.columns[:2]):
            sns.lineplot(x=joined.index, y=c, data=joined, label=c, ax=axs[0], marker=markers[i])

        for i, c in enumerate(joined.columns[1:]):
            sns.lineplot(x=joined.index, y=c, data=joined, label=c, ax=axs[1], marker=markers[i])

        # axs[0].axhline(y=means[0], label="Average F1 Score Baselines")
        # axs[0].axhline(y=means[1], label="Average F1 Score MTL")
        plt.legend()
        axs[0].set(ylabel=None)
        axs[1].set(ylabel=None)
        axs[0].set_xticks([])
        axs[1].set_xticks([])
        fig.suptitle("Multi-Task vs Single-Task Model F1 Score Comparison", fontsize=16)
        fig.supylabel("F1 Scores")
        fig.supxlabel("Datasets")
        fig.tight_layout()

        plt.savefig("outputs/f1_scores_comparison.png")

    def get_top_k_table(self):
        # table = df.nsmallest(columns="combined_eval_loss", n=24)
        df = self.df

        table = df.nlargest(columns="combined_f1_score", n=8)
        table = table.sort_values(by="combined_f1_score")  # The 8 best runs (highest f1 score) are >0.700 f1

        table = table[["es", "scaling", "grad_agg", "resurrection", "combined_f1_score", "combined_eval_loss"]]
        styler = table.style
        styler.format_index(formatter=float_formatter)

        with open(os.path.join("outputs", "experiment_0.tex"), "w") as tf:
            latex_code = styler.to_latex(
                caption="Output of Experiment 0, tok 8 experiments", label="fig:experiment_0", **TABLE_CONFIG
            )
            tf.write(latex_code)

    def analyse(self):
        """Analyze the wandb data of experiment 0."""
        plt.close()
        df = self.df

        # Plot loss
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=FIGSIZE)

        w = 0.4
        b1 = sns.boxplot(data=df, x="resurrection", y="combined_eval_loss", showmeans=False, ax=axs[0, 0], width=w)
        b2 = sns.boxplot(data=df, x="grad_agg", y="combined_eval_loss", showmeans=False, ax=axs[0, 1], width=w)
        b3 = sns.boxplot(data=df, x="es", y="combined_eval_loss", showmeans=False, ax=axs[1, 0], width=w)
        b4 = sns.boxplot(data=df, x="scaling", y="combined_eval_loss", showmeans=False, ax=axs[1, 1], width=w)
        b1.set(ylabel=None)
        b2.set(ylabel=None)
        b3.set(ylabel=None)
        b4.set(ylabel=None)
        fig.suptitle("Evaluation Losses for Multi-Task Experiments with Various Strategies", fontsize=16)
        fig.supylabel("Aggregated Evaluation Loss")
        fig.tight_layout()
        plt.savefig("outputs/eval_losses_experiment_0.png")

        # Plot f1
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=FIGSIZE)

        w = 0.4
        b1 = sns.boxplot(data=df, x="resurrection", y="combined_f1_score", showmeans=False, ax=axs[0, 0], width=w)
        b2 = sns.boxplot(data=df, x="grad_agg", y="combined_f1_score", showmeans=False, ax=axs[0, 1], width=w)
        b3 = sns.boxplot(data=df, x="es", y="combined_f1_score", showmeans=False, ax=axs[1, 0], width=w)
        b4 = sns.boxplot(data=df, x="scaling", y="combined_f1_score", showmeans=False, ax=axs[1, 1], width=w)

        b1.set(ylabel=None)
        b2.set(ylabel=None)
        b3.set(ylabel=None)
        b4.set(ylabel=None)
        fig.suptitle("F1 Scores for Multi-Task Experiments with Various Strategies", fontsize=16)
        fig.supylabel("Aggregated F1 Scores")
        fig.tight_layout()
        plt.savefig("outputs/f1_scores_experiment_0.png")


experiment_0_explorer = Experiment0Explorer(run_path="media-bias-group/experiment_0", force_download=False)
experiment_0_explorer.analyse()
experiment_0_explorer.plot_vs_baselines()
experiment_0_explorer.get_top_k_table()
