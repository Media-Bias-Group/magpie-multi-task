"""Explore the results of Experiment 0."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import wandb
from config import FIGSIZE, WANDB_API_KEY

wandb.login(key=WANDB_API_KEY)
api = wandb.Api(timeout=60)

METRIC_CLS = "_eval_f1"
METRIC_REG = "_eval_R2"

all_families = [
    "media_bias",
    "subjective_bias",
    "hate_speech",
    "gender_bias",
    "sentiment_analysis",
    "fake_news",
    "group_bias",
    "emotionality",
    "stance_detection",
]


class Experiment1Explorer:
    """Explore Experiment 1."""

    def __init__(self, run_path: str, force_download: bool = False):
        """Initialize the Explorer."""
        filename_runs = "logging/experiment_1/final_runs_comparison.csv"
        filename_hists = "logging/experiment_1/hists.json"

        if not os.path.exists(filename_hists) or not os.path.exists(filename_runs) or force_download:
            hist_dict = {}
            runs = api.runs(run_path)
            row_list = []

            for run in tqdm(runs):
                run_dict = {}
                sweep = run.name
                run_dict.update({"sweep": sweep})
                run_dict.update(run.summary._json_dict)

                run_dict.update(run.config)  # hyperparameters

                row_list.append(run_dict)
                hist_dict[sweep] = run.history().to_dict()

            df = pd.DataFrame(row_list)
            df.to_csv(filename_runs, index=False)
            json_object = json.dumps(hist_dict, indent=4)
            with open(filename_hists, "w") as outfile:
                outfile.write(json_object)
        else:
            df = pd.read_csv(filename_runs)
        self.df = df

        # self.df["combined_f1_score"] = self.df.filter(regex=METRIC_CLS).mean(axis=1)
        # self.df["combined_r2_score"] = self.df.filter(regex=METRIC_REG).mean(axis=1)
        # self.df["combined_metric"] = df.filter(regex=f"{METRIC_CLS}|{METRIC_REG}").mean(axis=1)
        self.hist = json.load(open(filename_hists))

    def between_family_transfer(self):
        """Evaluate between-family transfer."""
        df = self.df.copy(deep=True)

        between_family_transfer_matrix = np.empty(shape=(len(all_families), len(all_families)))

        for i, family1 in enumerate(all_families):
            family1_df = df[df["sweep"] == family1].dropna(axis=1)
            family1_eval_columns = family1_df.filter(regex="_eval_f1")
            family_1_performance = family1_eval_columns.mean(axis=1)

            for j, family2 in enumerate(all_families):
                if family1 == family2:
                    continue

                if len(df[df["sweep"] == f"co-training-{family1}-{family2}"]) == 1:
                    co_training_df = df[df["sweep"] == f"co-training-{family1}-{family2}"]
                elif len(df[df["sweep"] == f"co-training-{family2}-{family1}"]) == 1:
                    co_training_df = df[df["sweep"] == f"co-training-{family2}-{family1}"]
                else:
                    raise Exception

                family_1_co_training_performance = co_training_df[family1_eval_columns.columns.to_list()].mean(axis=1)
                difference = family_1_co_training_performance.values - family_1_performance.values
                relative_difference = difference / family_1_performance.values
                between_family_transfer_matrix[i, j] = relative_difference

                # # Now we want to check how well the tasks from family_1 performed when trained together with all tasks from that family + all from the second family
                # st_ids = list(map(lambda x: int(x.split("_")[0]), family1_eval_columns.columns.to_list()))
                # d = dict(zip(family1_eval_columns.columns.to_list(), st_ids))
                #
                # a = co_training_df[family1_eval_columns.columns.to_list()].rename(columns=d).T
                # a.columns = [f"co-training-{family1}-{family2}"]
                # joined = task_specific_baselines.join(a, how="right")
                # within_family_transfer = (joined[family1] - joined["metric"]) / joined["metric"]

    def within_family_transfer(self):
        """Evaluate within-family transfer."""
        df = self.df.copy(deep=True)
        task_specific_baselines = pd.read_csv("logging/hyperparam-tuning/task_specific_var.csv", index_col=0)
        # within_family_transfer_dict = {f: None for f in all_families}
        # single_family_performance = np.empty(shape=len(all_families))
        tasks = {}
        conflicting_gradients_ratio = {}
        for i, family in enumerate(all_families):
            family_df = df[df["sweep"] == family].dropna(axis=1)
            family_eval_columns = family_df.filter(regex="_eval_f1")
            # family_performance = family_eval_columns.mean(axis=1)
            # single_family_performance[i] = family_performance

            st_ids = list(map(lambda x: int(x.split("_")[0]), family_eval_columns.columns.to_list()))
            d = dict(zip(family_eval_columns.columns.to_list(), st_ids))

            within_family_training = family_eval_columns.rename(columns=d).T
            within_family_training.columns = [family]
            joined = within_family_training.join(task_specific_baselines)
            joined["within_family_transfer"] = (joined[family] - joined["metric"]) / joined["metric"]
            joined.rename(columns={"metric": "single_task_performance", family: "mtl_single_family"}, inplace=True)
            joined["family"] = family
            out = joined[["family", "within_family_transfer"]].T.to_dict()
            out = joined.T.to_dict()
            tasks = {**tasks, **out}
            hist = self.hist[family]
            conflicting_gradients_ratio[family] = pd.Series(hist["conflicting_gradients_ratio"]).dropna().values.mean()

        fig, axs = plt.subplots(ncols=2, figsize=FIGSIZE)
        result = pd.DataFrame.from_dict(tasks, orient="index")
        medians = result.groupby("family").agg(median_transfer=("within_family_transfer", "median"))
        result = result.join(medians, on="family")
        result.dropna(inplace=True)
        # result.sort_values(by="median_transfer", inplace=True)
        # result.sort_values(by="within_family_transfer", inplace=True)

        result.reset_index(drop=True, inplace=True)
        s1 = sns.scatterplot(data=result, y="within_family_transfer", x=result.index, hue="family", ax=axs[0])
        s1._remove_legend(axs[0].legend)
        s1.set_xticks([])

        a = result.groupby("family").agg({"median_transfer": "first"}).sort_values(by="median_transfer")
        s2 = sns.scatterplot(a, hue="family", y="median_transfer", x=a.index, ax=axs[1])
        s2.set_xticks([])
        axs[1].axhline(0)
        axs[0].axhline(0)
        fig.supylabel("Relative Performance Change")
        fig.supxlabel("Datasets")
        fig.suptitle("Evaluation of Within-Family Transfer", fontsize=16)
        axs[0].set_title("Per-Dataset Transfer")
        axs[1].set_title("Per-Family Transfer (Median)")
        axs[0].set(ylabel=None)
        axs[1].set(ylabel=None)
        axs[0].set(xlabel=None)
        axs[1].set(xlabel=None)
        handles, labels = axs[0].get_legend_handles_labels()
        plt.tight_layout()
        plt.show()

        ## Analyze conflicting gradients ratio
        conflicting_gradients = pd.DataFrame.from_dict(conflicting_gradients_ratio, orient="index")
        conflicting_gradients.reset_index(inplace=True)
        conflicting_gradients.columns = ["family", "conflicting_gradients_ratio"]

        conflicting_gradients.sort_values(by="family", inplace=True)

        median_transfer = result.sort_values(by="family").groupby("family").agg({"median_transfer": "first"})
        median_transfer.values
        conflicting_gradients.values
        plt.scatter(
            conflicting_gradients["conflicting_gradients_ratio"].values.squeeze().tolist(),
            median_transfer.values.squeeze().tolist(),
        )
        plt.show()

        fig, ax = plt.subplots(figsize=FIGSIZE)

        # result = result.sort_values(by="single_task_performance").reset_index(drop=True)
        # sns.lineplot(
        #     x=result.index,
        #     y="single_task_performance",
        #     data=result,
        #     marker="o",
        #     ax=axs[0]
        # )
        # sns.lineplot(
        #     x=result.index,
        #     y="mtl_single_family",
        #     data=result,
        #     marker="o",
        #     ax=axs[0]
        # )
        # plt.show()


experiment_1_explorer = Experiment1Explorer(run_path="media-bias-group/experiment_1", force_download=False)
experiment_1_explorer.within_family_transfer()
# experiment_1_explorer.between_family_transfer()
