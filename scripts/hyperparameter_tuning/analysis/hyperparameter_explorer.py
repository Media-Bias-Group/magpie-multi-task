"""Explore the hyperparameter runs."""
import os
from functools import reduce

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import wandb
from config import FIGSIZE, WANDB_API_KEY

# sns.set_style("dark")

wandb.login(key=WANDB_API_KEY)
api = wandb.Api(timeout=60)

METRIC_CLS = "_eval_f1"
METRIC_REG = "_eval_R2"


class HyperparameterExplorer:
    """A HyperparameterExplorer."""

    def __init__(self, run_path: str, force_download: bool = False):
        """Initialize a HyperparameterExplorer."""
        filename = "logging/hyperparam-tuning/all_data.csv"
        if not os.path.exists(filename) or force_download:
            # Project is specified by <entity/project-name>
            runs = api.runs(run_path)
            row_list = []

            for run in tqdm(runs):
                run_dict = {}
                # it threw an error in one case, might be something with unfinished sweeps, too lazy to find the cause
                if run.sweep is not None:
                    st_id = run.sweep.name
                else:
                    continue
                run_dict.update({"subtask": st_id})
                if st_id + METRIC_CLS in run.summary._json_dict.keys():
                    run_dict.update({"metric": run.summary._json_dict[st_id + METRIC_CLS]})
                elif st_id + METRIC_REG in run.summary._json_dict.keys():
                    run_dict.update({"metric": run.summary._json_dict[st_id + METRIC_REG]})

                run_dict.update(run.config)  # hyperparameters

                row_list.append(run_dict)

            df = pd.DataFrame(row_list)
            df.to_csv(filename, index=False)
        else:
            df = pd.read_csv(filename)
        self.df = df

    def process(self):
        """Process the different hyperparameter settings."""
        df = self.df.copy(deep=True)
        df = df[df.max_epoch < 15]
        #### GET TASK SPECIFIC HYPERPARAMETER BY OPTIMIZING FOR BEST LOWER BOUND

        lr = (df.groupby(["subtask", "lr"]).agg("mean") - df.groupby(["subtask", "lr"]).agg("std"))[["metric"]]
        lr = lr.groupby("subtask").idxmax().reset_index()
        lr["lr"] = lr["metric"].apply(lambda x: x[1])
        lr.drop(columns="metric", inplace=True)

        pt = (df.groupby(["subtask", "patience"]).agg("mean") - df.groupby(["subtask", "patience"]).agg("std"))[
            ["metric"]
        ]
        pt = pt.groupby("subtask").idxmax().reset_index()
        pt["patience"] = pt["metric"].apply(lambda x: x[1])
        pt.drop(columns="metric", inplace=True)

        epoch = (df.groupby(["subtask", "max_epoch"]).agg("mean") - df.groupby(["subtask", "max_epoch"]).agg("std"))[
            ["metric"]
        ]
        epoch = epoch.groupby("subtask").idxmax().reset_index()
        epoch["max_epoch"] = epoch["metric"].apply(lambda x: x[1])
        epoch.drop(columns="metric", inplace=True)

        data_frames = [lr, epoch, pt]
        df_merged = reduce(lambda left, right: pd.merge(left, right, on=["subtask"], how="outer"), data_frames)

        final = df_merged[["subtask", "lr", "max_epoch", "patience"]]
        # to extract the original score of particular hyperparameters
        df.merge(final, how="right", on=["subtask", "lr", "max_epoch", "patience"])[["subtask", "metric"]].to_csv(
            "logging/hyperparam-tuning/task_specific_var.csv", index=False
        )

        #### GET GLOBAL HYPERPARAMS
        df = self.df.copy(deep=True)
        # special case for regression tasks
        df = df[df.max_epoch < 15]

        best_lr = float(df.groupby("lr").agg({"metric": "mean"}).idxmax(axis=0))
        best_patience = int(df.groupby("patience").agg({"metric": "mean"}).idxmax(axis=0))
        best_max_epoch = int(df.groupby("max_epoch").agg({"metric": "mean"}).idxmax(axis=0))

        glob = df[(df.max_epoch == best_max_epoch) & (df.lr == best_lr) & (df.patience == best_patience)][
            ["subtask", "metric"]
        ]
        glob.to_csv("logging/hyperparam-tuning/global_var.csv", index=False)

        #### GET TASK SPECIFIC HYPERPARAMETERS BY TAKING THE BEST PERFORMING PARAMETERS
        df = self.df.copy(deep=True)
        df = df[df.max_epoch < 15]
        df_optimal = df.loc[df.groupby(["subtask"])["metric"].idxmax()]
        df_optimal[["subtask", "metric"]].to_csv("logging/hyperparam-tuning/optimal_hyperparameters.csv", index=False)

    def analyze(self):
        """Analyze the different hyperparameter settings."""
        optimal = pd.read_csv("logging/hyperparam-tuning/optimal_hyperparameters.csv", index_col="subtask")
        task_specific_var = pd.read_csv("logging/hyperparam-tuning/task_specific_var.csv", index_col="subtask")
        global_var = pd.read_csv("logging/hyperparam-tuning/global_var.csv", index_col="subtask")

        optimal = optimal.sort_index()
        task_specific_var = task_specific_var.sort_index()
        global_var = global_var.sort_index()

        assert all(global_var.index == task_specific_var.index) and all(global_var.index == optimal.index)
        optimal.rename(columns={"metric": "metric_optimal"}, inplace=True)
        task_specific_var.rename(columns={"metric": "metric_task_specific_var"}, inplace=True)
        global_var.rename(columns={"metric": "metric_global_var"}, inplace=True)
        df_all = pd.concat([optimal, task_specific_var, global_var], axis=1)
        df_all = df_all.dropna().sort_values(by="metric_optimal").reset_index()

        fig, ax = plt.subplots(ncols=1, figsize=FIGSIZE)

        fig.suptitle("Hyperparameter Optimization Strategies and Model Performance", fontsize=16)

        ax.set(xticklabels=[])
        ax.set(xlabel="")
        ax.set(ylabel="")

        fig.supylabel("Metric")
        fig.supxlabel("Datasets")

        sns.lineplot(
            x=df_all.index,
            y="metric_optimal",
            data=df_all,
            ax=ax,
            label="Optimal Hyperparameters (Task-Specific)",
            marker="o",
        )
        sns.lineplot(
            x=df_all.index,
            y="metric_task_specific_var",
            data=df_all,
            ax=ax,
            label="Hyperparameters (Task-Specific Adjustment)",
            marker="v",
        )
        sns.lineplot(
            x=df_all.index, y="metric_global_var", data=df_all, ax=ax, label="Hyperparameters (Global)", marker="s"
        )

        ax.set(xlabel="")
        ax.set(ylabel="")
        fig.tight_layout()
        plt.savefig("outputs/hyperparam_tuning.png")


hyperparam_explorer = HyperparameterExplorer(run_path="media-bias-group/hyperparam-tuning", force_download=False)
hyperparam_explorer.process()
hyperparam_explorer.analyze()
