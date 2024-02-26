"""Explore results of GradTS on media bias domain."""

import os

import dataframe_image as dfi
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from scipy.stats import kendalltau
from tqdm import tqdm

import wandb
from config import WANDB_API_KEY
from storage import storage_client


class Experiment2Explorer:
    """Explore results of GradTS on media bias domain."""

    def __init__(self):
        """Initialize the Explorer."""
        self.run_path = "media-bias-group/experiment_2"
        self.head_matrices_path = "model_files/gradts/"
        self.logging_path = "logging/experiment_2/"
        self.f1_results_path = self.logging_path + "results.csv"
        self.head_matrices = {}
        self.hist_dict = {}
        self.results = None
        self.corr_ranking = None

        # download head_matrices files
        if not os.path.exists(self.head_matrices_path):
            storage_client.download_from_gcs_to_local_directory_or_file(
                local_path="", gcs_path=self.head_matrices_path
            )

        if not os.path.exists(self.f1_results_path):
            wandb.login(key=WANDB_API_KEY)
            api = wandb.Api(timeout=20)
            runs = api.runs(self.run_path)
            row_list = []

            for run in tqdm(runs):
                run_dict = {}
                name = run.name
                if "first" not in name or "pre-training" in name:
                    continue
                run_dict.update({"name": name})
                run_dict.update(run.summary._json_dict)
                row_list.append(run_dict)
                self.hist_dict[name] = run.scan_history()

            self.df = pd.DataFrame(row_list)
            self.get_eval_results_from_history()
        else:
            self.results = pd.read_csv(self.f1_results_path)

    def get_eval_results_from_history(self):
        """Go through 15 runs (we kept adding tasks until 15) and iterate over history_scan (which is a wandb object used when there is too much data, unfortunately, non-serializable) and take the first eval_f1 score."""
        row_list = []
        f1 = "10001_eval_f1"
        loss = "10001_eval_loss"

        for i in tqdm(range(1, 16)):
            num = str(i)
            d = {}
            d["num_of_tasks"] = num

            # pretrained = it was pretrained on $num tasks and then finetuned on babe
            for h in self.hist_dict["pretrained-first-" + num + "-finetuning"]:
                if f1 in h.keys():
                    if h[f1] != "NaN":
                        d["pretrained_f1"] = h[f1]
                        d["pretrained_loss"] = h[loss]
                        break

            # first-$num-tasks is co-training babe with tasks in standard mtl way
            for h in self.hist_dict["first-" + num + "-tasks"]:
                if f1 in h.keys():
                    if h[f1] != "NaN":
                        d["cotrained_f1"] = h[f1]
                        d["cotrained_loss"] = h[loss]
                        break

            # the same as previous but with backbone option. I accidentally ran the experiment with this option at first. we can still analyze it
            for h in self.hist_dict["first-" + num + "-tasks-backbone"]:
                if f1 in h.keys():
                    if h[f1] != "NaN":
                        d["cotrained_bb_f1"] = h[f1]
                        d["cotrained_bb_loss"] = h[loss]
                        break

            row_list.append(d)

        self.results = pd.DataFrame(row_list)
        self.results.to_csv(self.f1_results_path, index=False)

    def get_att_head_ranking(self):
        """Load all attention head matrices, and compare correlation wrt babe."""
        matrices = os.listdir(self.head_matrices_path)
        babe_mat = torch.load(self.head_matrices_path + "10001_att_mat.pt")
        row_list = []
        for m in matrices:
            st_id = m[:-11]
            mat = torch.load(self.head_matrices_path + m)
            corr, _ = kendalltau(babe_mat.reshape(12 * 12, -1), mat.reshape(12 * 12, -1))
            row_list.append({"subtask": st_id, "corr": corr})
            self.head_matrices[st_id] = mat

        df = pd.DataFrame(row_list).sort_values(by="corr", ascending=False)
        self.corr_ranking = df

        cmap = LinearSegmentedColormap.from_list("rg", ["r", "w", "g"], N=256)
        dfi.export(df.style.background_gradient(cmap=cmap), self.logging_path + "corr_ranking.png")

        return df

    def plot_att_head_matrix(self, st_id: str):
        """Plot attention head matrix for one subtask."""
        gaus = gaussian_filter(self.head_matrices[st_id], sigma=2)
        sns.heatmap(gaus, vmin=gaus.min(), vmax=gaus.max(), cmap="Blues", cbar=False)
        plt.show()

    def plot_top_3_matrices(self):
        """Plot BABE + two most correlating matrices and 3 least correlated ones."""
        if self.corr_ranking is None:
            self.get_att_head_ranking()
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8), sharey=True)
        babe_mat = gaussian_filter(self.head_matrices["10001"], sigma=2)
        sns.heatmap(babe_mat, vmin=babe_mat.min(), vmax=babe_mat.max(), cmap="Blues", ax=axes[0][0], cbar=False)
        axes[0][0].set_title("BABE")

        # plotting 2 most correlated matrices to BABE
        for i in range(1, 3):
            st = str(self.corr_ranking.iloc[i]["subtask"])
            mat = gaussian_filter(self.head_matrices[st], sigma=2)
            sns.heatmap(mat, vmin=mat.min(), vmax=mat.max(), cmap="Blues", ax=axes[0][i], cbar=False)
            axes[0][i].set_title(st)

        # plotting 3 least correlated matrices to BABE
        for i in range(1, 4):
            st = str(self.corr_ranking.iloc[-i]["subtask"])
            mat = gaussian_filter(self.head_matrices[st], sigma=2)
            sns.heatmap(mat, vmin=mat.min(), vmax=mat.max(), cmap="Blues", ax=axes[1][i - 1], cbar=False)
            axes[1][i - 1].set_title(st)

        plt.savefig(self.logging_path + "corr_matrices.png")

    def plot_mtl_training(self, plt_type="lines"):
        """Plot the evolution of the f1 score on BABE when adding the tasks."""
        if plt_type == "lines":
            g1 = sns.lineplot(x=self.results["num_of_tasks"], y=self.results["cotrained_f1"])
            plt.savefig(self.logging_path + "f1s_cotrained.png")
            plt.close(g1.get_figure())
            g2 = sns.lineplot(x=self.results["num_of_tasks"], y=self.results["cotrained_loss"])
            plt.savefig(self.logging_path + "losses_cotrained.png")
            plt.close(g2.get_figure())
            g3 = sns.lineplot(x=self.results["num_of_tasks"], y=self.results["pretrained_f1"])
            plt.savefig(self.logging_path + "f1s_pretrained.png")
            plt.close(g3.get_figure())
            _ = sns.lineplot(x=self.results["num_of_tasks"], y=self.results["pretrained_loss"])
            plt.savefig(self.logging_path + "losses_pretrained.png")
        if plt_type == "lm":
            g = sns.lmplot(data=self.results, x="num_of_tasks", y="cotrained_f1", fit_reg=True, order=7, ci=None)
            ax1 = g.axes[0][0]
            ax1.axhline(0.8073, ls="--")
            plt.savefig(self.logging_path + "f1s_fitted.png")


exp = Experiment2Explorer()
exp.plot_mtl_training()
