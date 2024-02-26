"""Explore the metadata."""

import itertools
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from config import (
    FIGSIZE,
    TABLE_CONFIG,
    TaskFamilies,
    dataset_id_to_dataset_name,
    dataset_id_to_family,
)
from storage import storage_client
from training.data import all_tasks
from utils import integer_formatter

# import seaborn as sns
# sns.set_context("poster")
# sns.color_palette("tab10")

matplotlib.use("TkAgg")


class MetadataExplorer:
    """MetadataExplorer."""

    def __init__(self):
        """Initialize a MetadataExplorer."""
        self.storage_client = storage_client
        self.storage_client.download_from_gcs_to_local_directory_or_file(local_path="", gcs_path="datasets/logging")

        metadata_files = [
            os.path.join("datasets/logging", f"{ds_id}-metadata.json") for ds_id in dataset_id_to_family.keys()
        ]
        self.df = pd.DataFrame([json.load(open(m)) for m in metadata_files])
        self.df["dataset_name"] = self.df.dataset_id.apply(lambda x: dataset_id_to_dataset_name[x])
        self.df["task_family"] = self.df.dataset_id.apply(lambda x: dataset_id_to_family[x].name)

    def get_plot_for_text_lengths(self):
        """Explore."""
        plt.close()
        fig, axs = plt.subplots(2, figsize=FIGSIZE)
        names = self.df.dataset_name.to_list()

        # plt.close()
        distributions = self.df.length_of_text_distribution_before_cleaning
        means = [d.get("mean") for d in distributions]
        top_95 = [int(d.get("95%")) for d in distributions]
        top_90 = [int(d.get("90%")) for d in distributions]
        counts = [d.get("count") for d in distributions]
        avg_top_95 = sum([t[0] * t[1] for t in list(zip(counts, top_95))]) / sum(counts)
        avg_top_90 = sum([t[0] * t[1] for t in list(zip(counts, top_90))]) / sum(counts)

        # with sns.axes_style("darkgrid"):
        stds = [d.get("std") for d in distributions]
        axs[0].errorbar(names, means, stds, linestyle="None", marker="o")
        axs[0].set_xticks([])
        axs[0].axhline(y=avg_top_95, linestyle="-", label="95% threshold", c="orange")
        axs[0].axhline(y=avg_top_90, linestyle="-", alpha=0.5, label="90% threshold", c="orange")
        axs[0].legend()

        axs[0].set_title("Before preprocessing")

        distributions_1 = self.df.length_of_text_distribution
        means_1 = [d.get("mean") for d in distributions_1]
        top_99_1 = [int(d.get("99%")) for d in distributions_1]
        top_95_1 = [int(d.get("95%")) for d in distributions_1]
        top_90_1 = [int(d.get("90%")) for d in distributions_1]
        counts_1 = [d.get("count") for d in distributions_1]
        avg_top_99_1 = sum([t[0] * t[1] for t in list(zip(counts_1, top_99_1))]) / sum(counts_1)
        avg_top_95_1 = sum([t[0] * t[1] for t in list(zip(counts_1, top_95_1))]) / sum(counts_1)
        avg_top_90_1 = sum([t[0] * t[1] for t in list(zip(counts_1, top_90_1))]) / sum(counts_1)

        stds_1 = [d.get("std") for d in distributions_1]
        axs[1].errorbar(names, means_1, stds_1, linestyle="None", marker="o")
        # axs[1].set_xticks(ticks=[x+1 for x in range(len(self.df))], labels=names, ha="right")
        axs[1].set_xticklabels(names, rotation=60, ha="right")

        axs[1].axhline(y=avg_top_99_1, linestyle="-", alpha=0.5, label="99% threshold cleaned", c="orange")
        axs[1].axhline(y=avg_top_95_1, linestyle="-", label="95% threshold cleaned", c="orange")
        axs[1].axhline(y=avg_top_90_1, linestyle="-", alpha=0.5, label="90% threshold cleaned", c="orange")
        axs[1].legend()
        axs[1].set_title("After preprocessing")

        fig.supylabel("Number of words", y=0.55)
        fig.suptitle("Distribution of text lengths before and after preprocessing", fontsize=16)
        fig.tight_layout()

        plt.savefig("outputs/length_distribution_before_and_after_preprocessing.png")

    def get_table_for_all_families(self):
        """Create and save one table that describes all Families as well as the Tasks and Subtasks per Family."""
        df = self.df[["dataset_id", "dataset_name", "original_size", "final_size", "task_family"]]
        subtasks = list(itertools.chain.from_iterable(t.subtasks_list for t in all_tasks))
        subtasks_df = pd.DataFrame.from_dict(
            {st.id: {"dataset_id": st.task_id, "Subtask": str(st)} for st in subtasks}, orient="index"
        )
        df = df.merge(subtasks_df)
        df["task_family"] = df["task_family"].apply(lambda x: TaskFamilies[x].value)

        df.rename(
            columns={
                "task_family": "Task Family",
                "dataset_name": "Dataset",
                "original_size": "Size",
                "final_size": "Final Size",
            },
            inplace=True,
        )

        df = (
            df.set_index(["Task Family", "Dataset", "Size", "Final Size", "Subtask"])
            .drop(columns=["dataset_id"])
            .sort_index()
        )
        styler = df.style
        styler.format_index(formatter=integer_formatter)

        with open(os.path.join("outputs", "all_families.tex"), "w") as tf:
            latex_code = styler.to_latex(
                caption="All Datasets", label="fig:all_datasets", environment="longtable", **TABLE_CONFIG
            )
            tf.write(latex_code)

    def get_table_for_each_family(self):
        """Create and save one table per Family that describes the Tasks and Subtasks in that Family."""
        df = self.df[["dataset_id", "dataset_name", "original_size", "final_size", "task_family"]]
        subtasks = list(itertools.chain.from_iterable(t.subtasks_list for t in all_tasks))
        subtasks_df = pd.DataFrame.from_dict(
            {st.id: {"dataset_id": st.task_id, "Subtask": str(st)} for st in subtasks}, orient="index"
        )
        df = df.merge(subtasks_df)

        for v in TaskFamilies:
            if v == TaskFamilies.MLM:
                continue
            df_tmp = df[(df[["task_family"]] == v.name).task_family.to_list()]

            df_tmp = df_tmp.rename(
                columns={
                    "dataset_name": "Dataset",
                    "original_size": "Size",
                    "final_size": "Final Size",
                }
            )

            df_tmp = (
                df_tmp.set_index(["Dataset", "Size", "Final Size", "Subtask"])
                .drop(columns=["dataset_id", "task_family"])
                .sort_index()
            )
            styler = df_tmp.style
            styler.format_index(formatter=integer_formatter)

            with open(os.path.join("outputs", f"family_{v.name}.tex"), "w") as tf:
                latex_code = styler.to_latex(
                    caption=f"Family {v.value}", label=f"fig:family_{v.value}", **TABLE_CONFIG
                )
                tf.write(latex_code)


if __name__ == "__main__":
    me = MetadataExplorer()
    me.get_plot_for_text_lengths()
    me.get_table_for_all_families()
    me.get_table_for_each_family()
