"""This module contains the Preprocessor for the 119_SemEval2023Task4 dataset."""

import os
from typing import Tuple

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint

MAPPING = {"in favor of": 1, "against": 0}


class Preprocessor119SemEval2023Task4(PreprocessorBlueprint):
    """Preprocessor for the 119_SemEval2023Task4 dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor119SemEval2023Task4."""
        super(Preprocessor119SemEval2023Task4, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: Tuple[pd.DataFrame, pd.DataFrame], length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 119_SemEval2023Task4."""
        # Regarding values: labels_training_df.drop(columns=["Argument ID"]).sum(axis=1) shows that we have a multi-label problem
        # In the future and for another project we might want to extract the value aswell. Not important for our
        # stance-detection task.
        self._log_before_preprocessing(data=raw_data[0])
        arguments_training_df, labels_training_df = raw_data
        arguments_training_df.set_index("Argument ID", inplace=True)
        labels_training_df.set_index("Argument ID", inplace=True)
        df = arguments_training_df.join(labels_training_df, how="left")

        df["Stance"] = df.Stance.map(MAPPING)
        df["text"] = df["Conclusion"] + [". "] * len(df) + df["Premise"]
        df.rename(columns={"Stance": "label"}, inplace=True)
        df = df[["text", "label"]]
        df = self._clean(df=df, length_threshold=length_threshold)
        self._log_after_preprocessing(data=df)

        self.set_logger_data(key="primary_label_distribution", value=df.label.value_counts().to_dict())
        self.set_logger_data(
            key="additional_data",
            value={
                "MAPPING": MAPPING,
            },
        )

        return df

    def _load_raw_data_from_local(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load the raw data of 119_SemEval2023Task4."""
        arguments_training_df = pd.read_csv(
            os.path.join(self._raw_data_local_path, "arguments-training.tsv"), sep="\t"
        )
        labels_training_df = pd.read_csv(os.path.join(self._raw_data_local_path, "labels-training.tsv"), sep="\t")
        return arguments_training_df, labels_training_df
