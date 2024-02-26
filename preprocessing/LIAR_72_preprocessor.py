"""This module contains the Preprocessor for the 72 LIAR dataset."""

import os

import numpy as np
import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint

COUNT_COLUMNS = ["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"]
MAPPING = dict(zip(COUNT_COLUMNS, list(np.linspace(0, 1, len(COUNT_COLUMNS)))))


class Preprocessor72LIAR(PreprocessorBlueprint):
    """Preprocessor for the 72_LIAR dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor72LIAR."""
        super(Preprocessor72LIAR, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: pd.DataFrame, length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 72_LIAR."""
        self._log_before_preprocessing(data=raw_data)

        raw_data["label"] = raw_data["label"].map(MAPPING)
        raw_data["label_binary"] = raw_data["label"].apply(lambda x: int(np.round(x)))  # np.round(0.5) := 0
        df = raw_data[["statement", "label", "label_binary"]].rename(columns={"statement": "text"})

        df = self._clean(df=df, length_threshold=length_threshold)
        self._log_after_preprocessing(data=df)
        self.set_logger_data(key="primary_label_distribution", value=df.label.value_counts().to_dict())
        self.set_logger_data(key="additional_data", value={"label_binary": df.label_binary.value_counts().to_dict()})

        return df

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 72_LIAR."""
        columns = [
            "id",
            "label",
            "statement",
            "subject_s",
            "speaker",
            "speaker_job_title",
            "state_info",
            "party_affiliation",
            "barely_true_counts",
            "false_counts",
            "half_true_counts",
            "mostly_true_counts",
            "pants_on_fire_counts",
            "context",
        ]
        df_train = pd.read_csv(os.path.join(self._raw_data_local_path, "train.tsv"), sep="\t", header=None)
        df_valid = pd.read_csv(os.path.join(self._raw_data_local_path, "valid.tsv"), sep="\t", header=None)
        df_test = pd.read_csv(os.path.join(self._raw_data_local_path, "test.tsv"), sep="\t", header=None)
        df_train.columns = columns
        df_valid.columns = columns
        df_test.columns = columns
        df = pd.concat([df_train, df_valid, df_test], axis=0)
        return df
