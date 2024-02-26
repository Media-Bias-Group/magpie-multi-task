"""This module contains the Preprocessor for the 10 BABE dataset."""

import os
from ast import literal_eval

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint

MAPPING = {"Non-biased": 0, "Biased": 1}


class Preprocessor10BABE(PreprocessorBlueprint):
    """Preprocessor for the 31 SUBJ dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor10BABE."""
        super(Preprocessor10BABE, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: pd.DataFrame, length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 10_BABE."""
        self._log_before_preprocessing(data=raw_data)

        df = raw_data[raw_data["label_bias"] != "No agreement"]
        df.rename(columns={"label_bias": "label"}, inplace=True)
        df["label"] = df["label"].map(MAPPING)

        df["biased_words"] = df["biased_words"].apply(
            lambda pos_list: ";".join([self._unify_text(text=x, rm_hashtag=False) for x in pos_list if ";" not in x])
        )

        cleaned = self._clean(df, length_threshold=length_threshold)
        self._log_after_preprocessing(data=cleaned)
        self.set_logger_data("primary_label_distribution", cleaned["label"].value_counts().to_dict())
        self.set_logger_data("additional_data", MAPPING)

        return cleaned

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 10_BABE."""
        df = pd.read_csv(
            os.path.join(self._raw_data_local_path, "babe.csv"),
            usecols=["text", "label_bias", "biased_words"],
            converters={"biased_words": literal_eval},
        )

        return df
