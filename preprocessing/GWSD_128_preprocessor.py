"""This module contains the Preprocessor for the 128 GWSD dataset."""

import os

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint

MAPPING = {"neutral": 0, "agree": 1, "disagree":2}


class Preprocessor128GWSD(PreprocessorBlueprint):
    """Preprocessor for the 128 GWSD dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor128GWSD."""
        super(Preprocessor128GWSD, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: pd.DataFrame, length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 128_GWSD."""
        self._log_before_preprocessing(data=raw_data)

        df = raw_data
        df['final_label_prob'] = df[['neutral','agree','disagree']].max(axis=1)
        #drop those that final label probability is below 0.5
        df=df[df.final_label_prob > 0.5]
        df['label'] = df[['neutral','agree','disagree']].idxmax(axis=1)

        df.rename(columns={"sentence": "text"}, inplace=True)
        df["label"] = df["label"].map(MAPPING)
        df = df[['text','label']]

        cleaned = self._clean(df, length_threshold=length_threshold)
        self._log_after_preprocessing(data=cleaned)
        self.set_logger_data("primary_label_distribution", cleaned["label"].value_counts().to_dict())
        self.set_logger_data("additional_data", MAPPING)

        return cleaned

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 128_GWSD."""
        df = pd.read_csv(
            os.path.join(self._raw_data_local_path, "gwsd.tsv"),
            sep='\t',
            usecols=["sentence", "disagree", "agree","neutral","in_held_out_test"],
            on_bad_lines='skip'
        )

        return df
