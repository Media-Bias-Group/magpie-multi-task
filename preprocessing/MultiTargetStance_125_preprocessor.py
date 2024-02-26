"""This module contains the Preprocessor for the 125 MultiTargetStance dataset."""

import os

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint

MAPPING = dict(zip(["AGAINST", "NONE", "FAVOR"], range(3)))


class Preprocessor125MultiTargetStance(PreprocessorBlueprint):
    """Preprocessor for the 125_MultiTargetStance dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor109Stereotype."""
        super(Preprocessor125MultiTargetStance, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: pd.DataFrame, length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 125_MultiTargetStance."""
        t1 = raw_data[[" Target1", " Stance1", "text"]]
        t2 = raw_data[[" Target2", " Stance2", "text"]]

        t1 = t1.rename(columns={" Target1": "target", " Stance1": "label"})
        t2 = t2.rename(columns={" Target2": "target", " Stance2": "label"})

        df = pd.concat([t1, t2])
        df["label"] = df["label"].map(MAPPING)

        df["text"] = df["target"] + [". "] * len(df) + df["text"]
        df.drop(columns=["target"], inplace=True, axis=0)

        self._log_before_preprocessing(data=df)
        cleaned = self._clean(df, length_threshold=length_threshold)
        self._log_after_preprocessing(data=cleaned)
        self.set_logger_data("primary_label_distribution", cleaned["label"].value_counts().to_dict())
        self.set_logger_data("additional_data", MAPPING)

        return cleaned

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 125_MultiTargetStance."""
        df = pd.read_csv(
            os.path.join(self._raw_data_local_path, "fetched.csv"),
            usecols=[" Target1", " Stance1", " Target2", " Stance2", "text"],
        )
        return df
