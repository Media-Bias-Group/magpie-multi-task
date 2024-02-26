"""This module contains the Preprocessor for the 87 OnlineHarassmentDataset."""

import os

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint

MAPPING = {"H": 1, "N": 0}


class Preprocessor87OnlineHarassmentDataset(PreprocessorBlueprint):
    """Preprocessor for the 87_OnlineHarassmentDataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor87OnlineHarassmentDataset."""
        super(Preprocessor87OnlineHarassmentDataset, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: pd.DataFrame, length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 87_OnlineHarassmentDataset."""
        self._log_before_preprocessing(data=raw_data)

        df = raw_data.drop(columns=["ID"])
        df.rename(columns={"Code": "label", "Tweet": "text"}, inplace=True)
        df["label"] = df["label"].map(MAPPING)

        df = self._clean(df=df, length_threshold=length_threshold)
        self._log_after_preprocessing(data=df)
        self.set_logger_data(key="primary_label_distribution", value=df.label.value_counts().to_dict())

        return df

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 87_OnlineHarassmentDataset."""
        df = pd.read_csv(os.path.join(self._raw_data_local_path, "hatespeech_golbeck.csv"))
        return df
