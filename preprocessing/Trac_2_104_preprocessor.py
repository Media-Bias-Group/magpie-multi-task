"""This module contains the Preprocessor for the 104 TRAC-2 dataset."""

import os

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint

MAPPING = {"NGEN": 0, "GEN": 1}


class Preprocessor104TRAC2(PreprocessorBlueprint):
    """Preprocessor for the 31 SUBJ dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor104TRAC2."""
        super(Preprocessor104TRAC2, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: pd.DataFrame, length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 104_TRAC2."""
        self._log_before_preprocessing(data=raw_data)

        # sub-task A is aggression classification
        df = raw_data.drop(["ID", "Sub-task A"], axis=1)
        df.rename(columns={"Sub-task B": "label", "Text": "text"}, inplace=True)
        df["label"] = df["label"].map(MAPPING)

        cleaned = self._clean(df, length_threshold=length_threshold)
        self._log_after_preprocessing(data=cleaned)

        self.set_logger_data("primary_label_distribution", cleaned["label"].value_counts().to_dict())
        self.set_logger_data("additional_data", MAPPING)

        return cleaned

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 104_TRAC2."""
        dev = pd.read_csv(os.path.join(self._raw_data_local_path, "trac2_eng_dev.csv"))
        train = pd.read_csv(os.path.join(self._raw_data_local_path, "trac2_eng_train.csv"))

        return pd.concat([dev, train])
