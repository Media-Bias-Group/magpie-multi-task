"""This module contains the Preprocessor for the 88 HatespeechTwitter dataset."""

import os

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint

MAPPING = {"normal": 0, "spam": 1, "abusive": 2, "hateful": 3}


class Preprocessor88HatespeechTwitter(PreprocessorBlueprint):
    """Preprocessor for the 88 HatespeechTwitter dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor88HatespeechTwitter."""
        super(Preprocessor88HatespeechTwitter, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: pd.DataFrame, length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 88_HatespeechTwitter."""
        self._log_before_preprocessing(data=raw_data)

        raw_data["label"] = raw_data["label"].map(MAPPING)
        raw_data["text"] = raw_data["text"].astype(str)

        cleaned = self._clean(raw_data, length_threshold=length_threshold)
        self._log_after_preprocessing(data=cleaned)
        self.set_logger_data("primary_label_distribution", cleaned["label"].value_counts().to_dict())
        self.set_logger_data("additional_data", MAPPING)

        return cleaned

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 88_HatespeechTwitter."""
        # load train, test, dev files
        df = pd.read_csv(os.path.join(self._raw_data_local_path, "fetched.csv"), engine="python")

        return df
