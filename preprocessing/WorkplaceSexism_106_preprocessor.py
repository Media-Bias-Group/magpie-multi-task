"""This module contains the Preprocessor for the 106 Workplace Sexism dataset."""

import os

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint


class Preprocessor106WorkplaceSexism(PreprocessorBlueprint):
    """Preprocessor for the 106 Workplace Sexism dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor106WorkplaceSexism."""
        super(Preprocessor106WorkplaceSexism, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: pd.DataFrame, length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 106_WorkplaceSexism."""
        self._log_before_preprocessing(data=raw_data)
        cleaned = self._clean(raw_data, length_threshold=length_threshold)
        self._log_after_preprocessing(data=cleaned)
        self.set_logger_data("primary_label_distribution", cleaned["label"].value_counts().to_dict())

        return cleaned

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 106_WorkplaceSexism."""
        df = pd.read_csv(os.path.join(self._raw_data_local_path, "SD_dataset_FINAL.csv"), names=["text", "label"])
        return df
