"""This module contains the Preprocessor for the 86 OffensiveLanguage dataset."""

import os

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint


class Preprocessor86OffensiveLanguage(PreprocessorBlueprint):
    """Preprocessor for the 86_OffensiveLanguage dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor86OffensiveLanguage."""
        super(Preprocessor86OffensiveLanguage, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: pd.DataFrame, length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 86_OffensiveLanguage."""
        self._log_before_preprocessing(data=raw_data)
        df = raw_data[["tweet", "class"]].rename(columns={"tweet": "text", "class": "label"})
        df = self._clean(df=df, length_threshold=length_threshold, rm_hashtag=True)
        self._log_after_preprocessing(data=df)
        self.set_logger_data(key="primary_label_distribution", value=df.label.value_counts().to_dict())

        return df

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 86_OffensiveLanguage."""
        df = pd.read_csv(os.path.join(self._raw_data_local_path, "data", "labeled_data.csv"))
        return df
