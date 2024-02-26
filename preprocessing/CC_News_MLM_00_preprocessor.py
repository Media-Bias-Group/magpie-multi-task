"""This module contains the Preprocessor for the 00 CC_NEWS_MLM dataset."""

import os

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint


class PreprocessorCCNewsMLM(PreprocessorBlueprint):
    """Preprocessor for the 00_CC_NEWS_MLM dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a PreprocessorCCNewsMLM."""
        super(PreprocessorCCNewsMLM, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: pd.DataFrame, length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 00_CC_NEWS_ML."""
        self._log_before_preprocessing(data=raw_data)
        df = self._clean(df=raw_data, length_threshold=length_threshold)
        self._log_after_preprocessing(data=df)
        return df

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 00_CC_NEWS_MLM."""
        df = pd.read_csv(os.path.join(self._raw_data_local_path, "cc-news.csv"),usecols=['text'])
        return df
