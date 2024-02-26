"""This module contains the Preprocessor for 118_WizardsOfWikipedia dataset."""

import os

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint

MAPPING = {"unknown": 0, "w": 1, "m": 2}


class Preprocessor118WizardsOfWikipedia(PreprocessorBlueprint):
    """Preprocessor for the Preprocessor118WizardsOfWikipedia dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor118WizardsOfWikipedia."""
        super(Preprocessor118WizardsOfWikipedia, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: pd.DataFrame, length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 118_WizardsOfWikipedia."""
        self._log_before_preprocessing(data=raw_data)
        df = raw_data[["text", "gender"]].rename(columns={"gender": "label"})
        df = self._clean(df=df, length_threshold=length_threshold)
        self._log_after_preprocessing(data=df)
        self.set_logger_data(key="primary_label_distribution", value=df.label.value_counts().to_dict())
        self.set_logger_data(key="additional_data", value={"MAPPING": MAPPING})
        return df

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 118_WizardsOfWikipedia."""
        df = pd.read_csv(os.path.join(self._raw_data_local_path, "wizard.csv"))
        return df
