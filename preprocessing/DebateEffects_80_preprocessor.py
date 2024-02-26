"""This module contains the Preprocessor for the 80 DebateEffects dataset."""

import os

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint


class Preprocessor80DebateEffects(PreprocessorBlueprint):
    """Preprocessor for the 80 DebateEffects dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a 80_DebateEffects."""
        super(Preprocessor80DebateEffects, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: pd.DataFrame, length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 80_DebateEffects."""
        self._log_before_preprocessing(data=raw_data)
        df = raw_data[["response", "fact-feeling", "fact-feeling_unsure"]].rename(
            columns={
                "fact-feeling": "label",
                "fact-feeling_unsure": "uncertain_fact_feeling",
                "response": "text",
            },
        )

        df = self._clean(df=df, length_threshold=length_threshold)
        self._log_after_preprocessing(data=df)
        self.set_logger_data(key="primary_label_distribution", value=df.label.describe().to_dict())
        self.set_logger_data(
            key="additional_data",
            value={
                "uncertain_fact_feeling": df.uncertain_fact_feeling.describe().to_dict(),
            },
        )
        return df

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 80_DebateEffects."""
        df = pd.read_csv(
            os.path.join(self._raw_data_local_path, "IAC_clean.csv"),
        )

        return df
