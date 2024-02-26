"""This module contains the Preprocessor for the 19 MultiDimNews dataset."""

import os
from typing import Tuple

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint


class Preprocessor19MultiDimNews(PreprocessorBlueprint):
    """Preprocessor for the 19 MultiDimNews dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor19MultiDimNews."""
        super(Preprocessor19MultiDimNews, self).__init__(*args, **kwargs)

    def _preprocess(
        self, raw_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], length_threshold: int
    ) -> pd.DataFrame:
        """Preprocess the raw data of 19_MultiDimNews."""
        df_bias, df_framing, df_hidden_assumpt, df_subj = raw_data
        self._log_before_preprocessing(data=df_bias)

        # We manually checked that in each df the rows contain the same sentences
        df_framing.drop("text", inplace=True, axis=1)
        df_hidden_assumpt.drop("text", inplace=True, axis=1)
        df_subj.drop("text", inplace=True, axis=1)

        df = pd.concat([df_bias, df_framing, df_hidden_assumpt, df_subj], axis=1)
        df.fillna(0)

        df = self._clean(df=df, length_threshold=length_threshold)
        self._log_after_preprocessing(data=df)
        self.set_logger_data(key="primary_label_distribution", value=df.label_bias.value_counts().to_dict())
        self.set_logger_data(
            key="additional_data",
            value={
                "label_framing_distribution": df.label_framing.value_counts().to_dict(),
                "label_hidden_assumpt_distribution": df.label_hidden_assumpt.value_counts().to_dict(),
                "label_subj": df.label_subj.value_counts().to_dict(),
            },
        )
        return df

    def _load_raw_data_from_local(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the raw data of 19_MultiDimNews."""
        df_bias = pd.read_csv(
            os.path.join(self._raw_data_local_path, "sentences-with-binary-labels-bias.csv"),
            names=["text", "label_bias"],
        )
        df_framing = pd.read_csv(
            os.path.join(self._raw_data_local_path, "sentences-with-binary-labels-framing.csv"),
            names=["text", "label_framing"],
        )
        df_hidden_assumpt = pd.read_csv(
            os.path.join(self._raw_data_local_path, "sentences-with-binary-labels-hidden-assumpt.csv"),
            names=["text", "label_hidden_assumpt"],
        )
        df_subj = pd.read_csv(
            os.path.join(self._raw_data_local_path, "sentences-with-binary-labels-subj.csv"),
            names=["text", "label_subj"],
        )

        return df_bias, df_framing, df_hidden_assumpt, df_subj
