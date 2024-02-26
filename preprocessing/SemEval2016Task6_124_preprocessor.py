"""This module contains the Preprocessor for the 124_SemEval2016Task6 dataset."""

import os
from typing import Tuple

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint

MAPPING = dict(zip(["AGAINST", "NONE", "FAVOR"], range(3)))


class Preprocessor124SemEval2016Task6(PreprocessorBlueprint):
    """Preprocessor for the 124_SemEval2016Task6 dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor124SemEval2016Task6."""
        super(Preprocessor124SemEval2016Task6, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: Tuple[pd.DataFrame, pd.DataFrame], length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 124_SemEval2016Task6."""
        self._log_before_preprocessing(data=raw_data)

        df_train, df_test = raw_data
        df = pd.concat([df_train, df_test], axis=0)
        df = df[["Tweet", "Target", "Stance"]].rename(
            columns={"Tweet": "text", "Stance": "label_stance", "Target": "label_target"}
        )
        df["label_stance"] = df.label_stance.map(MAPPING)
        label_target_distribution = df.label_target.value_counts().to_dict()

        # We do not discard hashtags as they convey lots of useful information.

        df["text"] = df["label_target"] + ". " + df.text
        df = df[["text", "label_stance"]].rename(columns={"label_stance": "label"})

        df = self._clean(df=df, length_threshold=length_threshold)
        self._log_after_preprocessing(data=df)

        self.set_logger_data(key="primary_label_distribution", value=df.label.value_counts().to_dict())
        self.set_logger_data(
            key="additional_data",
            value={"MAPPING": MAPPING, "label_target_distribution": label_target_distribution},
        )
        return df

    def _load_raw_data_from_local(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load the raw data of 124_SemEval2016Task6."""
        with open(os.path.join(self._raw_data_local_path, "train.csv"), errors="replace") as f:
            df_train = pd.read_csv(f)

        with open(os.path.join(self._raw_data_local_path, "test.csv"), errors="replace") as f:
            df_test = pd.read_csv(f)

        return df_train, df_test

    def _log_before_preprocessing(self, data: Tuple[pd.DataFrame, pd.DataFrame]):
        self.set_logger_data(key="original_size", value=len(data[0]) + len(data[1]))
