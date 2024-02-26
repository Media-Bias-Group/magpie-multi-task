"""This module contains the Preprocessor for the 116_MDGender dataset."""

import os

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint

MAPPING = dict(zip(["[0]", "[1]", "[2]", "[3]", "[4]", "[5]"], range(6)))
MAPPING_LABELS = dict(zip(["about_w", "about_m", "to_w", "to_m", "as_w", "as_m"], range(6)))


class Preprocessor116MDGender(PreprocessorBlueprint):
    """Preprocessor for the 116_MDGender dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor116MDGender."""
        super(Preprocessor116MDGender, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: pd.DataFrame, length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 116_MDGender."""
        self._log_before_preprocessing(data=raw_data)

        df = raw_data[["text", "labels"]].rename(columns={"labels": "label"})
        df["label"] = df.label.map(MAPPING)

        df = self._clean(df=df, length_threshold=length_threshold)
        self._log_after_preprocessing(data=df)
        self.set_logger_data(key="primary_label_distribution", value=df.label.value_counts().to_dict())
        self.set_logger_data(
            key="additional_data",
            value={"MAPPING": MAPPING_LABELS},
        )

        return df

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 116_MDGender."""
        df = pd.read_csv(
            os.path.join(self._raw_data_local_path, "MDGender.csv"),
            usecols=["text", "original", "labels", "class_type", "turker_gender", "confidence"],
        )
        return df
