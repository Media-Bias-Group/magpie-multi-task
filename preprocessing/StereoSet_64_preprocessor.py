"""This module contains the Preprocessor for the 64 StereoSet dataset."""

import json
import os

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint

MAPPING = dict(zip(["race", "profession", "gender", "religion"], range(4)))


class Preprocessor64StereoSet(PreprocessorBlueprint):
    """Preprocessor for the 64_StereoSet dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor64StereoSet."""
        super(Preprocessor64StereoSet, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: pd.DataFrame, length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 64_StereoSet."""
        self._log_before_preprocessing(data=raw_data)
        stereotype = (raw_data[["bias_type", "stereotype"]]).rename(columns={"stereotype": "text"})
        stereotype["label"] = 1
        anti_stereotype = (raw_data[["bias_type", "anti-stereotype"]]).rename(columns={"anti-stereotype": "text"})
        anti_stereotype["label"] = 0
        df = pd.concat([stereotype, anti_stereotype], axis=0)
        df["bias_type"] = df.bias_type.map(MAPPING)
        df.rename(columns={"bias_type": "stereotype_label"}, inplace=True)

        df = self._clean(df=df, length_threshold=length_threshold)
        self._log_after_preprocessing(data=df)
        self.set_logger_data(key="primary_label_distribution", value=df.label.value_counts().to_dict())
        self.set_logger_data(
            key="additional_data", value={"stereotype_label": df.stereotype_label.value_counts().to_dict()}
        )

        return df

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 64_StereoSet."""
        with open(os.path.join(self._raw_data_local_path, "data", "processed_stereoset.json")) as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return df
