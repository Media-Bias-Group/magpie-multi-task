"""This module contains the Preprocessor for the 84 Emotion Tweets dataset."""

import os

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint

MAPPING = dict(zip(["Anger", "Anticipation", "Disgust", "Fear", "Joy", "Sadness", "Surprise", "Truest"], range(7)))


class Preprocessor84EmotionTweets(PreprocessorBlueprint):
    """Preprocessor for the 84 Emotion Tweets dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor107CMSB."""
        super(Preprocessor84EmotionTweets, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: pd.DataFrame, length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 84_EmotionTweets."""
        self._log_before_preprocessing(data=raw_data)
        raw_data["label"] = raw_data["label"].apply(lambda x: x - 1)
        cleaned = self._clean(raw_data, length_threshold=length_threshold)
        self._log_after_preprocessing(data=cleaned)
        self.set_logger_data("primary_label_distribution", cleaned["label"].value_counts().to_dict())
        self.set_logger_data("additional_data", MAPPING)

        return cleaned

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 84_EmotionTweets."""
        data = pd.read_csv(os.path.join(self._raw_data_local_path, "fetched.csv"))
        return data
