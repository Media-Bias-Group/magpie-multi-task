"""This module contains the Preprocessor for the 99 Stanford Sentiment dataset."""

import os

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint

MAPPING = {"positive": 1, "negative": 0}


class Preprocessor99StanfordSentiment(PreprocessorBlueprint):
    """Preprocessor for the 99 Stanford Sentiment dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a 99_SST2."""
        super(Preprocessor99StanfordSentiment, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: pd.DataFrame, length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 99_SST2."""
        self._log_before_preprocessing(data=raw_data)
        df = self._clean(df=raw_data, length_threshold=length_threshold)
        self._log_after_preprocessing(data=df)
        self.set_logger_data(key="primary_label_distribution", value=raw_data["label"].value_counts().to_dict())
        self.set_logger_data(key="additional_data", value={"MAPPING": MAPPING})
        return df

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 99_SST2."""
        # load train, test, dev files
        train_file = os.path.join(self._raw_data_local_path, "train.csv")
        df_train = pd.read_csv(train_file)

        test_file = os.path.join(self._raw_data_local_path, "test.csv")
        df_test = pd.read_csv(test_file)

        dev_file = os.path.join(self._raw_data_local_path, "dev.csv")
        df_dev = pd.read_csv(dev_file)

        df = pd.concat([df_train, df_test, df_dev])

        return df
