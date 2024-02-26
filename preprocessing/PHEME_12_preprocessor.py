"""This module contains the Preprocessor for the 12_PHEME dataset."""

import json
import os
from typing import Dict, Tuple
from zipfile import ZipFile

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint


class Preprocessor12PHEME(PreprocessorBlueprint):
    """Preprocessor for the 12_PHEME dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor12PHEME."""
        super(Preprocessor12PHEME, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: Tuple[pd.DataFrame, Dict], length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 12_PHEME."""
        raw_tweets, annotations = raw_data
        self._log_before_preprocessing(data=raw_tweets)

        def get_veracity(annotation):
            """
            Retrieve the veracity for a tweet from its annotation.

            :return: 0 if False, 1 if True, 2 if annotation is not available.
            """
            veracity = annotation.get("true")
            return 0 if (veracity == 0 or veracity == str(0)) else 1 if veracity == 1 or veracity == str(1) else 2

        rumours = raw_tweets[raw_tweets["label"] == 1]
        non_rumours = raw_tweets[raw_tweets["label"] == 0]
        rumours["veracity"] = rumours.apply(lambda x: get_veracity(annotation=annotations[x.tweetID]), axis=1)
        non_rumours["veracity"] = 1  # 1 := true

        df = pd.concat([rumours, non_rumours], axis=0)
        df.rename(columns={"veracity": "veracity_label"}, inplace=True)
        df.drop(columns=["tweetID", "topic"], inplace=True)

        df = self._clean(df=df, length_threshold=length_threshold)
        self._log_after_preprocessing(data=df)
        self.set_logger_data(key="primary_label_distribution", value=df.label.value_counts().to_dict())
        self.set_logger_data(
            key="additional_data",
            value={
                "veracity_label_distribution": df.veracity_label.value_counts().to_dict(),
            },
        )
        return df

    def _load_raw_data_from_local(self) -> Tuple[pd.DataFrame, Dict]:
        """Load the raw data of 12_PHEME."""
        with ZipFile(os.path.join(self._raw_data_local_path, "12_PHEME.zip"), "r") as zipObj:
            zipObj.extractall(path=self._raw_data_local_path)

        raw_tweets = pd.read_csv(os.path.join(self._raw_data_local_path, "fetched.csv"))
        annotations = {}
        rumours = raw_tweets[raw_tweets["label"] == 1]
        for _, row in rumours.iterrows():
            path = os.path.join(
                self._raw_data_local_path,
                "12_PHEME",
                "PHEME_veracity",
                "all-rnr-annotated-threads",
                row["topic"],
                "rumours",
                str(row["tweetID"]),
                "annotation.json",
            )
            with open(
                path,
                "r",
            ) as f:
                annotation = json.load(f)
                annotations[row["tweetID"]] = annotation

        return raw_tweets, annotations
