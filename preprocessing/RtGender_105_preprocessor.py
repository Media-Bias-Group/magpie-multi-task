"""This module contains the Preprocessor for the 105_RtGender dataset."""

import os
import random
from typing import List

import numpy as np
import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint

MAPPING = {"M": 0, "W": 1}


class Preprocessor105RtGender(PreprocessorBlueprint):
    """Preprocessor for the 105_RtGender dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor105RtGender."""
        super(Preprocessor105RtGender, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: List[pd.DataFrame], length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 105_RtGender."""
        for i, t in enumerate(raw_data):
            t["op_gender"] = t["op_gender"].map(MAPPING).astype(int)
            t.set_index("post_id", inplace=True)
            raw_data[i] = t

        np.random.seed(42)
        df_facebook_wiki = raw_data[0].dropna().groupby("post_id").agg(np.random.choice)
        df_facebook_congress = raw_data[1].dropna().groupby("post_id").agg(np.random.choice)
        df_fitocracy = raw_data[2].dropna().groupby("post_id").agg(np.random.choice)

        df = pd.concat([df_facebook_wiki[:10000], df_facebook_congress[:10000], df_fitocracy[:10000]], axis=0)
        df.rename(
            columns={
                "op_gender": "label",
                "response_text": "text",
            },
            inplace=True,
        )
        df = self._clean(df=df, length_threshold=length_threshold)
        self._log_after_preprocessing(data=df)
        self.set_logger_data(key="primary_label_distribution", value=df.label.value_counts().to_dict())
        self.set_logger_data(
            key="additional_data",
            value={
                "MAPPING": MAPPING,
            },
        )

        return df

    def _load_raw_data_from_local(
        self,
    ) -> List[pd.DataFrame]:
        """Load the raw data of 105_RtGender."""
        p = 0.003
        random.seed(42)
        facebook_wiki_responses_filename = os.path.join(self._raw_data_local_path, "facebook_wiki_responses.csv")
        facebook_congress_responses_filename = os.path.join(
            self._raw_data_local_path, "facebook_congress_responses.csv"
        )
        fitocracy_filename = os.path.join(self._raw_data_local_path, "fitocracy_responses.csv")

        facebook_wiki_responses = pd.read_csv(
            facebook_wiki_responses_filename,
            usecols=["op_gender", "post_id", "response_text"],
            header=0,
            skiprows=lambda i: i > 0 and random.random() > p,
        )
        facebook_congress_responses = pd.read_csv(
            facebook_congress_responses_filename,
            usecols=["op_gender", "post_id", "response_text"],
            header=0,
            skiprows=lambda i: i > 0 and random.random() > p / 2,
        )
        fitocracy_responses = pd.read_csv(
            fitocracy_filename,
            usecols=["op_gender", "post_id", "response_text"],
            header=0,
            skiprows=lambda i: i > 0 and random.random() > p * 30,
        )

        counter = 0
        with open(facebook_wiki_responses_filename) as f:
            c = sum(1 for line in f)
            counter += c

        with open(facebook_congress_responses_filename) as f:
            c = sum(1 for line in f)
            counter += c

        with open(fitocracy_filename) as f:
            c = sum(1 for line in f)
            counter += c

        self._log_before_preprocessing(data=counter)

        return [
            facebook_wiki_responses,
            facebook_congress_responses,
            fitocracy_responses,
        ]

    def _log_before_preprocessing(self, data: int):
        self.set_logger_data(key="original_size", value=data)
