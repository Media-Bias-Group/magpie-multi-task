"""This module contains the Preprocessor for the 25 Fake News dataset."""

import os
from typing import Tuple

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint


class Preprocessor25FakeNewsNet(PreprocessorBlueprint):
    """Preprocessor for the 25 Fake News dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor25FakeNewsNet."""
        super(Preprocessor25FakeNewsNet, self).__init__(*args, **kwargs)

    def _preprocess(
        self, raw_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], length_threshold: int
    ) -> pd.DataFrame:
        """Preprocess the raw data of 25_FakeNews."""
        gossip_fake, gossip_real, poli_fake, poli_real = raw_data

        # drop everything except titles
        gossip_fake.drop(["id", "news_url", "tweet_ids"], axis=1, inplace=True)
        gossip_real.drop(["id", "news_url", "tweet_ids"], axis=1, inplace=True)
        poli_fake.drop(["id", "news_url", "tweet_ids"], axis=1, inplace=True)
        poli_real.drop(["id", "news_url", "tweet_ids"], axis=1, inplace=True)

        gossip_fake_df = pd.DataFrame({"text": gossip_fake["title"].to_list(), "label": [1] * len(gossip_fake)})
        gossip_real_df = pd.DataFrame({"text": gossip_real["title"].to_list(), "label": [0] * len(gossip_real)})
        poli_fake_df = pd.DataFrame({"text": poli_fake["title"].to_list(), "label": [1] * len(poli_fake)})
        poli_real_df = pd.DataFrame({"text": poli_real["title"].to_list(), "label": [0] * len(poli_real)})

        data = pd.concat([gossip_fake_df, gossip_real_df, poli_fake_df, poli_real_df])

        self._log_before_preprocessing(data=data)

        cleaned = self._clean(data, length_threshold=length_threshold)
        self._log_after_preprocessing(data=cleaned)
        self.set_logger_data("primary_label_distribution", cleaned["label"].value_counts().to_dict())

        return cleaned

    def _load_raw_data_from_local(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the raw data of 25_FakeNews."""
        # load real,fake data from gossip and politifact
        gossip_fake = pd.read_csv(os.path.join(self._raw_data_local_path, "gossipcop_fake.csv"))
        gossip_real = pd.read_csv(os.path.join(self._raw_data_local_path, "gossipcop_real.csv"))
        poli_fake = pd.read_csv(os.path.join(self._raw_data_local_path, "politifact_fake.csv"))
        poli_real = pd.read_csv(os.path.join(self._raw_data_local_path, "politifact_real.csv"))

        return gossip_fake, gossip_real, poli_fake, poli_real
