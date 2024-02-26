"""This module contains the Fetcher for the 127_VaccineLies dataset."""
import os

import pandas as pd

from tweet_fetching.fetcher_blueprint import FetcherBlueprint
from tweet_fetching.twitter.tweetLoader import TweetLoader


class Fetcher127VaccineLies(FetcherBlueprint):
    """Fetcher for the 127_VaccineLies dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Fetcher127VaccineLies."""
        super(Fetcher127VaccineLies, self).__init__(*args, **kwargs)
        self._tweetloader = TweetLoader()

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 127_VaccineLies."""
        # COVID
        train_cov = pd.read_json(
            os.path.join(self._raw_data_local_path, "covid19/annotations/train.jsonl"), lines=True
        )
        dev_cov = pd.read_json(os.path.join(self._raw_data_local_path, "covid19/annotations/dev.jsonl"), lines=True)
        test_cov = pd.read_json(os.path.join(self._raw_data_local_path, "covid19/annotations/test.jsonl"), lines=True)

        # HPV
        train_hpv = pd.read_json(os.path.join(self._raw_data_local_path, "hpv/annotations/train.jsonl"), lines=True)
        dev_hpv = pd.read_json(os.path.join(self._raw_data_local_path, "hpv/annotations/dev.jsonl"), lines=True)
        test_hpv = pd.read_json(os.path.join(self._raw_data_local_path, "hpv/annotations/test.jsonl"), lines=True)

        return pd.concat([train_cov, dev_cov, test_cov, train_hpv, dev_hpv, test_hpv])

    def _fetch(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Fetch the raw tweets data of 127_VaccineLies."""
        ids_list = raw_data["id"].tolist()
        original_tweets = self._tweetloader.fetch_list(ids_list=ids_list)  # 10.380

        raw_data.set_index("id", inplace=True)
        raw_data.index = raw_data.index.astype(int)
        original_tweets.set_index("tweetID", inplace=True)

        return raw_data.join(original_tweets, how="right").reset_index()  # 9474
