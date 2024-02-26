"""This module contains the Fetcher for the 125_MultiTargetStance dataset."""

import os

import pandas as pd

from tweet_fetching.fetcher_blueprint import FetcherBlueprint
from tweet_fetching.twitter.tweetLoader import TweetLoader


class Fetcher125MultiTargetStance(FetcherBlueprint):
    """Fetcher for the 125_MultiTargetStance dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Fetcher125MultiTargetStance."""
        super(Fetcher125MultiTargetStance, self).__init__(*args, **kwargs)
        self._tweetloader = TweetLoader()

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 125_MultiTargetStance."""
        df = pd.read_csv(os.path.join(self._raw_data_local_path, "train.csv"))
        return df

    def _fetch(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Fetch the raw tweets data of 125_MultiTargetStance."""
        ids_list = raw_data["Tweet_ID"].tolist()
        original_tweets = self._tweetloader.fetch_list(ids_list=ids_list)

        raw_data.set_index("Tweet_ID", inplace=True)
        original_tweets.set_index("tweetID", inplace=True)

        return raw_data.join(original_tweets, how="right")
