"""This module contains the Fetcher for the 108_MeTooMA dataset."""

import os

import pandas as pd

from tweet_fetching.fetcher_blueprint import FetcherBlueprint
from tweet_fetching.twitter.tweetLoader import TweetLoader


class Fetcher108MeTooMA(FetcherBlueprint):
    """Fetcher for the 108_MeTooMA dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Fetcher108MeTooMA."""
        super(Fetcher108MeTooMA, self).__init__(*args, **kwargs)
        self._tweetloader = TweetLoader()

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 108_MeTooMA."""
        df = pd.read_csv(os.path.join(self._raw_data_local_path, "MeTooMA.csv"))
        return df

    def _fetch(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Fetch the raw tweets data of 108_MeTooMA."""
        ids_list = raw_data["TweetId"].tolist()
        original_tweets = self._tweetloader.fetch_list(ids_list=ids_list)

        raw_data.set_index("TweetId", inplace=True)
        original_tweets.set_index("tweetID", inplace=True)

        return raw_data.join(original_tweets, how="right")
