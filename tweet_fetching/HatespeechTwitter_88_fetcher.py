"""This module contains the Fetcher for the 88 HatespeechTWitter dataset."""

import os

import pandas as pd

from tweet_fetching.fetcher_blueprint import FetcherBlueprint
from tweet_fetching.twitter.tweetLoader import TweetLoader


class Fetcher88HatespeechTwitter(FetcherBlueprint):
    """Fetcher for the 88 HatespeechTwitter dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Fetcher88HatespeechTwitter."""
        super(Fetcher88HatespeechTwitter, self).__init__(*args, **kwargs)
        self._tweetloader = TweetLoader()

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 88 HatespeechTwitter."""
        df = pd.read_csv(os.path.join(self._raw_data_local_path, "hatespeech_labels.csv"))
        return df

    def _fetch(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Fetch the raw tweets data of 88 HatespeechTwitter."""
        ids_list = raw_data["tweet_id"].tolist()
        original_tweets = self._tweetloader.fetch_list(ids_list=ids_list)

        raw_data.set_index("tweet_id", inplace=True)
        original_tweets.set_index("tweetID", inplace=True)

        return raw_data.join(original_tweets, how="right")
