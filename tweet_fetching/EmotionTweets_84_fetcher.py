"""This module contains the Fetcher for the 84 EmotionTweets dataset."""

import os

import pandas as pd

from tweet_fetching.fetcher_blueprint import FetcherBlueprint
from tweet_fetching.twitter.tweetLoader import TweetLoader


class Fetcher84EmotionTweets(FetcherBlueprint):
    """Fetcher for the 84 EmotionTweets dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Fetcher84EmotionTweets."""
        super(Fetcher84EmotionTweets, self).__init__(*args, **kwargs)
        self._tweetloader = TweetLoader()

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 84_EmotionTweets."""
        data = pd.read_csv(os.path.join(self._raw_data_local_path, "KAGGLE - Copy.csv"), sep=";")
        return data

    def _fetch(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Fetch the raw tweets data of 84_EmotionTweets."""
        data = raw_data[:300000]
        fetched = self._tweetloader.fetch_list(data["tweetID"].to_list())

        # join on recieved tweets (not all are succesful)
        data = fetched.join(data.set_index("tweetID"), on="tweetID")

        # minor processing
        data.rename(columns={" Annotated_to_Emotion_CLASS": "label"}, inplace=True)
        data.drop(["tweetID"], axis=1, inplace=True)

        return data
