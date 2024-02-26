"""This module contains the Fetcher for the 126_WTWT dataset."""

import json
import os

import pandas as pd

from tweet_fetching.fetcher_blueprint import FetcherBlueprint
from tweet_fetching.twitter.tweetLoader import TweetLoader


class Fetcher126WTWT(FetcherBlueprint):
    """Fetcher for the 126_WTWT dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Fetcher126WTWT."""
        super(Fetcher126WTWT, self).__init__(*args, **kwargs)
        self._tweetloader = TweetLoader()

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 126_WTWT."""
        with open(os.path.join(self._raw_data_local_path, "wtwt_ids.json")) as json_file:
            data = json.load(json_file)
        df = pd.DataFrame(data)
        return df

    def _fetch(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Fetch the raw tweets data of 126_WTWT."""
        # ~80% of the data is either "unrelated" or "comment" and neither in ["refute", "support"]
        df = raw_data
        ids_list = df["tweet_id"].tolist()
        original_tweets = self._tweetloader.fetch_list(ids_list=ids_list)  # 10.380

        df.set_index("tweet_id", inplace=True)
        df.index = df.index.astype(int)
        original_tweets.set_index("tweetID", inplace=True)

        # There are some duplicates in the data.
        # This is mainly due to spam, e.g. RT of M&A-related news.
        # We decided to remove these duplicates entirely. We lose 906 observations.
        duplicate_indices = original_tweets["text"].duplicated()
        original_tweets = original_tweets[~duplicate_indices]

        return df.join(original_tweets, how="right").reset_index()  # 9474
