"""This module contains the Fetcher for the 12 PHEME dataset."""

import os
from typing import Dict, Tuple
from zipfile import ZipFile

import pandas as pd

from tweet_fetching.fetcher_blueprint import FetcherBlueprint
from tweet_fetching.twitter.tweetLoader import TweetLoader


class Fetcher12PHEME(FetcherBlueprint):
    """Fetcher for the 12 PHEME dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Fetcher12PHEME."""
        super(Fetcher12PHEME, self).__init__(*args, **kwargs)
        self._tweetloader = TweetLoader()

    def _load_raw_data_from_local(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Load the raw data of 12_PHEME."""
        with ZipFile(os.path.join(self._raw_data_local_path, "12_PHEME.zip"), "r") as zipObj:
            zipObj.extractall(path=self._raw_data_local_path)

        path = os.path.join(self._raw_data_local_path, "12_PHEME", "PHEME_veracity", "all-rnr-annotated-threads")
        topics = [ele for ele in os.listdir(path) if "." not in ele]

        dict_non_rumors = {}
        for topic in topics:
            for ele in os.listdir(os.path.join(path, topic, "non-rumours")):
                if "." in ele:
                    continue
                v = topic
                dict_non_rumors[ele] = v

        dict_rumors = {}
        for topic in topics:
            for ele in os.listdir(os.path.join(path, topic, "rumours")):
                if "." in ele:
                    continue
                v = topic
                dict_rumors[ele] = v

        return dict_rumors, dict_non_rumors

    def _fetch(self, raw_data: Tuple[Dict[str, str], Dict[str, str]]) -> pd.DataFrame:
        """Fetch the raw tweets data of 12_PHEME."""
        dict_rumors, dict_non_rumors = raw_data

        ids_list_rumors_all = list(dict_rumors.keys())
        tweets_rumour = self._tweetloader.fetch_list(ids_list=ids_list_rumors_all)
        tweets_rumour["topic"] = tweets_rumour["tweetID"].apply(lambda x: dict_rumors[str(x)])
        tweets_rumour["label"] = 1

        ids_list_non_rumors_all = list(dict_non_rumors.keys())
        tweets_non_rumour = self._tweetloader.fetch_list(ids_list=ids_list_non_rumors_all)
        tweets_non_rumour["topic"] = tweets_non_rumour["tweetID"].apply(lambda x: dict_non_rumors[str(x)])
        tweets_non_rumour["label"] = 0

        df = pd.concat([tweets_rumour, tweets_non_rumour], axis=0)
        return df
