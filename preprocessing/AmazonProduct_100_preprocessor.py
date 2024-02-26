"""This module contains the Preprocessor for the 100 Amazon Product dataset."""

import os
from typing import List

import pandas as pd
from tqdm import tqdm

from preprocessing.preprocessors import PreprocessorBlueprint


class Preprocessor100AmazonProduct(PreprocessorBlueprint):
    """Preprocessor for the 100 Amazon Product Reviews dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor100AmazonProduct."""
        super(Preprocessor100AmazonProduct, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: List, length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 100_AmazonProduct."""
        self._log_before_preprocessing(data=raw_data)

        # take just a sample chunk
        data = raw_data[:200000]

        rowlist = []
        for sent in tqdm(data):
            label = sent[:10]  # first 10 characters of every line are __label__1 or __label__2
            sentence = sent.strip(label + " ")  # label is seperated by one whitespace
            rowlist.append({"text": sentence, "label": label[-1:]})

        df = pd.DataFrame(rowlist)
        df["label"] = df["label"].astype(int) - 1
        cleaned = self._clean(df=df, length_threshold=length_threshold)
        self._log_after_preprocessing(data=cleaned)
        self.set_logger_data("primary_label_distribution", cleaned["label"].value_counts().to_dict())
        return cleaned

    def _load_raw_data_from_local(self) -> List:
        """Load the raw data of 100_AmazonProduct."""
        # load train, test and mix em

        with open(os.path.join(self._raw_data_local_path, "test.ft.txt"), "r") as fd:
            test = fd.readlines()

        with open(os.path.join(self._raw_data_local_path, "train.ft.txt"), "r") as fd:
            train = fd.readlines()

        mixed = test + train
        return mixed
