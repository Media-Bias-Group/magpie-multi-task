"""This module contains the Preprocessor for the 03 CW-HARD dataset."""

import os
from typing import List, Tuple

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint


class Preprocessor03CWHARD(PreprocessorBlueprint):
    """Preprocessor for the 03 CW_HARD dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor03CWHARD."""
        super(Preprocessor03CWHARD, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: Tuple[List, List], length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 03_CW_HARD."""
        self._log_before_preprocessing(data=raw_data)
        biased, neutral = raw_data

        data = pd.DataFrame({"text": biased + neutral, "label": [1] * len(biased) + [0] * len(neutral)})

        cleaned = self._clean(data, length_threshold=length_threshold)
        self._log_after_preprocessing(data=cleaned)
        self.set_logger_data("primary_label_distribution", cleaned["label"].value_counts().to_dict())
        return cleaned

    def _load_raw_data_from_local(self) -> Tuple[List, List]:
        """Load the raw data of 03_CW_HARD."""
        with open(os.path.join(self._raw_data_local_path, "statements_biased"), "r") as fd:
            biased_sentences = fd.readlines()

        with open(os.path.join(self._raw_data_local_path, "statements_neutral_featured"), "r") as fd:
            neutral_sentences = fd.readlines()

        return biased_sentences, neutral_sentences

    def _log_before_preprocessing(self, data: Tuple[List, List]):
        biased, neutral = data
        self.set_logger_data("original_size", len(biased + neutral))
