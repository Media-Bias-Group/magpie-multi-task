"""This module contains the Preprocessor for the 31 Subjectivity dataset SUBJ."""

import os
from typing import List, Tuple

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint


class Preprocessor31SUBJ(PreprocessorBlueprint):
    """Preprocessor for the 31 SUBJ dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor31SUBJ."""
        super(Preprocessor31SUBJ, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: Tuple[List, List], length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 31_SUBJ."""
        obj, subj = raw_data
        sentences = obj + subj

        labels = [0] * len(obj) + [1] * len(subj)
        df = pd.DataFrame({"text": sentences, "label": labels})

        self._log_before_preprocessing(data=df)
        cleaned = self._clean(df, length_threshold=length_threshold)
        self._log_after_preprocessing(data=cleaned)
        self.set_logger_data("primary_label_distribution", cleaned["label"].value_counts().to_dict())

        return cleaned

    def _load_raw_data_from_local(self) -> Tuple[List, List]:
        """Load the raw data of 31_SUBJ."""
        with open(os.path.join(self._raw_data_local_path, "objective.5000"), "r", errors="ignore") as f:
            obj = f.read().splitlines()

        with open(os.path.join(self._raw_data_local_path, "subjective.5000"), "r", errors="ignore") as f:
            subj = f.read().splitlines()

        return obj, subj
