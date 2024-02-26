"""This module contains the Preprocessor for the 127 VaccineLies dataset."""

import ast
import os
from typing import Tuple

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint

MAPPING = {"No Stance": 0, "Not Relevant": 1, "Accept": 2, "Reject": 3}


class Preprocessor127VaccineLies(PreprocessorBlueprint):
    """Preprocessor for the 127 VaccineLies dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor127VaccineLies."""
        super(Preprocessor127VaccineLies, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: Tuple[pd.DataFrame, pd.Series], length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 127_VaccineLies."""
        sentence_data, targets = raw_data
        rowlist = []
        for _, row in sentence_data.iterrows():
            text = row["text"]
            annotation_dict = ast.literal_eval(row["misinfo"])
            for id, annotation in annotation_dict.items():
                rowlist.append({"target": targets[id], "text": text, "label": annotation})

        df = pd.DataFrame(rowlist)
        df["label"] = df["label"].map(MAPPING)
        self._log_before_preprocessing(data=df)

        cleaned = self._clean(df, length_threshold=length_threshold)
        self._log_after_preprocessing(data=cleaned)
        self.set_logger_data("primary_label_distribution", cleaned["label"].value_counts().to_dict())
        self.set_logger_data("additional_data", MAPPING)

        return cleaned

    def _load_raw_data_from_local(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load the raw data of 127_VaccineLies."""
        sentence_data = pd.read_csv(
            os.path.join(self._raw_data_local_path, "fetched.csv"), usecols=["misinfo", "text"]
        )
        cov_targets = pd.read_json(os.path.join(self._raw_data_local_path, "covid19/taxonomy/misinfo.json")).iloc[0]
        hpv_targets = pd.read_json(os.path.join(self._raw_data_local_path, "hpv/taxonomy/misinfo.json")).iloc[0]

        targets_all = pd.concat([cov_targets, hpv_targets])

        return sentence_data, targets_all
