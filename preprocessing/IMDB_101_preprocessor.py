"""This module contains the Preprocessor for the 101 IMDB movie review dataset."""

import os
from typing import Any, Dict, List
from zipfile import ZipFile

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint


class Preprocessor101IMDB(PreprocessorBlueprint):
    """Preprocessor for the 101_IMDB dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor101IMDB."""
        super(Preprocessor101IMDB, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: Dict[str, List[Any]], length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 101_IMDB."""
        neg = pd.DataFrame(raw_data["neg"], columns=["text"])
        pos = pd.DataFrame(raw_data["pos"], columns=["text"])

        neg["label"] = 0
        pos["label"] = 1
        df = pd.concat([neg, pos], axis=0)
        self._log_before_preprocessing(data=df)

        df = self._clean(df=df, length_threshold=length_threshold)
        self._log_after_preprocessing(data=df)
        self.set_logger_data(key="primary_label_distribution", value=df.label.value_counts().to_dict())

        return df

    def _load_raw_data_from_local(self) -> Dict[str, List[Any]]:
        """Load the raw data of 101_IMDB."""
        with ZipFile(os.path.join(self._raw_data_local_path, "raw.zip"), "r") as zipObj:
            zipObj.extractall(path=self._raw_data_local_path)

        split = ["train", "test"]
        sentiment = ["pos", "neg"]
        sents: Dict[str, List[Any]] = {"pos": [], "neg": []}
        for se in sentiment:
            for sp in split:
                files = [
                    os.path.join(self._raw_data_local_path, "aclImdb", sp, se, f)
                    for f in os.listdir(os.path.join(self._raw_data_local_path, "aclImdb", sp, se))
                ]
                for file in files:
                    with open(file, "r") as f:
                        # We manually checked that each file contains one and only one sentence.
                        data = f.readline()
                        sents[se].append(data)
        return sents
