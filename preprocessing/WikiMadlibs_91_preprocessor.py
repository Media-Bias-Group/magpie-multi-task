"""This module contains the Preprocessor for the 91 Wiki Madlibs dataset."""

import os
from zipfile import ZipFile

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint

MAPPING = {"toxic": 1, "nontoxic": 0}


class Preprocessor91WikiMadlibs(PreprocessorBlueprint):
    """Preprocessor for the 91 Wiki Madlibs dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor91WikiMadlibs."""
        super(Preprocessor91WikiMadlibs, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: pd.DataFrame, length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 91_WikiMadlibs."""
        self._log_before_preprocessing(data=raw_data)

        # Drop those sentences as they do not constitute "real sentences" (i.e. consist of verb and adj only)
        df = raw_data[raw_data["template"] != "verb_adj"]
        df.drop("template", inplace=True, axis=1)
        df.columns = ["label", "text"]
        df.label = df.label.apply(lambda x: 1 if x == "toxic" else 0)
        df.text = df.text.apply(lambda x: x + ".")

        df = self._clean(df=df, length_threshold=length_threshold)
        self._log_after_preprocessing(data=df)
        self.set_logger_data(key="primary_label_distribution", value=df.label.value_counts().to_dict())
        self.set_logger_data(key="additional_data", value={"MAPPING": MAPPING})

        return df

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 91_WikiMadlibs."""
        with ZipFile(os.path.join(self._raw_data_local_path, "raw.zip"), "r") as zipObj:
            zipObj.extractall(path=self._raw_data_local_path)

        df = pd.read_csv(
            os.path.join(self._raw_data_local_path, "sentence_templates", "en_sentence_templates.csv"),
        )

        return df
