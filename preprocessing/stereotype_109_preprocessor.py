"""This module contains the Preprocessor for the 109 stereotype dataset."""

import os

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint

MAPPING = {"yes": 1, "no": 0}


class Preprocessor109Stereotype(PreprocessorBlueprint):
    """Preprocessor for the 109_stereotype dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor109Stereotype."""
        super(Preprocessor109Stereotype, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: pd.DataFrame, length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 109_stereotype."""
        raw_data.rename(
            columns={
                "Text": "text",
                "Explicit Stereotype?": "stereotype_explicit_label",
                "Implicit Stereotypical Association?": "stereotype_implicit_label",
            },
            inplace=True,
        )
        self._log_before_preprocessing(data=raw_data)

        df = raw_data[["text"]]
        assert set(raw_data["stereotype_explicit_label"].unique().tolist()) == set(["yes", "no"])
        assert set(raw_data["stereotype_implicit_label"].unique().tolist()) == set(["yes", "no"])
        df["stereotype_explicit_label"] = raw_data["stereotype_explicit_label"].map(MAPPING)
        df["stereotype_implicit_label"] = raw_data["stereotype_implicit_label"].map(MAPPING)

        # Collapse both, explicit and implicit to create overall label: Stereotype y/n
        df["label"] = df["stereotype_explicit_label"] | df["stereotype_implicit_label"]

        df = self._clean(df=df, length_threshold=length_threshold)
        self._log_after_preprocessing(data=df)
        self.set_logger_data(key="primary_label_distribution", value=df.label.value_counts().to_dict())
        self.set_logger_data(
            key="additional_data",
            value={
                "stereotype_explicit_label_distribution": df.stereotype_explicit_label.value_counts().to_dict(),
                "stereotype_implicit_label_distribution": df.stereotype_implicit_label.value_counts().to_dict(),
            },
        )

        return df

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 109_stereotype."""
        df = pd.read_csv(os.path.join(self._raw_data_local_path, "annotated_data.csv"))
        return df
