"""This module contains the Preprocessor for the 40 JIGSAW dataset."""

import os
from zipfile import ZipFile

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint

MAPPING_PRIMARY_LABEL = {"toxic": 1, "nontoxic": 0}
MAPPING_GENDER_LABEL = {"mention-of-identity": 1, "no-mention": 0}
MAPPING_ETHNICITY_LABEL = {"mention-of-identity": 1, "no-mention": 0}
MAPPING_RELIGION_LABEL = {"mention-of-identity": 1, "no-mention": 0}
MAPPING_DISABILITY_LABEL = {"mention-of-identity": 1, "no-mention": 0}


class Preprocessor40JIGSAW(PreprocessorBlueprint):
    """Preprocessor for the 40_JIGSAW dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor40JIGSAW."""
        super(Preprocessor40JIGSAW, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: pd.DataFrame, length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 40_JIGSAW."""
        self._log_before_preprocessing(data=raw_data)
        meta_columns = [
            "comment_text",
            "toxicity",
            "toxicity_annotator_count",
            "other_disability",
            "identity_annotator_count",
        ]

        gender_columns = [
            "male",
            "female",
            "transgender",
            "other_gender",
            "heterosexual",
            "homosexual_gay_or_lesbian",
            "bisexual",
            "other_sexual_orientation",
        ]

        religion_columns = ["christian", "jewish", "muslim", "hindu", "buddhist", "atheist", "other_religion"]

        ethnicity_columns = ["black", "white", "asian", "latino", "other_race_or_ethnicity"]

        disability_columns = [
            "physical_disability",
            "intellectual_or_learning_disability",
            "psychiatric_or_mental_illness",
        ]

        df = raw_data[meta_columns + gender_columns + religion_columns + ethnicity_columns + disability_columns]
        df = df[df.identity_annotator_count > 0]
        df.rename(columns={"comment_text": "text", "toxicity": "label"}, inplace=True)

        df["gender_label"] = (df[gender_columns].max(axis=1) > 0.5).astype(int)
        df["religion_label"] = (df[religion_columns].max(axis=1) > 0.5).astype(int)
        df["ethnicity_label"] = (df[ethnicity_columns].max(axis=1) > 0.5).astype(int)
        df["disability_label"] = (df[disability_columns].max(axis=1) > 0.5).astype(int)

        df["label"] = df["label"].apply(lambda x: int(x > 0.5))

        df = df[df.toxicity_annotator_count > 9]
        df = df[["text", "label", "gender_label", "religion_label", "ethnicity_label", "disability_label"]]

        df = self._clean(df=df, length_threshold=length_threshold)
        self._log_after_preprocessing(data=df)
        self.set_logger_data(key="primary_label_distribution", value=df.label.value_counts().to_dict())
        self.set_logger_data(
            key="additional_data",
            value={
                "label_gender_distribution": df.gender_label.value_counts().to_dict(),
                "label_ethnicity_distribution": df.ethnicity_label.value_counts().to_dict(),
                "label_religion_distribution": df.religion_label.value_counts().to_dict(),
                "label_disability_distribution": df.disability_label.value_counts().to_dict(),
                "MAPPING": {
                    "primary_label": MAPPING_PRIMARY_LABEL,
                    "label_gender_distribution": MAPPING_GENDER_LABEL,
                    "label_ethnicity_distribution": MAPPING_ETHNICITY_LABEL,
                    "label_religion_distribution": MAPPING_RELIGION_LABEL,
                    "label_disability_distribution": MAPPING_DISABILITY_LABEL,
                },
            },
        )

        return df

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 40_JIGSAW."""
        with ZipFile(os.path.join(self._raw_data_local_path, "raw.zip"), "r") as zipObj:
            zipObj.extractall(path=self._raw_data_local_path)

        df = pd.read_csv(os.path.join(self._raw_data_local_path, "all_data.csv"))
        return df
