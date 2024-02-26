"""This module contains the Preprocessor for the 120_SemEval2023Task3 dataset."""

import os
from zipfile import ZipFile

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint

MAPPING = {
    "Flag_Waving": 0,
    "Slogans": 1,
    "Doubt": 2,
    "Conversation_Killer": 3,
    "Loaded_Language": 4,
    "Causal_Oversimplification": 5,
    "Obfuscation-Vagueness-Confusion": 6,
    "Appeal_to_Fear-Prejudice": 7,
    "Straw_Man": 8,
    "Appeal_to_Popularity": 9,
    "Appeal_to_Hypocrisy": 10,
    "Name_Calling-Labeling": 11,
    "Exaggeration-Minimisation": 12,
    "Whataboutism": 13,
    "Appeal_to_Authority": 14,
    "Guilt_by_Association": 15,
    "Repetition": 16,
    "Red_Herring": 17,
    "False_Dilemma-No_Choice": 18,
}


class Preprocessor120SemEval2023Task3(PreprocessorBlueprint):
    """Preprocessor for the 120_SemEval2023Task3 dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor120SemEval2023Task3."""
        super(Preprocessor120SemEval2023Task3, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: pd.DataFrame, length_threshold: int):
        """Preprocess the raw data of 120_SemEval2023Task3."""
        self._log_before_preprocessing(data=raw_data)

        df = raw_data.fillna("")

        for k in MAPPING.keys():
            df[k] = df.apply(
                lambda row: 1 if k in [t for t in row.used_techniques.split(",") if t != ""] else 0, axis=1
            )
        df["label"] = df.apply(lambda row: any(row[list(MAPPING.keys())]), axis=1)
        df.drop(columns=["used_techniques"], inplace=True)
        df = self._clean(df=df, length_threshold=length_threshold)
        self._log_after_preprocessing(data=df)
        self.set_logger_data(key="primary_label_distribution", value=df.label.value_counts().to_dict())
        self.set_logger_data(
            key="additional_data",
            value={"multi_label_distribution": {f"{k}": df[k].value_counts().to_dict() for k in MAPPING.keys()}},
        )

        return df

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 120_SemEval2023Task3."""
        # Here first unzip the raw zip file
        with ZipFile(os.path.join(self._raw_data_local_path, "raw_data.zip"), "r") as zipObj:
            zipObj.extractall(path=self._raw_data_local_path)

        train_texts = pd.read_csv(
            os.path.join(self._raw_data_local_path, "data", "en", "train-labels-subtask-3.template"),
            sep="\t",
            header=None,
            names=["article_id", "paragraph_id", "text"],
        )

        train_labels = pd.read_csv(
            os.path.join(self._raw_data_local_path, "data", "en", "train-labels-subtask-3.txt"),
            sep="\t",
            header=None,
            names=["article_id", "paragraph_id", "used_techniques"],
        )

        train_labels.set_index(["article_id", "paragraph_id"], inplace=True)
        train_texts.set_index(["article_id", "paragraph_id"], inplace=True)

        df = train_labels.join(train_texts, how="right")
        return df
