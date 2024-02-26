"""This module contains the Preprocessor for the 18_GAP dataset."""

import os
from typing import Tuple

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint

MAPPING = {"A": "<TAG-A>", "B": "<TAG-B>", "Pronoun": "<TAG-P>"}
MAPPING_PRONOUNS = {"Her": 1, "His": 0, "he": 0, "hers": 1, "him": 0, "his": 0, "He": 0, "she": 1, "her": 1, "She": 1}


class Preprocessor18GAP(PreprocessorBlueprint):
    """Preprocessor for the 18_GAP dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor18GAP."""
        super(Preprocessor18GAP, self).__init__(*args, **kwargs)

    def _preprocess(
        self, raw_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], length_threshold: int
    ) -> pd.DataFrame:
        """Preprocess the raw data of 18_GAP."""
        self._log_before_preprocessing(data=raw_data)

        def tag_text(t, s, le, ta):
            new = t[:s] + f"{ta} " + t[s : s + le] + f" {ta}" + t[s + le :]
            return new

        # print(myfunc("hi i am jerome", 5, 2, "TAG"))

        df = pd.concat(raw_data, axis=0)
        df.reset_index(inplace=True)
        df["text_new"] = df["Text"]
        for i, row in df.iterrows():
            data = list(
                zip(
                    [row["A-offset"], row["B-offset"], row["Pronoun-offset"]],
                    [len(row["A"]), len(row["B"]), len(row["Pronoun"])],
                    ["A", "B", "Pronoun"],
                )
            )
            sorted_by_occurance = sorted(data, key=lambda tup: -tup[0])
            text = row["Text"]
            for start, length, column in sorted_by_occurance:
                text = tag_text(t=text, s=start, le=length, ta=MAPPING[column])
            df.loc[i, "text_new"] = text

        # mask = df.apply(lambda x: x["A-coref"] ^ x["B-coref"], axis=1)
        # 0=neither 1=A-coref, 2=B-coref
        df["label"] = df.apply(lambda x: 0 if not (x["A-coref"] ^ x["B-coref"]) else 1 if x["A-coref"] else 2, axis=1)
        df["label_gender"] = df.Pronoun.map(MAPPING_PRONOUNS)
        df = df[["text_new", "label", "Text", "label_gender"]].rename(columns={"text_new": "text", "Text": "text_raw"})
        df = self._clean(df=df, length_threshold=length_threshold)
        self._log_after_preprocessing(data=df)
        self.set_logger_data(key="primary_label_distribution", value=df.label.value_counts().to_dict())
        self.set_logger_data(
            key="additional_data",
            value={
                "MAPPING": MAPPING,
                "MAPPING_PRONOUNS": MAPPING_PRONOUNS,
                "label_gender_distribution": df.label_gender.value_counts().to_dict(),
            },
        )
        return df

    def _load_raw_data_from_local(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the raw data of 18_GAP."""
        gap_train = pd.read_csv(
            os.path.join(self._raw_data_local_path, "gap-development.tsv"),
            sep="\t",
            usecols=["Text", "Pronoun", "Pronoun-offset", "A", "A-offset", "A-coref", "B", "B-offset", "B-coref"],
        )

        gap_val = pd.read_csv(
            os.path.join(self._raw_data_local_path, "gap-validation.tsv"),
            sep="\t",
            usecols=["Text", "Pronoun", "Pronoun-offset", "A", "A-offset", "A-coref", "B", "B-offset", "B-coref"],
        )

        gap_test = pd.read_csv(
            os.path.join(self._raw_data_local_path, "gap-test.tsv"),
            sep="\t",
            usecols=["Text", "Pronoun", "Pronoun-offset", "A", "A-offset", "A-coref", "B", "B-offset", "B-coref"],
        )

        return gap_train, gap_val, gap_test

    def _log_before_preprocessing(self, data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]):
        self.set_logger_data(key="original_size", value=len(data[0]) + len(data[1]) + len(data[2]))
