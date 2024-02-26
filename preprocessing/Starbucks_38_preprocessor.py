"""This module contains the Preprocessor for the 38 Starbucks dataset."""

import os
import re

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint

MAPPING = {"not_biased": 1, "slightly_biased": 2, "biased": 3, "very_biased": 4}


class Preprocessor38Starbucks(PreprocessorBlueprint):
    """Preprocessor for the 38 Starbucks dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor38Starbucks."""
        super(Preprocessor38Starbucks, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: pd.DataFrame, length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 38_Starbucks."""
        raw_data.drop(
            [
                "id_event",
                "event",
                "date_event",
                "source",
                "source_bias",
                "url",
                "ref",
                "reftitle",
                "ref_url",
                "article_bias",
                "preknow",
                "reftext",
                "docbody",
            ],
            inplace=True,
            axis=1,
        )

        row_list = []

        # each row in the group represents one annotator, therefore in this loop we aggreagate and extract their annotations
        for _, df in raw_data.groupby("id_article"):
            # title
            row_list.append({"text": df["doctitle"].iloc[0], "label": df["t"].mean()})
            for sent in range(20):
                # article has less sentences
                if not df["s" + str(sent)].any():
                    continue

                sentence = df["s" + str(sent)].iloc[0]  # sentences are same within the group, just take first one
                label = df[str(sent)].mean()  # we take mean of annotations of the sentence
                row_list.append({"text": sentence, "label": label})

        data = pd.DataFrame(row_list)
        self._log_before_preprocessing(data=data)

        # cleaning
        # remove numbers in front of sentences
        data["text"] = data["text"].apply(lambda x: re.sub(r"(\[[0-9]*\]:\ )", "", x))
        # normalize labels ~ cast them to 0-1 scale
        data["label"] = (data["label"] - data["label"].min()) / (data["label"].max() - data["label"].min())

        cleaned = self._clean(data, length_threshold=length_threshold)
        self._log_after_preprocessing(data=cleaned)
        self.set_logger_data(
            "primary_label_distribution", {"mean": data["label"].mean(), "median": data["label"].median()}
        )
        self.set_logger_data("additional_data", MAPPING)

        return cleaned

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 38_Starbucks."""
        df = pd.read_csv(os.path.join(self._raw_data_local_path, "Sora_LREC2020_biasedsentences.csv"))

        return df
