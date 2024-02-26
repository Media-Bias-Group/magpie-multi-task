"""This module contains the Preprocessor for the 75 RedditBias dataset."""

import os
from typing import Any, Tuple

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint


class Preprocessor75RedditBias(PreprocessorBlueprint):
    """Preprocessor for the 75_RedditBias dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor75RedditBias."""
        super(Preprocessor75RedditBias, self).__init__(*args, **kwargs)
        self._processed_data_local_path = os.path.join(kwargs["local_path"], "preprocessed.csv")
        self._processed_data_gcs_path = os.path.join(kwargs["local_path"], "preprocessed.csv")

    def _preprocess(
        self,
        raw_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
        length_threshold: int,
    ) -> pd.DataFrame:
        """Preprocess the raw data of 75_RedditBias."""
        self._log_before_preprocessing(data=raw_data)
        dfs = [df[["comment", "phrase", "bias_sent", "bias_phrase"]] for df in raw_data]

        for i, df in enumerate(dfs):
            df["label_group"] = i
            dfs[i] = df

        def clean_labels(label: Any):
            try:
                label = int(label)
                return label if label in [0, 1] else float("NaN")
            except ValueError:
                return float("NaN")

        df = pd.concat(dfs, axis=0)

        df.dropna(inplace=True)
        # Exclude those cases, where we have a biased sentence AND a NON biased phrase
        # orientation (18), gender_female (19), religion_jew (788), religion_muslim (37), race_black (42)
        mask_not_0_1 = df.apply(lambda x: not bool(x.bias_sent and not x.bias_phrase), axis=1).to_numpy()
        # In only one case, the phrase is not contained in the comment
        mask_phrase_in_sent = df.apply(lambda x: bool(x.phrase in x.comment), axis=1).to_numpy()
        df = df[mask_not_0_1 & mask_phrase_in_sent]
        df.drop(columns=["bias_phrase"], inplace=True)
        df.rename(columns={"comment": "text", "phrase": "bias_pos", "bias_sent": "label"}, inplace=True)
        df["label"] = df["label"].apply(clean_labels)
        df.dropna(inplace=True)
        df["bias_pos"] = df["bias_pos"].apply(lambda x: self._unify_text(text=x, rm_hashtag=False))

        df = self._clean(df=df, length_threshold=length_threshold, rm_hashtag=False)
        self._log_after_preprocessing(data=df)

        self.set_logger_data(key="primary_label_distribution", value=df.label.value_counts().to_dict())
        self.set_logger_data(key="additional_data", value=df.label_group.value_counts().to_dict())

        return df

    def _load_raw_data_from_local(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the raw data of 75_RedditBias."""
        df_orientation = pd.read_csv(
            os.path.join(
                self._raw_data_local_path,
                "data",
                "orientation",
                "reddit_comments_orientation_lgbtq_processed_phrase_annotated.csv",
            )
        )
        df_gender_female = pd.read_csv(
            os.path.join(
                self._raw_data_local_path,
                "data",
                "gender",
                "reddit_comments_gender_female_processed_phrase_annotated.csv",
            )
        )
        df_gender_female.columns = df_gender_female.columns[:-1].to_list() + ["id"]
        with open(
            os.path.join(
                self._raw_data_local_path,
                "data",
                "religion1",
                "reddit_comments_religion1_jews_processed_phrase_annotated.csv",
            ),
            errors="replace",
        ) as f:
            df_religion_jew = pd.read_csv(f)
        df_religion_muslim = pd.read_csv(
            os.path.join(
                self._raw_data_local_path,
                "data",
                "religion2",
                "reddit_comments_religion2_muslims_processed_phrase_annotated.csv",
            )
        )
        df_race_black = pd.read_csv(
            os.path.join(
                self._raw_data_local_path, "data", "race", "reddit_comments_race_black_processed_phrase_annotated.csv"
            )
        )

        return df_orientation, df_gender_female, df_religion_jew, df_religion_muslim, df_race_black

    def _log_before_preprocessing(self, data: Any):
        self.set_logger_data(key="original_size", value=sum(len(df) for df in data))
