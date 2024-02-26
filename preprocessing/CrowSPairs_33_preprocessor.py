"""This module contains the Preprocessor for the 33 CrowSPairs dataset."""

import os

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint

MAPPING = dict(
    zip(
        [
            "race-color",
            "socioeconomic",
            "gender",
            "disability",
            "nationality",
            "sexual-orientation",
            "physical-appearance",
            "religion",
            "age",
        ],
        range(9),
    )
)


class Preprocessor33CrowSPairs(PreprocessorBlueprint):
    """Preprocessor for the 33_CrowSPairs dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor33CrowSPairs."""
        super(Preprocessor33CrowSPairs, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: pd.DataFrame, length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 33_CrowSPairs."""
        self._log_before_preprocessing(data=raw_data)

        df = raw_data.drop(columns=["stereo_antistereo", "anon_writer", "anon_annotators", "annotations"])
        stereotype = df.drop(columns=["sent_less"]).rename(columns={"sent_more": "text"})
        stereotype["label"] = 1
        anti_stereotype = df.drop(columns=["sent_more"]).rename(columns={"sent_less": "text"})
        anti_stereotype["label"] = 0

        sent_less_split = df["sent_less"].apply(lambda x: x.split(" ")).tolist()
        sent_more_split = df["sent_more"].apply(lambda x: x.split(" ")).tolist()

        sent_less_split = [set(tokens) for tokens in sent_less_split]
        sent_more_split = [set(tokens) for tokens in sent_more_split]

        zips = list(zip(sent_less_split, sent_more_split))
        diffs = [(list(z[1] - z[0]), list(z[0] - z[1])) for z in zips]
        diffs_stereotype = [diff[0] for diff in diffs]
        diffs_anti_stereotype = [diff[1] for diff in diffs]
        stereotype["pos"] = diffs_stereotype
        anti_stereotype["pos"] = diffs_anti_stereotype

        df = pd.concat([stereotype, anti_stereotype], axis=0)
        df["bias_type"] = df.bias_type.map(MAPPING)
        df.rename(columns={"bias_type": "stereotype_label"}, inplace=True)

        df["pos"] = df["pos"].apply(
            lambda pos_list: ";".join([self._unify_text(text=x, rm_hashtag=False) for x in pos_list if ";" not in x])
        )

        df = self._clean(df=df, length_threshold=length_threshold, rm_hashtag=False)
        self._log_after_preprocessing(data=df)
        self.set_logger_data(key="primary_label_distribution", value=df.label.value_counts().to_dict())
        self.set_logger_data(
            key="additional_data",
            value={
                "stereotype_label_distribution": df.stereotype_label.value_counts().to_dict(),
            },
        )
        return df

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 33_CrowSPairs."""
        df = pd.read_csv(os.path.join(self._raw_data_local_path, "data", "crows_pairs_anonymized.csv"), index_col=0)
        return df
