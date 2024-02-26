"""This module contains the Preprocessor for the 892_WikiDetoxAggressionAndAttack dataset."""

import os
import re
from typing import Tuple

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint


class Preprocessor892WikiDetoxAggressionAndAttack(PreprocessorBlueprint):
    """Preprocessor for the 892_WikiDetoxAggressionAndAttack dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor892WikiDetoxAggressionAndAttack."""
        super(Preprocessor892WikiDetoxAggressionAndAttack, self).__init__(*args, **kwargs)

    def _preprocess(
        self, raw_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], length_threshold: int
    ) -> pd.DataFrame:
        """Preprocess the raw data of 892_WikiDetoxAggressionAndAttack."""
        (
            df_aggression_comments,
            df_aggression_annotations,
            df_attack_comments,
            df_attack_annotations,
        ) = raw_data
        self._log_before_preprocessing(data=(df_aggression_comments, df_attack_comments))

        # Aggression
        df_aggression_comments = df_aggression_comments[["rev_id", "comment"]]
        df_aggression_annotations = df_aggression_annotations.groupby(["rev_id"]).agg(
            aggression_score=("aggression_score", "mean"),
            rev_id=("rev_id", "first"),
        )
        df_aggression_comments.set_index("rev_id", inplace=True)
        df_aggression_annotations.set_index("rev_id", inplace=True)

        df_aggression = df_aggression_comments.join(df_aggression_annotations, on="rev_id")
        df_aggression = df_aggression[~df_aggression["aggression_score"].isna()]
        df_aggression.columns = ["text", "label_aggression"]

        # Attack
        df_attack_comments = df_attack_comments[["rev_id", "comment"]]
        df_attack_annotations = df_attack_annotations.groupby(["rev_id"]).agg(
            rev_id=("rev_id", "first"),
            attack_votes=("attack", "sum"),
            count_annotations=("rev_id", "count"),
        )
        df_attack_comments.set_index("rev_id", inplace=True)
        df_attack_annotations.set_index("rev_id", inplace=True)

        df_attack_annotations["label_attack"] = df_attack_annotations.apply(
            lambda row: int(row.attack_votes / row.count_annotations > 0.5), axis=1
        )

        df_attack = df_attack_comments.join(df_attack_annotations, on="rev_id")
        df_attack = df_attack[["comment", "label_attack"]]
        df_attack.columns = ["text", "label_attack"]

        # Merge aggression and attack together since they contain the exact same observations
        df_aggression_and_attack = df_attack.join(df_aggression, on="rev_id", rsuffix="right")
        df_aggression_and_attack = df_aggression_and_attack[["text", "label_aggression", "label_attack"]]

        df_aggression_and_attack["text"] = df_aggression_and_attack["text"].apply(
            lambda x: re.sub("NEWLINE_TOKEN", "", x)
        )
        df_aggression_and_attack = self._clean(df=df_aggression_and_attack, length_threshold=length_threshold)
        self._log_after_preprocessing(data=df_aggression_and_attack)

        self.set_logger_data(
            key="primary_label_distribution", value=df_aggression_and_attack.label_attack.value_counts().to_dict()
        )
        self.set_logger_data(
            "additional_data",
            value={
                "label_aggression_distribution": df_aggression_and_attack.label_aggression.describe().to_dict(),
                "label_attack_distribution": df_aggression_and_attack.label_attack.value_counts().to_dict(),
            },
        )

        return df_aggression_and_attack

    def _load_raw_data_from_local(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the raw data of 892_WikiDetoxAggressionAndAttack."""
        aggression_comments = os.path.join(self._raw_data_local_path, "4267550", "aggression_annotated_comments.tsv")
        df_aggression_comments = pd.read_csv(aggression_comments, sep="\t")
        aggression_annotations = os.path.join(self._raw_data_local_path, "4267550", "aggression_annotations.tsv")
        df_aggression_annotations = pd.read_csv(aggression_annotations, sep="\t")

        attack_comments = os.path.join(self._raw_data_local_path, "4054689", "attack_annotated_comments.tsv")
        df_attack_comments = pd.read_csv(attack_comments, sep="\t")
        attack_annotations = os.path.join(self._raw_data_local_path, "4054689", "attack_annotations.tsv")
        df_attack_annotations = pd.read_csv(attack_annotations, sep="\t")

        return (
            df_aggression_comments,
            df_aggression_annotations,
            df_attack_comments,
            df_attack_annotations,
        )

    def _log_before_preprocessing(self, data: Tuple[pd.DataFrame, pd.DataFrame]):
        df_aggression_comments, df_attack_comments = data
        # Original size equals the amount of comments
        self.set_logger_data(
            key="original_size",
            value=len(df_aggression_comments) + len(df_attack_comments),
        )
