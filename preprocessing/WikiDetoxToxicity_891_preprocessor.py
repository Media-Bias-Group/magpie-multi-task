"""This module contains the Preprocessor for the 891_WikiDetoxToxicity dataset."""

import os
import re
from typing import Tuple

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint


class Preprocessor891WikiDetoxToxicity(PreprocessorBlueprint):
    """Preprocessor for the 891_WikiDetoxToxicity dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor89WikiDetox."""
        super(Preprocessor891WikiDetoxToxicity, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: Tuple[pd.DataFrame, pd.DataFrame], length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 891_WikiDetoxToxicity."""
        df_toxicity_comments, df_toxicity_annotations = raw_data
        self._log_before_preprocessing(data=df_toxicity_annotations)

        df_toxicity_comments = df_toxicity_comments[["rev_id", "comment"]]
        df_toxicity_annotations = df_toxicity_annotations.groupby(["rev_id"]).agg(
            toxicity_score=("toxicity_score", "mean"),
            rev_id=("rev_id", "first"),
        )
        df_toxicity_comments.set_index("rev_id", inplace=True)
        df_toxicity_annotations.set_index("rev_id", inplace=True)

        df_toxicity = df_toxicity_comments.join(df_toxicity_annotations, on="rev_id")
        df_toxicity = df_toxicity[~df_toxicity["toxicity_score"].isna()]
        df_toxicity.columns = ["text", "label"]

        df_toxicity["text"] = df_toxicity["text"].apply(lambda x: re.sub("NEWLINE_TOKEN", "", x))

        df_toxicity = self._clean(df=df_toxicity, length_threshold=length_threshold)
        self._log_after_preprocessing(data=df_toxicity)

        self.set_logger_data(key="primary_label_distribution", value=df_toxicity.label.describe().to_dict())

        return df_toxicity

    def _load_raw_data_from_local(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load the raw data of 89_wiki-detox."""
        toxicity_comments = os.path.join(self._raw_data_local_path, "4563973", "toxicity_annotated_comments.tsv")
        df_toxicity_comments = pd.read_csv(toxicity_comments, sep="\t")
        toxicity_annotations = os.path.join(self._raw_data_local_path, "4563973", "toxicity_annotations.tsv")
        df_toxicity_annotations = pd.read_csv(toxicity_annotations, sep="\t")

        return (
            df_toxicity_comments,
            df_toxicity_annotations,
        )
