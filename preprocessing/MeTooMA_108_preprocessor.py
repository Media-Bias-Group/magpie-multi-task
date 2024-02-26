"""This module contains the Preprocessor for the 108_MeTooMA dataset."""

import os

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint


class Preprocessor108MeTooMA(PreprocessorBlueprint):
    """Preprocessor for the 108_MeTooMA dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor108MeTooMA."""
        super(Preprocessor108MeTooMA, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: pd.DataFrame, length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 108_MeTooMA."""
        self._log_before_preprocessing()
        df = raw_data.drop(columns=["Country"])
        # (1) Relevance is not a useful feature for us.
        # Authors: We expect this relevance annotation could serve as a useful filter for downstream modeling.
        df.drop(columns=["Relevance"], inplace=True)

        # (2) Stance
        # We model this task as a regression task (-1=oppose, 0=neutral, 1=support)
        df["oppose_support_label"] = df[["Oppose", "Support"]].apply(
            lambda x: -1 if x.Oppose else 1 if x.Support else 0, axis=1
        )
        df.drop(columns=["Oppose", "Support"], inplace=True)

        # (3) HateSpeech
        # We collapse the labels of directed and generalized hate and model this
        # task as binary classification (1=hate_speech, 0=no-hate_speech)
        df["hate_speech_label"] = df[["Directed_Hate", "Generalized_Hate"]].apply(
            lambda x: (x.Directed_Hate or x.Generalized_Hate), axis=1
        )
        df.drop(columns=["Directed_Hate", "Generalized_Hate"], inplace=True)

        # (4) Sarcasm
        # We model this task as binary classification (1=sarcasm, 0=no-sarcasm)
        df.rename(columns={"Sarcasm": "sarcasm_label"}, inplace=True)

        # (5) Dialogue Acts
        # We model this task as a multi-label classification task
        df.rename(
            columns={
                "Allegation": "allegation_label",
                "Justification": "justification_label",
                "Refutation": "refutation_label",
            },
            inplace=True,
        )
        df[
            [
                "allegation_label",
                "justification_label",
                "refutation_label",
                "oppose_support_label",
                "hate_speech_label",
            ]
        ] = df[
            [
                "allegation_label",
                "justification_label",
                "refutation_label",
                "oppose_support_label",
                "hate_speech_label",
            ]
        ].astype(
            int
        )

        df = self._clean(df=df, length_threshold=length_threshold)
        self._log_after_preprocessing(data=df)
        self.set_logger_data(key="primary_label_distribution", value=df.hate_speech_label.value_counts().to_dict())
        self.set_logger_data(
            key="additional_data",
            value={
                "allegation_label": df.allegation_label.value_counts().to_dict(),
                "justification_label": df.justification_label.value_counts().to_dict(),
                "refutation_label": df.refutation_label.value_counts().to_dict(),
                "oppose_support_label": df.oppose_support_label.value_counts().to_dict(),
                "hate_speech_label": df.hate_speech_label.value_counts().to_dict(),
                "retrieved_tweets": len(raw_data),
            },
        )

        return df

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 108_MeTooMA."""
        df = pd.read_csv(os.path.join(self._raw_data_local_path, "fetched.csv"))
        return df

    def _log_before_preprocessing(self):
        self.set_logger_data(key="original_size", value=9973)
