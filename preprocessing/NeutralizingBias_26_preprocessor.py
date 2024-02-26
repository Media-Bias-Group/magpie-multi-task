"""This module contains the Preprocessor for the 26 Neutralizing Bias dataset."""

import os
from zipfile import ZipFile

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint


class Preprocessor26NeutralizingBias(PreprocessorBlueprint):
    """Preprocessor for the 26_Neutralizing-bias dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor26NeutralizingBias."""
        super(Preprocessor26NeutralizingBias, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: pd.DataFrame, length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 26_neutralizing-bias."""
        self._log_before_preprocessing(data=raw_data)
        df = raw_data[["src_raw", "tgt_raw"]]

        df["src_raw"] = df[["src_raw"]].astype("string")
        df["tgt_raw"] = df[["tgt_raw"]].astype("string")
        df["src_split"] = df["src_raw"].apply(lambda s: s.split(" "))
        df["tgt_split"] = df["tgt_raw"].apply(lambda s: s.split(" "))

        # This is the part of our preprocessing pipeline that takes time (<60 seconds)
        df["bias_inducing_words"] = df.apply(
            lambda x: [token for token in x.src_split if token not in x.tgt_split], axis=1
        )
        df = df[["src_raw", "bias_inducing_words"]].rename(columns={"src_raw": "text", "bias_inducing_words": "pos"})
        # Only take those observations where we have one and only one bias inducing word.
        df = df[df["pos"].apply(lambda x: len(x) == 1)]
        # Cast list to string. Now list contains only one element.
        df["pos"] = df["pos"].apply(lambda x: x[0])
        df["pos"] = df["pos"].fillna("")
        df["pos"] = df["pos"].apply(lambda x: self._unify_text(text=x, rm_hashtag=False))

        df = self._clean(df=df, length_threshold=length_threshold)
        self._log_after_preprocessing(data=df)

        return df

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 26_neutralizing-bias."""
        # Here first unzip the raw zip file
        with ZipFile(os.path.join(self._raw_data_local_path, "wnc_raw_files.zip"), "r") as zipObj:
            zipObj.extractall(path=self._raw_data_local_path)

        df = pd.read_csv(
            os.path.join(self._raw_data_local_path, "WNC", "biased.word.train"),
            sep="\t",
            names=["id", "src_tok", "tgt_tok", "src_raw", "tgt_raw", "src_POS_tags", "tgt_parse_tags"],
        )

        # The dev- and testset were sampled from the train set. Therefore, we only load the training-set.
        return df
