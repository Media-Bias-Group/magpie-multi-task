"""This module contains the Preprocessor for the 42 GoodNewsEveryone dataset."""

import os
import re
from ast import literal_eval

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint


class Preprocessor42GoodNewsEveryone(PreprocessorBlueprint):
    """Preprocessor for the 42_GoodNewsEveryone dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor42GoodNewsEveryone."""
        super(Preprocessor42GoodNewsEveryone, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: pd.DataFrame, length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 42_GoodNewsEveryone."""
        self._log_before_preprocessing(data=raw_data)

        raw_data.drop(columns=["url", "country", "source"], inplace=True)
        df = raw_data[["headline", "experiencer", "cue"]]

        def load_list_literal_eval(l_as_str):
            try:
                return literal_eval(l_as_str)
            except SyntaxError:
                return float("NaN")

        experiencer = df.experiencer.apply(load_list_literal_eval)
        cues = df.cue.apply(load_list_literal_eval)
        df = pd.concat([df["headline"], experiencer, cues], axis=1).dropna()

        # Drop those observations where the cues and experiencers do not match the usual format
        # usual format: [[...]]
        # Here drop eg: []. We lose 5 rows
        df["mask_cue"] = df["cue"].apply(lambda x: True if len(x) > 0 else float("NaN"))
        df["mask_experiencer"] = df["experiencer"].apply(lambda x: True if len(x) > 0 else float("NaN"))
        df = df.dropna()

        # Flatten the lists from [[...]] to [...]
        df["cue"] = df["cue"].apply(lambda x: x[0])
        df["experiencer"] = df["experiencer"].apply(lambda x: x[0])

        # Drop those observations whith multiple cues and experiencers, s.t.
        # we are left with only one span per POS task
        # We lose 472 rows
        df["mask_cue"] = df["cue"].apply(lambda x: True if len(x) <= 1 else float("NaN"))
        df["mask_experiencer"] = df["experiencer"].apply(lambda x: True if len(x) <= 1 else float("NaN"))
        df = df.dropna()

        # Since we only have lists of cues and experiencers of len(1) here, we can safely extract the first element
        df["cue"] = df["cue"].apply(lambda x: x[0] if len(x) else "")  # Note: "" in "some_string" := True
        df["experiencer"] = df["experiencer"].apply(lambda x: x[0] if len(x) else "")

        # Many of the extracted spans do not properly match the true span.
        # Hence, we have to replace that span with the true span in the source text/ headline
        # We drop those rows where we can't match the given span (pattern) with the headline.
        # We lose ~92 rows.
        def replace_span(pattern: str, text: str):
            """Replace a span with it's matching span in the original text."""
            try:
                span = re.search(pattern.lower(), text.lower())
                if not span:
                    return float("NaN")
                return text[span.span()[0] : span.span()[1]]
            except re.error:  # we lose 4 observations here
                return float("NaN")

        df["cue_pos"] = df.apply(lambda x: replace_span(pattern=x.cue, text=x.headline), axis=1)
        df["experiencer_pos"] = df.apply(lambda x: replace_span(pattern=x.experiencer, text=x.headline), axis=1)
        df.rename(columns={"headline": "text"}, inplace=True)
        df = df[["text", "cue_pos", "experiencer_pos"]]
        df = df.dropna()  # we lose ~4 observations here
        df["cue_pos"] = df["cue_pos"].apply(lambda x: self._unify_text(text=x, rm_hashtag=False))
        df["experiencer_pos"] = df["experiencer_pos"].apply(lambda x: self._unify_text(text=x, rm_hashtag=False))

        df = self._clean(df=df, length_threshold=length_threshold, rm_hashtag=False)
        self._log_after_preprocessing(data=df)

        return df

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 42_GoodNewsEveryone."""
        raw_data = pd.read_csv(os.path.join(self._raw_data_local_path, "gne-release-v1.0.tsv"), sep="\t", index_col=0)
        return raw_data
