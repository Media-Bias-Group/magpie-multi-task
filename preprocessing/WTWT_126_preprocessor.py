"""This module contains the Preprocessor for the 126_WTWT dataset."""

import os

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint

MAPPING = dict(zip(["comment", "unrelated", "refute", "support"], [1, 1, 0, 2]))
COMPANY_MAPPING = {
    "FOXA_DIS": "Disney wants to buy 21st Century Fox.",
    "CVS_AET": "CVS Health wants to buy Aetna.",
    "ANTM_CI": "Anthem wants to buy Cigna.",
    "AET_HUM": "Aetna wants to buy Humana.",
    "CI_ESRX": "Cigna wants to buy Express Scripts.",
}


class Preprocessor126WTWT(PreprocessorBlueprint):
    """Preprocessor for the 126_WTWT dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor126WTWT."""
        super(Preprocessor126WTWT, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: pd.DataFrame, length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 126_WTWT."""
        self._log_before_preprocessing(data=raw_data)

        df = raw_data.rename(columns={"merger": "label_target", "stance": "label_stance"})
        df = df[df["label_stance"] != "unrelated"]
        df["label_stance"] = df.label_stance.map(MAPPING)
        df = df.dropna()
        df["label_stance"] = df["label_stance"].astype(int)
        df["text"] = df["label_target"].map(COMPANY_MAPPING) + " " + df.text
        label_target_distribution = df.label_target.value_counts().to_dict()

        df = df[["text", "label_stance"]].rename(columns={"label_stance": "label"})

        df["text"] = df["text"].astype(str)
        df = self._clean(df=df, length_threshold=length_threshold)
        self._log_after_preprocessing(data=df)
        self.set_logger_data(key="primary_label_distribution", value=df.label.value_counts().to_dict())
        self.set_logger_data(
            key="additional_data",
            value={
                "MAPPING": MAPPING,
                "label_target_distribution": label_target_distribution,
                "COMPANY_MAPPING": COMPANY_MAPPING,
            },
        )

        return df

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 126_WTWT."""
        return pd.read_csv(os.path.join(self._raw_data_local_path, "fetched.csv"))
