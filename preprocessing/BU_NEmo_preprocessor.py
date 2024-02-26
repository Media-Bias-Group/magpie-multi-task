"""This module contains all the Preprocessing classes."""

import os
from typing import List

import numpy as np
import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint

MAPPING = {
    "Amusement": 0,
    "Anger": 1,
    "Awe": 2,
    "Contentment": 3,
    "Disgust": 4,
    "Excitement": 5,
    "Fear": 6,
    "Sadness": 7,
}


class Preprocessor96BUNEmo(PreprocessorBlueprint):
    """Preprocessor for the 22 News WCL 50 dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor96BUNEmo."""
        super(Preprocessor96BUNEmo, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: pd.DataFrame, length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 96_BU-NEmo."""
        # different annotations scheme, overlaping but responses are different
        txt = raw_data["text_only"].to_numpy()
        img = raw_data["image_only"].to_numpy()
        txt_img = raw_data["text_image"].to_numpy()

        # mask out NaNs
        mask_txt = np.where(txt == txt)
        mask_img = np.where(img == img)
        mask_txt_img = np.where(txt_img == txt_img)

        # extract all the non-NaN responses as a list of dictionaries
        txt_responses = [resp for responses in txt[mask_txt] for resp in responses["responses"]]
        img_responses = [resp for responses in img[mask_img] for resp in responses["responses"]]
        txt_img_responses = [resp for responses in txt_img[mask_txt_img] for resp in responses["responses"]]

        all_responses = txt_responses + img_responses + txt_img_responses

        self._log_before_preprocessing(data=all_responses)

        all_df = pd.DataFrame(all_responses)
        all_df.drop(
            ["annotator_politics", "annotator_media_time", "url", "intensity", "feeling"], axis=1, inplace=True
        )
        all_df.rename(columns={"emotion": "label", "reason": "text"}, inplace=True)
        all_df["label"] = all_df.label.map(MAPPING)

        cleaned = self._clean(all_df, length_threshold=length_threshold)
        self._log_after_preprocessing(data=cleaned)
        self.set_logger_data("primary_label_distribution", cleaned["label"].value_counts().to_dict())
        self.set_logger_data("additional_data", MAPPING)
        return cleaned

    def _load_raw_data_from_local(self) -> pd.DataFrame:
        """Load the raw data of 96_BU-NEmo."""
        df = pd.read_json(os.path.join(self._raw_data_local_path, "data_all.json"))
        return df

    def _log_before_preprocessing(self, data: List):
        self.set_logger_data("original_size", len(data))
