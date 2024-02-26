"""This module contains the PreprocesserBlueprint."""

import json
import os
import re
from typing import Any, Dict

import numpy as np
import pandas as pd

from storage.storage import StorageClient


class PreprocessorBlueprint:
    """An abstract class that other Preprocessors inherit from."""

    def __init__(self, local_path: str, gcs_path: str, storage_client: StorageClient, dataset_id: int = 0):
        """Initialize a concrete Preprocessor."""
        self._storage_client = storage_client
        self._raw_data_local_path = os.path.join(local_path, "raw")
        self._processed_data_local_path = os.path.join(local_path, "preprocessed.csv")
        self._logger_data_local_path = os.path.join(local_path, "metadata.json")
        self._raw_data_gcs_path = os.path.join(gcs_path, "raw")
        self._processed_data_gcs_path = os.path.join(gcs_path, "preprocessed.csv")
        self._logger_data_gcs_path = os.path.join("datasets", "logging", f"{dataset_id}-metadata.json")

        self._logger_data: Dict = {
            "dataset_id": dataset_id,
            "original_size": None,
            "final_size": None,
            "primary_label_distribution": None,
            "additional_data": {},
            "length_of_text_distribution": None,
        }

    def _unify_text(self, text, rm_hashtag: bool = False):
        """Unify text e.g. remove URLs, lowercase, etc.

        Use this method whenever we want to 'unify' text.
        :param rm_hashtag: Flag if remove hashtags from text.
        :param text: A plain text as string.
        :return: The unified text as string.
        """

        if text != text:
            return text

        if rm_hashtag:
            text = re.sub(r"#[A-Za-z0-9_]+", "", text)  # remove #hashtag

        text = re.sub(r"RT\ ", " ", text)  # remove 'RT' from tweets
        text = re.sub(r"@[A-Za-z0-9_]+", " ", text)  # remove @user
        text = re.sub(r"https?://[A-Za-z0-9./]+", " ", text)  # remove links
        text = re.sub("\t", " ", text)  # remove tab
        text = re.sub("\n", " ", text)  # remove newlines
        text = re.sub("\r", " ", text)  # remove \r type newlines
        text = re.sub(r" +", " ", text)  # remove multiple whitespaces
        text = re.sub(r"linebreak", "", text)  # remove linebreaks
        return text

    def _clean(self, df: pd.DataFrame, rm_hashtag: bool = False, length_threshold: int = 20) -> pd.DataFrame:
        """Clean the dataframe in unified way, usable for all datasets.

        Args:
            df (pd.DataFrame): data in DataFrame
            rm_hashtag (bool, optional): flag if remove hashtags from text. Defaults to False.
            length_threshold (int): Set shortest length parameter for sentences.

        Returns:
            pd.DataFrame: Cleaned dataframe.
        """
        # 1: clean text
        def clean_text(x):
            return self._unify_text(x, rm_hashtag=rm_hashtag)

        tmp = df.text.apply(lambda x: len(x.split(" ")))
        self.set_logger_data(
            key="length_of_text_distribution_before_cleaning",
            value={
                **tmp.describe().to_dict(),
                **{"95%": np.quantile(tmp, 0.95)},
                **{"90%": np.quantile(tmp, 0.9)},
                **{"99%": np.quantile(tmp, 0.99)},
            },
        )

        df["text"] = df["text"].apply(clean_text)

        # 2.1: discard short sentences
        df["length"] = df["text"].apply(lambda x: len(x))
        df = df[length_threshold <= df["length"]].drop("length", axis=1)

        # 2.2 Discard long sentences
        df["length"] = df["text"].apply(lambda x: len(x.split(" ")))
        df = df[df["length"] <= 128].drop("length", axis=1)

        # 3: remove duplicates (possibly created in step 1)
        duplicate_indices = df.duplicated()
        df = df[~duplicate_indices]

        # 4: sampling
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df = df.dropna()

        return df

    def _download_raw_data(self, force_download: bool):
        """Download the raw data from GCS.

        :param force_download: Download the raw data even if it already exists locally.
        """
        if not os.path.exists(self._raw_data_local_path) or force_download:
            self._storage_client.download_from_gcs_to_local_directory_or_file(
                local_path="", gcs_path=self._raw_data_gcs_path
            )

    def _load_raw_data_from_local(self):
        """Load the raw data from local directory. This needs to be implemented."""
        raise NotImplementedError

    def _preprocess(self, raw_data: Any, length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data. This needs to be implemented."""
        raise NotImplementedError

    def _save_data_to_local(self, processed_data: Any):
        """Save the processed data locally. This method needs to be overwritten in case we have multiple files to save."""
        processed_data.to_csv(self._processed_data_local_path, index=False)

    def _save_logger_data_to_local(self):
        """Log some information about that dataset."""
        with open(self._logger_data_local_path, "w") as f:
            json.dump(self._logger_data, f)

    def process(
        self,
        force_download: bool = True,
        force_upload: bool = True,
        force_preprocessing: bool = True,
        write_to_local: bool = True,
        length_threshold: int = 20,
    ):
        """Process a dataset.

        This method is the entrypoint of every concrete Dataset preprocessor. It downloads the raw data if necessary,
        stores it locally, loads it into memory and preprocesses it. After preprocessing, it saves it locally and
        uploads it to GCS.
        :param length_threshold: (int, optional): Set shortest length parameter for sentences. Defaults to 20.
        :param write_to_local: Write preprocessed file to local.
        :param force_preprocessing: Preprocess the dataset even if it already exists preprocessed in GCS.
        :param force_upload: Upload the preprocessed dataset even if it already exists preprocessed in GCS.
        :param force_download: Download the raw data even if it already exists locally.
        """
        exists_preprocessed_gcs = self._storage_client.blob_exists(gcs_path=self._processed_data_gcs_path)

        if exists_preprocessed_gcs and not force_preprocessing:
            # If the dataset is already processed, and we don't force preprocessing
            print("Not processing as the dataset is already processed in GCS and no force_preprocessing flag is set.")
            return

        self._download_raw_data(force_download=force_download)
        raw_data = self._load_raw_data_from_local()
        processed_data = self._preprocess(raw_data=raw_data, length_threshold=length_threshold)

        if write_to_local:
            self._save_data_to_local(processed_data=processed_data)
            self._save_logger_data_to_local()

        if (not exists_preprocessed_gcs) or force_upload:
            self._storage_client.upload_local_directory_or_file_to_gcs(
                local_path=self._processed_data_local_path, gcs_path=self._processed_data_gcs_path
            )
            self._storage_client.upload_local_directory_or_file_to_gcs(
                local_path=self._logger_data_local_path, gcs_path=self._logger_data_gcs_path
            )

    def set_logger_data(self, key: str, value: Any) -> None:
        """Set the logger data."""
        assert key in [
            "original_size",
            "final_size",
            "primary_label_distribution",
            "additional_data",
            "length_of_text_distribution",
            "length_of_text_distribution_before_cleaning",
        ]
        self._logger_data[key] = value

    def _log_before_preprocessing(self, data: Any):
        self.set_logger_data("original_size", len(data))

    def _log_after_preprocessing(self, data: Any):
        tmp = data.text.apply(lambda x: len(x.split(" ")))
        self.set_logger_data(
            key="length_of_text_distribution",
            value={
                **tmp.describe().to_dict(),
                **{"95%": np.quantile(tmp, 0.95)},
                **{"90%": np.quantile(tmp, 0.9)},
                **{"99%": np.quantile(tmp, 0.99)},
            },
        )
        self.set_logger_data("final_size", len(data))
