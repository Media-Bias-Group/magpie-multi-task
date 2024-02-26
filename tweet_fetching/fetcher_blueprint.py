"""This module contains the FetcherBlueprint."""

import os
from typing import Any

import pandas as pd

from storage.storage import StorageClient


class FetcherBlueprint:
    """An abstract class that other Fetchers inherit from."""

    def __init__(self, local_path: str, gcs_path: str, storage_client: StorageClient):
        """Initialize a concrete Fetcher."""
        self._storage_client = storage_client
        self._raw_data_local_path = os.path.join(local_path, "raw")
        self._fetched_data_local_path = os.path.join(local_path, "raw", "fetched.csv")
        self._raw_data_gcs_path = os.path.join(gcs_path, "raw")
        self._fetched_data_gcs_path = os.path.join(gcs_path, "raw", "fetched.csv")

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

    def _save_data_to_local(self, fetched_data: Any):
        """Save the fetched data locally. This method needs to be overwritten in case we have multiple files to save."""
        fetched_data.to_csv(self._fetched_data_local_path, index=False)

    def _fetch(self, raw_data: Any) -> pd.DataFrame:
        """Fetch samples the raw data.

        This needs to be implemented. Typically, this step included sampling from
        the raw data and calling a third party's client to fetch original data.
        """
        raise NotImplementedError

    def process(
        self,
        force_download: bool = False,
        force_upload: bool = False,
        force_fetching: bool = False,
        write_to_local: bool = True,
    ):
        """Process a the fetching step of a dataset.

        This method is the entrypoint of every concrete Dataset Fetcher. It downloads the raw data if necessary from GCS,
        stores it locally, loads it into memory and fetches its original data (optionally a sample from said data).
        After fetching the original data, it saves it locally and uploads it to GCS into the 'raw' directory.
        :param write_to_local: Write preprocessed file to local.
        :param force_fetching: Preprocess the dataset even if it already exists preprocessed in GCS.
        :param force_upload: Upload the preprocessed dataset even if it already exists preprocessed in GCS.
        :param force_download: Download the raw data from GCS even if it already exists locally.
        """
        exists_fetched_gcs = self._storage_client.blob_exists(gcs_path=self._fetched_data_gcs_path)

        if exists_fetched_gcs and not force_fetching:
            # If the dataset is already fetched, and we don't force fetching
            print(
                "Not fetching original data. Original data for this dataset is already fetched in GCS and no force_fetching flag is set."
            )
            return

        self._download_raw_data(force_download=force_download)
        raw_data = self._load_raw_data_from_local()
        fetched_data = self._fetch(raw_data=raw_data)
        if write_to_local:
            self._save_data_to_local(fetched_data=fetched_data)

        if (not exists_fetched_gcs) or force_upload:
            self._storage_client.upload_local_directory_or_file_to_gcs(
                local_path=self._fetched_data_local_path, gcs_path=self._fetched_data_gcs_path
            )
