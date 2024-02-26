import pytest
from pytest_mock import MockerFixture

from fetching.fetcher_blueprint import FetcherBlueprint
from storage.storage import StorageClient


class FetcherBlueprintTests:
    def test__download_raw_data(self):
        # 1:1 the same as in Preprocessor-Blueprint, no need to test twice
        assert True

    def test__load_raw_data_from_local(self, mocker: MockerFixture):
        storage_client_mock = mocker.patch.object(StorageClient, "__init__")
        blueprint_fetcher = FetcherBlueprint(local_path="", gcs_path="", storage_client=storage_client_mock)

        with pytest.raises(NotImplementedError):
            blueprint_fetcher._load_raw_data_from_local()

    # def test__save_data_to_local(self):
    #     assert False

    def test__fetch(self, mocker: MockerFixture):
        storage_client_mock = mocker.patch.object(StorageClient, "__init__")
        blueprint_fetcher = FetcherBlueprint(local_path="", gcs_path="", storage_client=storage_client_mock)
        with pytest.raises(NotImplementedError):
            blueprint_fetcher._fetch(raw_data=None)

    def test__fetch_fetched_already_exists_and_not_force_fetching(self, mocker: MockerFixture):
        # storage_client = StorageClient()
        storage_client_mock = mocker.patch.object(StorageClient, "__init__")

        blueprint_fetcher = FetcherBlueprint(local_path="", gcs_path="", storage_client=storage_client_mock)

        # Test 1: If a version of fetched original data is already processed in GCS and we don't force fetching
        blob_exists_mock = mocker.patch.object(StorageClient, "blob_exists")
        blob_exists_mock.return_value = True
        blueprint_fetcher_download_raw_data_mock = mocker.patch.object(FetcherBlueprint, "_download_raw_data")
        blueprint_fetcher_load_raw_data_from_local_mock = mocker.patch.object(
            FetcherBlueprint, "_load_raw_data_from_local"
        )
        blueprint_fetcher_fetch_mock = mocker.patch.object(FetcherBlueprint, "_fetch")
        blueprint_fetcher_save_data_to_local_mock = mocker.patch.object(FetcherBlueprint, "_save_data_to_local")
        storage_client_upload_local_directory_or_file_to_gcs_mock = mocker.patch.object(
            storage_client_mock, "upload_local_directory_or_file_to_gcs"
        )

        blueprint_fetcher.process(force_upload=False, force_download=False, force_fetching=False, write_to_local=False)
        blueprint_fetcher_download_raw_data_mock.assert_not_called()
        blueprint_fetcher_load_raw_data_from_local_mock.assert_not_called()
        blueprint_fetcher_fetch_mock.assert_not_called()
        blueprint_fetcher_save_data_to_local_mock.assert_not_called()
        storage_client_upload_local_directory_or_file_to_gcs_mock.assert_not_called()

    def test__fetch_fetched_already_exists_and_force_fetching(self, mocker: MockerFixture):
        storage_client_mock = mocker.patch.object(StorageClient, "__init__")

        blueprint_fetcher = FetcherBlueprint(local_path="", gcs_path="", storage_client=storage_client_mock)

        # Test 1: If a version of fetched original data is already processed in GCS and we don't force fetching
        blob_exists_mock = mocker.patch.object(StorageClient, "blob_exists")
        blob_exists_mock.return_value = True
        blueprint_fetcher_download_raw_data_mock = mocker.patch.object(FetcherBlueprint, "_download_raw_data")
        blueprint_fetcher_load_raw_data_from_local_mock = mocker.patch.object(
            FetcherBlueprint, "_load_raw_data_from_local"
        )
        blueprint_fetcher_fetch_mock = mocker.patch.object(FetcherBlueprint, "_fetch")
        blueprint_fetcher_save_data_to_local_mock = mocker.patch.object(FetcherBlueprint, "_save_data_to_local")
        storage_client_upload_local_directory_or_file_to_gcs_mock = mocker.patch.object(
            storage_client_mock, "upload_local_directory_or_file_to_gcs"
        )

        blueprint_fetcher.process(force_upload=False, force_fetching=True)
        blueprint_fetcher_download_raw_data_mock.assert_called_once()
        blueprint_fetcher_load_raw_data_from_local_mock.assert_called_once()
        blueprint_fetcher_fetch_mock.assert_called_once()
        blueprint_fetcher_save_data_to_local_mock.assert_called_once()
        storage_client_upload_local_directory_or_file_to_gcs_mock.assert_not_called()

    def test_process_force_upload_exists_gcs(self, mocker: MockerFixture):
        storage_client_mock = mocker.patch.object(StorageClient, "__init__")
        blueprint_fetcher = FetcherBlueprint(
            local_path="",
            gcs_path="",
            storage_client=storage_client_mock,
        )

        # Force uploading even if processed file already exists in GCS.
        blob_exists_mock = mocker.patch.object(storage_client_mock, "blob_exists")
        blob_exists_mock.return_value = True
        blueprint_fetcher_download_raw_data_mock = mocker.patch.object(FetcherBlueprint, "_download_raw_data")
        blueprint_fetcher_load_raw_data_from_local_mock = mocker.patch.object(
            FetcherBlueprint, "_load_raw_data_from_local"
        )
        blueprint_fetcher_fetch_mock = mocker.patch.object(FetcherBlueprint, "_fetch")
        storage_client_upload_local_directory_or_file_to_gcs_mock = mocker.patch.object(
            storage_client_mock, "upload_local_directory_or_file_to_gcs"
        )

        blueprint_fetcher.process(force_upload=True, force_fetching=True)
        blueprint_fetcher_download_raw_data_mock.assert_called_once()
        blueprint_fetcher_load_raw_data_from_local_mock.assert_called_once()
        blueprint_fetcher_fetch_mock.assert_called_once()
        storage_client_upload_local_directory_or_file_to_gcs_mock.assert_called_once()  # must be uploaded
