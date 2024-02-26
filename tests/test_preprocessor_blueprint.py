import pytest
from pytest_mock import MockerFixture

from preprocessing.preprocessors import PreprocessorBlueprint
from storage.storage import StorageClient


class PreprocessorBlueprintTests:
    def test__download_raw_data(self, mocker: MockerFixture):
        storage_client = StorageClient()
        storage_client_download_from_gcs_to_local_directory_or_file_mock = mocker.patch.object(
            StorageClient, "download_from_gcs_to_local_directory_or_file"
        )
        blueprint_preprocessor = PreprocessorBlueprint(local_path="", gcs_path="", storage_client=storage_client)

        # Assert that file(s) where downloaded if force download is activated
        blueprint_preprocessor._download_raw_data(force_download=True)
        storage_client_download_from_gcs_to_local_directory_or_file_mock.assert_called_once()

        # TODO: Add more tests

    def test__load_raw_data_from_local(self, mocker: MockerFixture):
        storage_client_mock = mocker.patch.object(StorageClient, "__init__")
        blueprint_preprocessor = PreprocessorBlueprint(local_path="", gcs_path="", storage_client=storage_client_mock)

        with pytest.raises(NotImplementedError):
            blueprint_preprocessor._load_raw_data_from_local()

    def test__preprocess(self, mocker: MockerFixture):
        storage_client_mock = mocker.patch.object(StorageClient, "__init__")
        blueprint_preprocessor = PreprocessorBlueprint(local_path="", gcs_path="", storage_client=storage_client_mock)
        with pytest.raises(NotImplementedError):
            blueprint_preprocessor._preprocess(raw_data=None, length_threshold=20)

    def test_process_not_force_preprocessing(self, mocker: MockerFixture):
        storage_client_mock = mocker.patch.object(StorageClient, "__init__")
        blueprint_preprocessor = PreprocessorBlueprint(local_path="", gcs_path="", storage_client=storage_client_mock)

        # Test 1: If the DS is already processed in GCS and we don't force preprocessing
        blob_exists_mock = mocker.patch.object(StorageClient, "blob_exists")
        blob_exists_mock.return_value = True
        blueprint_preprocessor_download_raw_data_mock = mocker.patch.object(
            PreprocessorBlueprint, "_download_raw_data"
        )
        blueprint_preprocessor_load_raw_data_from_local_mock = mocker.patch.object(
            PreprocessorBlueprint, "_load_raw_data_from_local"
        )
        blueprint_preprocessor_preprocess_mock = mocker.patch.object(PreprocessorBlueprint, "_preprocess")
        blueprint_preprocessor_save_data_to_local_mock = mocker.patch.object(
            PreprocessorBlueprint, "_save_data_to_local"
        )
        storage_client_upload_local_directory_or_file_to_gcs_mock = mocker.patch.object(
            storage_client_mock, "upload_local_directory_or_file_to_gcs"
        )

        blueprint_preprocessor.process(
            force_preprocessing=False, force_upload=False, force_download=False, write_to_local=False
        )
        blueprint_preprocessor_download_raw_data_mock.assert_not_called()
        blueprint_preprocessor_load_raw_data_from_local_mock.assert_not_called()
        blueprint_preprocessor_preprocess_mock.assert_not_called()
        blueprint_preprocessor_save_data_to_local_mock.assert_not_called()
        storage_client_upload_local_directory_or_file_to_gcs_mock.assert_not_called()

    def test_process_force_preprocessing_exists_gcs(self, mocker: MockerFixture):
        storage_client_mock = mocker.patch.object(StorageClient, "__init__")
        blueprint_preprocessor = PreprocessorBlueprint(local_path="", gcs_path="", storage_client=storage_client_mock)

        # Test 2: If the DS is already processed in GCS, and we force preprocessing.
        blob_exists_mock = mocker.patch.object(StorageClient, "blob_exists")
        blob_exists_mock.return_value = True
        blueprint_preprocessor_download_raw_data_mock = mocker.patch.object(
            PreprocessorBlueprint, "_download_raw_data"
        )
        blueprint_preprocessor_load_raw_data_from_local_mock = mocker.patch.object(
            PreprocessorBlueprint, "_load_raw_data_from_local"
        )
        blueprint_preprocessor_preprocess_mock = mocker.patch.object(PreprocessorBlueprint, "_preprocess")
        blueprint_preprocessor_save_data_to_local_mock = mocker.patch.object(
            PreprocessorBlueprint, "_save_data_to_local"
        )

        storage_client_upload_local_directory_or_file_to_gcs_mock = mocker.patch.object(
            StorageClient, "upload_local_directory_or_file_to_gcs"
        )

        blueprint_preprocessor.process(force_preprocessing=True)
        blueprint_preprocessor_download_raw_data_mock.assert_called_once()
        blueprint_preprocessor_load_raw_data_from_local_mock.assert_called_once()
        blueprint_preprocessor_preprocess_mock.assert_called_once()
        blueprint_preprocessor_save_data_to_local_mock.assert_called_once()
        storage_client_upload_local_directory_or_file_to_gcs_mock.assert_not_called()

    def test_process_force_preprocessing_not_exists_gcs(self, mocker: MockerFixture):
        # Test if force_preprocessing works, i.e. if preprocess will be executed even if processed already exists
        # Also tests that this file will be uploaded, as it does not exist in GCS yet.
        storage_client_mock = mocker.patch.object(StorageClient, "__init__")
        blueprint_preprocessor = PreprocessorBlueprint(local_path="", gcs_path="", storage_client=storage_client_mock)

        # Test 3: If the DS is not yet processed in GCS, and we force preprocessing.
        blob_exists_mock = mocker.patch.object(storage_client_mock, "blob_exists")
        blob_exists_mock.return_value = False
        blueprint_preprocessor_download_raw_data_mock = mocker.patch.object(
            PreprocessorBlueprint, "_download_raw_data"
        )
        blueprint_preprocessor_load_raw_data_from_local_mock = mocker.patch.object(
            PreprocessorBlueprint, "_load_raw_data_from_local"
        )
        blueprint_preprocessor_preprocess_mock = mocker.patch.object(PreprocessorBlueprint, "_preprocess")
        storage_client_upload_local_directory_or_file_to_gcs_mock = mocker.patch.object(
            storage_client_mock, "upload_local_directory_or_file_to_gcs"
        )

        blueprint_preprocessor.process(force_preprocessing=True)
        blueprint_preprocessor_download_raw_data_mock.assert_called_once()
        blueprint_preprocessor_load_raw_data_from_local_mock.assert_called_once()
        blueprint_preprocessor_preprocess_mock.assert_called_once()
        storage_client_upload_local_directory_or_file_to_gcs_mock.assert_called()  # must be uploaded

    def test_process_force_upload_exists_gcs(self, mocker: MockerFixture):
        storage_client_mock = mocker.patch.object(StorageClient, "__init__")
        blueprint_preprocessor = PreprocessorBlueprint(
            local_path="",
            gcs_path="",
            storage_client=storage_client_mock,
        )

        # Force uploading even if processed file already exists in GCS.
        blob_exists_mock = mocker.patch.object(storage_client_mock, "blob_exists")
        blob_exists_mock.return_value = True
        blueprint_preprocessor_download_raw_data_mock = mocker.patch.object(
            PreprocessorBlueprint, "_download_raw_data"
        )
        blueprint_preprocessor_load_raw_data_from_local_mock = mocker.patch.object(
            PreprocessorBlueprint, "_load_raw_data_from_local"
        )
        blueprint_preprocessor_preprocess_mock = mocker.patch.object(PreprocessorBlueprint, "_preprocess")
        storage_client_upload_local_directory_or_file_to_gcs_mock = mocker.patch.object(
            storage_client_mock, "upload_local_directory_or_file_to_gcs"
        )

        blueprint_preprocessor.process(force_upload=True, force_preprocessing=True)
        blueprint_preprocessor_download_raw_data_mock.assert_called_once()
        blueprint_preprocessor_load_raw_data_from_local_mock.assert_called_once()
        blueprint_preprocessor_preprocess_mock.assert_called_once()
        storage_client_upload_local_directory_or_file_to_gcs_mock.assert_called()  # must be uploaded
