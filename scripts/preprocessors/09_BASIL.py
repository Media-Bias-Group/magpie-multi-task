"""Execute the 09_BASIL preprocessor."""

from preprocessing.BASIL_09_preprocessor import Preprocessor09Basil
from storage import storage_client

pp = Preprocessor09Basil(
    local_path="datasets/9_BASIL", gcs_path="datasets/9_BASIL", storage_client=storage_client, dataset_id=9
)

pp.process()
