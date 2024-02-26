"""Execute the 18_GAP preprocessor."""

from preprocessing.GAP_18_preprocessor import Preprocessor18GAP
from storage import storage_client

pp = Preprocessor18GAP(
    local_path="datasets/18_GAP", gcs_path="datasets/18_GAP", storage_client=storage_client, dataset_id=18
)
pp.process()
