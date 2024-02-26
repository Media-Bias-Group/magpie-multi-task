"""Execute the 03_CW_HARD preprocessor."""

from preprocessing.CW_HARD_03_preprocessor import Preprocessor03CWHARD
from storage import storage_client

pp = Preprocessor03CWHARD(
    local_path="datasets/03_CW_HARD", gcs_path="datasets/03_CW_HARD", storage_client=storage_client, dataset_id=3
)
pp.process()
