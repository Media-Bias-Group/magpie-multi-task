"""Execute the 126_WTWT preprocessor."""

from preprocessing.WTWT_126_preprocessor import Preprocessor126WTWT
from storage import storage_client

pp = Preprocessor126WTWT(
    local_path="datasets/126_WTWT", gcs_path="datasets/126_WTWT", storage_client=storage_client, dataset_id=126
)
pp.process()
