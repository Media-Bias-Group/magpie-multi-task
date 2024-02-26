"""Execute the 12_PHEME preprocessor."""

from preprocessing.PHEME_12_preprocessor import Preprocessor12PHEME
from storage import storage_client

pp = Preprocessor12PHEME(
    local_path="datasets/12_PHEME", gcs_path="datasets/12_PHEME", storage_client=storage_client, dataset_id=12
)
pp.process()
