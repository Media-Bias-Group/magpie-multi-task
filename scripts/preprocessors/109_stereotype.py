"""Execute the 109_stereotype preprocessor."""

from preprocessing.stereotype_109_preprocessor import Preprocessor109Stereotype
from storage import storage_client

pp = Preprocessor109Stereotype(
    local_path="datasets/109_stereotype",
    gcs_path="datasets/109_stereotype",
    storage_client=storage_client,
    dataset_id=109,
)

pp.process()
