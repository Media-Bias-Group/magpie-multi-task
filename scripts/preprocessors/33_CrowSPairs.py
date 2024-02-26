"""Execute the 33_CrowSPairs preprocessor."""

from preprocessing.CrowSPairs_33_preprocessor import Preprocessor33CrowSPairs
from storage import storage_client

pp = Preprocessor33CrowSPairs(
    local_path="datasets/33_CrowSPairs",
    gcs_path="datasets/33_CrowSPairs",
    storage_client=storage_client,
    dataset_id=33,
)

pp.process()
