"""Execute the 26_NeutralizingBias preprocessor."""

from preprocessing.NeutralizingBias_26_preprocessor import (
    Preprocessor26NeutralizingBias,
)
from storage import storage_client

pp = Preprocessor26NeutralizingBias(
    local_path="datasets/26_neutralizing-bias",
    gcs_path="datasets/26_neutralizing-bias",
    storage_client=storage_client,
    dataset_id=26,
)

pp.process()
