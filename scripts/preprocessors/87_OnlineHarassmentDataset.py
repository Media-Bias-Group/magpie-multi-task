"""Execute the 87_OnlineHarassmentDataset preprocessor."""

from preprocessing.OnlineHarassmentDataset_87_preprocessor import (
    Preprocessor87OnlineHarassmentDataset,
)
from storage import storage_client

pp = Preprocessor87OnlineHarassmentDataset(
    local_path="datasets/87_OnlineHarassmentDataset",
    gcs_path="datasets/87_OnlineHarassmentDataset",
    storage_client=storage_client,
    dataset_id=87,
)

pp.process()
