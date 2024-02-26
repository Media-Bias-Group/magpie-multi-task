"""Execute the 125_MultiTargetStance preprocessor."""

from preprocessing.MultiTargetStance_125_preprocessor import (
    Preprocessor125MultiTargetStance,
)
from storage import storage_client

pp = Preprocessor125MultiTargetStance(
    local_path="datasets/125_MultiTargetStance",
    gcs_path="datasets/125_MultiTargetStance",
    storage_client=storage_client,
    dataset_id=125
)
pp.process()
