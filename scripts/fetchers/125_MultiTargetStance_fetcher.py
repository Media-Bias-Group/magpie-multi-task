"""Execute the 125_MultiTargetStance fetcher."""

from fetching.MultiTargetStance_125_fetcher import Fetcher125MultiTargetStance
from storage import storage_client

fetcher = Fetcher125MultiTargetStance(
    local_path="datasets/125_MultiTargetStance",
    gcs_path="datasets/125_MultiTargetStance",
    storage_client=storage_client,
)

fetcher.process()
