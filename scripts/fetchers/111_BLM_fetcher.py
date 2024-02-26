"""Execute the 111_BLM fetcher."""

from fetching.BLM_111_fetcher import Fetcher111BLM
from storage import storage_client

fetcher = Fetcher111BLM(
    local_path="datasets/111_BLM-ALM", gcs_path="datasets/111_BLM-ALM", storage_client=storage_client
)

fetcher.process()
