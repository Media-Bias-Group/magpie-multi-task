"""Execute the 112_BLM_Harvard fetcher."""

from fetching.BLM_Harvard_112_fetcher import Fetcher112BLMharvard
from storage import storage_client

fetcher = Fetcher112BLMharvard(
    local_path="datasets/112_BLM-ALM-HARVARD", gcs_path="datasets/112_BLM-ALM-HARVARD", storage_client=storage_client
)

fetcher.process()
