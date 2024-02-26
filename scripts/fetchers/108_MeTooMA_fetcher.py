"""Execute the 108_MeTooMA fetcher."""

from fetching.MeTooMA_108_fetcher import Fetcher108MeTooMA
from storage import storage_client

fetcher = Fetcher108MeTooMA(
    local_path="datasets/108_MeTooMA", gcs_path="datasets/108_MeTooMA", storage_client=storage_client
)

fetcher.process()
