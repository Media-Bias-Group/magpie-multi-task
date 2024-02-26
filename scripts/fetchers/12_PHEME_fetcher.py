"""Execute the 12_PHEME fetcher."""

from fetching.PHEME_12_fetcher import Fetcher12PHEME
from storage import storage_client

fetcher = Fetcher12PHEME(local_path="datasets/12_PHEME", gcs_path="datasets/12_PHEME", storage_client=storage_client)

fetcher.process()
