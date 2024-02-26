"""Execute the 126_WTWT fetcher."""

from fetching.WTWT_126_fetcher import Fetcher126WTWT
from storage import storage_client

fetcher = Fetcher126WTWT(local_path="datasets/126_WTWT", gcs_path="datasets/126_WTWT", storage_client=storage_client)

fetcher.process()
