"""Execute the 88_HatespeechTwitter fetcher."""

from fetching.HatespeechTwitter_88_fetcher import Fetcher88HatespeechTwitter
from storage import storage_client

fetcher = Fetcher88HatespeechTwitter(
    local_path="datasets/88_HatespeechTwitter", gcs_path="datasets/88_HatespeechTwitter", storage_client=storage_client
)

fetcher.process()
