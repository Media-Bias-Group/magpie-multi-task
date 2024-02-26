"""Execute the 84_EmotionTweets fetcher."""

from fetching.EmotionTweets_84_fetcher import Fetcher84EmotionTweets
from storage import storage_client

fetcher = Fetcher84EmotionTweets(
    local_path="datasets/84_emotion_tweets", gcs_path="datasets/84_emotion_tweets", storage_client=storage_client
)

fetcher.process()
