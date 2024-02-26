"""Execute the 84_EmotionTweets preprocessor."""

from preprocessing.EmotionTweets_84_preprocessor import Preprocessor84EmotionTweets
from storage import storage_client

pp = Preprocessor84EmotionTweets(
    local_path="datasets/84_emotion_tweets", gcs_path="datasets/84_emotion_tweets", storage_client=storage_client,dataset_id=84
)

pp.process()
