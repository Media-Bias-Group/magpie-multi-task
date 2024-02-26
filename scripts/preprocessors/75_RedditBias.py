"""Execute the 75_RedditBias preprocessor."""

from preprocessing.RedditBias_75_preprocessor import Preprocessor75RedditBias
from storage import storage_client

pp = Preprocessor75RedditBias(
    local_path="datasets/75_RedditBias",
    gcs_path="datasets/75_RedditBias",
    storage_client=storage_client,
    dataset_id=75,
)

pp.process()
