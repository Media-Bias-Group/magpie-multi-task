"""Execute the 99_SST2 preprocessor."""

from preprocessing.SST2_99_preprocessor import Preprocessor99StanfordSentiment
from storage import storage_client

pp = Preprocessor99StanfordSentiment(
    local_path="datasets/99_SST2", gcs_path="datasets/99_SST2", storage_client=storage_client, dataset_id=99
)

pp.process()
