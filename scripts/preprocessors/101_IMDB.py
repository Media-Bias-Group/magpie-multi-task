"""Execute the 101_IMDB preprocessor."""

from preprocessing.IMDB_101_preprocessor import Preprocessor101IMDB
from storage import storage_client

pp = Preprocessor101IMDB(
    local_path="datasets/101_IMDB", gcs_path="datasets/101_IMDB", storage_client=storage_client, dataset_id=101
)

pp.process()
