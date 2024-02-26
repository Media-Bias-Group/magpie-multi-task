"""Execute the 117_Funpedia preprocessor."""

from preprocessing.Funpedia_117_preprocessor import Preprocessor117Funpedia
from storage import storage_client

pp = Preprocessor117Funpedia(
    local_path="datasets/117_Funpedia", gcs_path="datasets/117_Funpedia", storage_client=storage_client, dataset_id=117
)

pp.process()
