"""Execute the 10_BABE preprocessor."""

from preprocessing.BABE_10_preprocessor import Preprocessor10BABE
from storage import storage_client

pp = Preprocessor10BABE(local_path="datasets/10_BABE", gcs_path="datasets/10_BABE", storage_client=storage_client,dataset_id=10)
pp.process()
