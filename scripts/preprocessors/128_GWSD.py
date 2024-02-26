"""Execute the 128_GWSD preprocessor."""

from preprocessing.GWSD_128_preprocessor import Preprocessor128GWSD
from storage import storage_client

pp = Preprocessor128GWSD(local_path="datasets/128_GWSD", gcs_path="datasets/128_GWSD", storage_client=storage_client,dataset_id=128)
pp.process()
