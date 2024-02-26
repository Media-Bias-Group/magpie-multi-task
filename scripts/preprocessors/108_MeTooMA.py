"""Execute the 108_MeTooMA preprocessor."""


from preprocessing.MeTooMA_108_preprocessor import Preprocessor108MeTooMA
from storage import storage_client

pp = Preprocessor108MeTooMA(
    local_path="datasets/108_MeTooMA", gcs_path="datasets/108_MeTooMA", storage_client=storage_client, dataset_id=108
)

pp.process()
