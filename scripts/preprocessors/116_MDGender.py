"""Execute the 116_MDGender preprocessor."""

from preprocessing.MDGender_116_preprocessor import Preprocessor116MDGender
from storage import storage_client

pp = Preprocessor116MDGender(
    local_path="datasets/116_MDGender", gcs_path="datasets/116_MDGender", storage_client=storage_client, dataset_id=116
)

pp.process()
