"""Execute the 105_RtGender preprocessor."""

from preprocessing.RtGender_105_preprocessor import Preprocessor105RtGender
from storage import storage_client

pp = Preprocessor105RtGender(
    local_path="datasets/105_RtGender", gcs_path="datasets/105_RtGender", storage_client=storage_client, dataset_id=105
)

pp.process()
