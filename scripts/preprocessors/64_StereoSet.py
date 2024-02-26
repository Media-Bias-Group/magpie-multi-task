"""Execute the 64_StereoSet preprocessor."""

from preprocessing.StereoSet_64_preprocessor import Preprocessor64StereoSet
from storage import storage_client

pp = Preprocessor64StereoSet(
    local_path="datasets/64_StereoSet", gcs_path="datasets/64_StereoSet", storage_client=storage_client, dataset_id=64
)

pp.process()
