"""Execute the 25_FakeNewsNet preprocessor."""

from preprocessing.FakeNewsNet_25_preprocessor import Preprocessor25FakeNewsNet
from storage import storage_client

pp = Preprocessor25FakeNewsNet(
    local_path="datasets/25_FakeNewsNet", gcs_path="datasets/25_FakeNewsNet", storage_client=storage_client,dataset_id=25
)

pp.process()
