"""Execute the 92_HateXplain preprocessor."""

from preprocessing.HateXplain_92_preprocessor import Preprocessor92HateXplain
from storage import storage_client

pp = Preprocessor92HateXplain(
    local_path="datasets/92_HateXplain",
    gcs_path="datasets/92_HateXplain",
    storage_client=storage_client,
    dataset_id=92,
)

pp.process()
