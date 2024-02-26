"""Execute the 09_BASIL preprocessor."""

from preprocessing.GoodNewsEveryone_42_preprocessor import (
    Preprocessor42GoodNewsEveryone,
)
from storage import storage_client

pp = Preprocessor42GoodNewsEveryone(
    local_path="datasets/42_GoodNewsEveryone",
    gcs_path="datasets/42_GoodNewsEveryone",
    storage_client=storage_client,
    dataset_id=42,
)

pp.process()
