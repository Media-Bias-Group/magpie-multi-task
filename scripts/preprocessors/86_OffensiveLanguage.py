"""Execute the 86_OffensiveLanguage preprocessor."""

from preprocessing.OffensiveLanguage_86_preprocessor import (
    Preprocessor86OffensiveLanguage,
)
from storage import storage_client

pp = Preprocessor86OffensiveLanguage(
    local_path="datasets/86_OffensiveLanguage",
    gcs_path="datasets/86_OffensiveLanguage",
    storage_client=storage_client,
    dataset_id=86,
)

pp.process()
