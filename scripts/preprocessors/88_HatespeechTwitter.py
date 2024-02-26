"""Execute the 88_HatespeechTwitter preprocessor."""

from preprocessing.HatespeechTwitter_88_preprocessor import (
    Preprocessor88HatespeechTwitter,
)
from storage import storage_client

pp = Preprocessor88HatespeechTwitter(
    local_path="datasets/88_HatespeechTwitter", gcs_path="datasets/88_HatespeechTwitter", storage_client=storage_client,dataset_id=88
)

pp.process()
