"""Execute the 80_DebateEffects preprocessor."""

from preprocessing.DebateEffects_80_preprocessor import Preprocessor80DebateEffects
from storage import storage_client

pp = Preprocessor80DebateEffects(
    local_path="datasets/80_DebateEffects",
    gcs_path="datasets/80_DebateEffects",
    storage_client=storage_client,
    dataset_id=80,
)

pp.process()
