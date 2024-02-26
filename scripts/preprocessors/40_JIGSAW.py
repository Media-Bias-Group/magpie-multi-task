"""Execute the 40_JIGSAW preprocessor."""

from preprocessing.JIGSAW_40_preprocessor import Preprocessor40JIGSAW
from storage import storage_client

pp = Preprocessor40JIGSAW(
    local_path="datasets/40_JIGSAW", gcs_path="datasets/40_JIGSAW", storage_client=storage_client, dataset_id=40
)

pp.process()
