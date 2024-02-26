"""Execute the 111_BLM preprocessor."""

from preprocessing.BLM_111_preprocessor import Preprocessor111BLM
from storage import storage_client

pp = Preprocessor111BLM(
    local_path="datasets/111_BLM-ALM", gcs_path="datasets/111_BLM-ALM", storage_client=storage_client,dataset_id=111
)

pp.process()
