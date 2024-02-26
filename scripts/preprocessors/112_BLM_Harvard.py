"""Execute the 112_BLM_Harvard preprocessor."""

from preprocessing.BLM_Harvard_112_processor import Preprocessor112BLMHarvard
from storage import storage_client

pp = Preprocessor112BLMHarvard(
    local_path="datasets/112_BLM-ALM-HARVARD", gcs_path="datasets/112_BLM-ALM-HARVARD", storage_client=storage_client,dataset_id=112
)

pp.process()
