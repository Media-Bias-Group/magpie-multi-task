"""Execute the 38_Starbucks preprocessor."""

from preprocessing.Starbucks_38_preprocessor import Preprocessor38Starbucks
from storage import storage_client

pp = Preprocessor38Starbucks(
    local_path="datasets/38_starbucks", gcs_path="datasets/38_starbucks", storage_client=storage_client,dataset_id=38
)

pp.process()
