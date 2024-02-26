"""Execute the 104_TRAC_2 preprocessor."""

from preprocessing.Trac_2_104_preprocessor import Preprocessor104TRAC2
from storage import storage_client

pp = Preprocessor104TRAC2(
    local_path="datasets/104_TRAC2", gcs_path="datasets/104_TRAC2", storage_client=storage_client,dataset_id=104
)

pp.process()
