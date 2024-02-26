"""Execute the 107_CMSB preprocessor."""

from preprocessing.CMSB_107_preprocessor import Preprocessor107CSMB
from storage import storage_client

pp = Preprocessor107CSMB(
    local_path="datasets/107_CallMeSexistBut", gcs_path="datasets/107_CallMeSexistBut", storage_client=storage_client,dataset_id=107
)

pp.process()
