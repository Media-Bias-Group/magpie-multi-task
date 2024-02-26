"""Execute the 22_NewsWCL preprocessor."""

from preprocessing.NEWS_WCL_22_preprocessor import Preprocessor22NewsWCL
from storage import storage_client

pp = Preprocessor22NewsWCL(
    local_path="datasets/22_NewsWCL50", gcs_path="datasets/22_NewsWCL50", storage_client=storage_client,dataset_id=22
)

pp.process()
