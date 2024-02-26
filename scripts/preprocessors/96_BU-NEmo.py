"""Execute the 22_NewsWCL preprocessor."""

from preprocessing.BU_NEmo_preprocessor import Preprocessor96BUNEmo
from storage import storage_client

pp = Preprocessor96BUNEmo(
    local_path="datasets/96_Bu-NEMO", gcs_path="datasets/96_Bu-NEMO", storage_client=storage_client,dataset_id=96
)

pp.process()
