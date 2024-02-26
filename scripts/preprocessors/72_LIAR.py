"""Execute the 72_LIAR preprocessor."""

from preprocessing.LIAR_72_preprocessor import Preprocessor72LIAR
from storage import storage_client

pp = Preprocessor72LIAR(
    local_path="datasets/72_LIAR", gcs_path="datasets/72_LIAR", storage_client=storage_client, dataset_id=72
)

pp.process()
