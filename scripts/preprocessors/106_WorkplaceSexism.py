"""Execute the 106_WorkplaceSexism preprocessor."""

from preprocessing.WorkplaceSexism_106_preprocessor import (
    Preprocessor106WorkplaceSexism,
)
from storage import storage_client

pp = Preprocessor106WorkplaceSexism(
    local_path="datasets/106_WorkPlaceSexism", gcs_path="datasets/106_WorkPlaceSexism", storage_client=storage_client,dataset_id=106
)

pp.process()
