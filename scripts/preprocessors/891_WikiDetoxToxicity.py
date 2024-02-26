"""Execute the 891_WikiDetoxToxicity preprocessor."""

from preprocessing.WikiDetoxToxicity_891_preprocessor import (
    Preprocessor891WikiDetoxToxicity,
)
from storage import storage_client

pp = Preprocessor891WikiDetoxToxicity(
    local_path="datasets/891_WikiDetoxToxicity",
    gcs_path="datasets/891_WikiDetoxToxicity",
    storage_client=storage_client,
    dataset_id=891,
)

pp.process()
