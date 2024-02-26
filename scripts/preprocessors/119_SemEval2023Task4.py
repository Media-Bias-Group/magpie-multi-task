"""Execute the 119_SemEval2023Task4 preprocessor."""

from preprocessing.SemEval2023Task4_119_preprocessor import (
    Preprocessor119SemEval2023Task4,
)
from storage import storage_client

pp = Preprocessor119SemEval2023Task4(
    local_path="datasets/119_SemEval2023Task4",
    gcs_path="datasets/119_SemEval2023Task4",
    storage_client=storage_client,
    dataset_id=119,
)
pp.process()
