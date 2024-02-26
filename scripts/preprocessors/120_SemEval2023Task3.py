"""Execute the 120_SemEval2023Task3 preprocessor."""

from preprocessing.SemEval2023Task3_120_preprocessor import (
    Preprocessor120SemEval2023Task3,
)
from storage import storage_client

pp = Preprocessor120SemEval2023Task3(
    local_path="datasets/120_SemEval2023Task3",
    gcs_path="datasets/120_SemEval2023Task3",
    storage_client=storage_client,
    dataset_id=120,
)

pp.process()
