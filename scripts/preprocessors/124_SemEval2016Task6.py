"""Execute the 124_SemEval2016Task6 preprocessor."""

from preprocessing.SemEval2016Task6_124_preprocessor import (
    Preprocessor124SemEval2016Task6,
)
from storage import storage_client

pp = Preprocessor124SemEval2016Task6(
    local_path="datasets/124_SemEval2016Task6",
    gcs_path="datasets/124_SemEval2016Task6",
    storage_client=storage_client,
    dataset_id=124,
)
pp.process()
