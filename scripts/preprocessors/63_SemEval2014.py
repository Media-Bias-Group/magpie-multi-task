"""Execute the 63_SemEval preprocessor."""

from preprocessing.SemEval2014_63_processor import Preprocessor63SemEval2014
from storage import storage_client

pp = Preprocessor63SemEval2014(
    local_path="datasets/63_semeval2014",
    gcs_path="datasets/63_semeval2014",
    storage_client=storage_client,
    dataset_id=63,
)

pp.process()
