"""Execute the 19_MultiDimNews preprocessor."""

from preprocessing.MultiDimNews_19_preprocessor import Preprocessor19MultiDimNews
from storage import storage_client

pp = Preprocessor19MultiDimNews(
    local_path="datasets/19_MultiDimNews",
    gcs_path="datasets/19_MultiDimNews",
    storage_client=storage_client,
    dataset_id=19,
)

pp.process()
