"""Execute the 00_CC_NEWS_MLM preprocessor."""

from preprocessing.CC_News_MLM_00_preprocessor import PreprocessorCCNewsMLM
from storage import storage_client

pp = PreprocessorCCNewsMLM(
    local_path="datasets/00_MLM-CC-News", gcs_path="datasets/00_MLM-CC-News", storage_client=storage_client, dataset_id=0
)
pp.process()
