"""Execute the 31_SUBJ preprocessor."""

from preprocessing.SUBJ_31_preprocessor import Preprocessor31SUBJ
from storage import storage_client

pp = Preprocessor31SUBJ(local_path="datasets/31_SUBJ", gcs_path="datasets/31_SUBJ", storage_client=storage_client,dataset_id=31)

pp.process()
