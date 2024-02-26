"""Execute the 103_MPQA preprocessor."""

from preprocessing.MPQA_103_preprocessor import Preprocessor103MPQA
from storage import storage_client

pp = Preprocessor103MPQA(local_path="datasets/103_MPQA", gcs_path="datasets/103_MPQA", storage_client=storage_client,dataset_id=103)

pp.process()
