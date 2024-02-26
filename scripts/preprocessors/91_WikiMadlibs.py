"""Execute the 91 WikiMadlibs preprocessor."""

from preprocessing.WikiMadlibs_91_preprocessor import Preprocessor91WikiMadlibs
from storage import storage_client

pp = Preprocessor91WikiMadlibs(
    local_path="datasets/91_WikiMadlibs",
    gcs_path="datasets/91_WikiMadlibs",
    storage_client=storage_client,
    dataset_id=91,
)

pp.process()
