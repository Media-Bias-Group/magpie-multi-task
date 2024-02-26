"""Execute the 118_WizardsOfWikipedia preprocessor."""

from preprocessing.WizardsOfWikipedia_118_preprocessor import (
    Preprocessor118WizardsOfWikipedia,
)
from storage import storage_client

pp = Preprocessor118WizardsOfWikipedia(
    local_path="datasets/118_WizardsOfWikipedia",
    gcs_path="datasets/118_WizardsOfWikipedia",
    storage_client=storage_client,
    dataset_id=118,
)

pp.process()
