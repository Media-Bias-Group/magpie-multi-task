"""Execute the 892_WikiDetoxAggressionAndAttack preprocessor."""

from preprocessing.WikiDetoxAggressionAndAttack_892_preprocessor import (
    Preprocessor892WikiDetoxAggressionAndAttack,
)
from storage import storage_client

pp = Preprocessor892WikiDetoxAggressionAndAttack(
    local_path="datasets/892_WikiDetoxAggressionAndAttack",
    gcs_path="datasets/892_WikiDetoxAggressionAndAttack",
    storage_client=storage_client,
    dataset_id=892,
)

pp.process()
