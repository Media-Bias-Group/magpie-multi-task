"""Execute the 127_VaccineLies fetcher."""

from fetching.VaccineLies_127_fetcher import Fetcher127VaccineLies
from storage import storage_client

fetcher = Fetcher127VaccineLies(
    local_path="datasets/127_VaccineLies", gcs_path="datasets/127_VaccineLies", storage_client=storage_client
)

fetcher.process()
