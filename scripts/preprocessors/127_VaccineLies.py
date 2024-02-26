"""Execute the 127_VaccineLies preprocessor."""

from preprocessing.VaccineLies_127_preprocessor import Preprocessor127VaccineLies
from storage import storage_client

pp = Preprocessor127VaccineLies(
    local_path="datasets/127_VaccineLies", gcs_path="datasets/127_VaccineLies", storage_client=storage_client,dataset_id=127
)

pp.process()
