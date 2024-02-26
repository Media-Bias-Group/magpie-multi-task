"""Execute the 100_AmazonProduct preprocessor."""

from preprocessing.AmazonProduct_100_preprocessor import Preprocessor100AmazonProduct
from storage import storage_client

pp = Preprocessor100AmazonProduct(
    local_path="datasets/100_Amazon_reviews", gcs_path="datasets/100_Amazon_reviews", storage_client=storage_client, dataset_id=100
)

pp.process()
