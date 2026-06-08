"""
A script that downloads all taxi data to a google bucket.
"""
from collections.abc import Generator
from pathlib import Path

import requests
from google.cloud import storage

SOURCE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/"


def all_data_sources(year: int = 2024) -> Generator[str]:
    for month in range(1, 13):
        yield f"{SOURCE_URL}yellow_tripdata_{year}-{month:02d}.parquet"


def downloader():
    bucket_name = "bdcc_code_v03"
    for year in range(2009, 2024):
        for data_source in all_data_sources(year):
            response = requests.get(data_source)
            response.raise_for_status()
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(f"data/taxi/{year}/{Path(data_source).name}")
            blob.upload_from_string(response.content)
            print(
                f"Uploaded {data_source} to "
                f"gs://{bucket_name}/data/taxi/{year}/{Path(data_source).name}"
            )


if __name__ == "__main__":
    downloader()
