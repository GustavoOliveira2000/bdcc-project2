import sys
import os
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

import requests
import dask.dataframe as dd
import numpy as np
import pandas as pd

from glob import glob

from pydantic import ByteSize
from dask.distributed import Client
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dask_ml.model_selection import KFold, train_test_split
from sklearn.metrics import make_scorer

from dask_ml.model_selection import train_test_split
from dask_ml.linear_model import LogisticRegression as DaskLogisticRegression
from xgboost.dask import DaskXGBRegressor

from datetime import timedelta

from collections.abc import Generator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from time import perf_counter
from typing import TypedDict

from pathlib import Path

print("All imports successful!")

# my_custom_dask_script.py


def resolve_scheduler_address() -> str:
    explicit_address = os.environ.get("DASK_SCHEDULER_ADDRESS")
    if explicit_address:
        return explicit_address

    explicit_host = os.environ.get("DASK_SCHEDULER_HOST")
    if explicit_host:
        return f"tcp://{explicit_host}:8786"

    metadata_url = (
        "http://metadata.google.internal/computeMetadata/v1/"
        "instance/attributes/dataproc-master"
    )
    metadata_request = Request(
        metadata_url,
        headers={"Metadata-Flavor": "Google"},
    )
    try:
        with urlopen(metadata_request, timeout=5) as response:
            master_host = response.read().decode("utf-8").strip()
    except URLError:
        master_host = ""

    if master_host:
        return f"tcp://{master_host}:8786"

    return "tcp://127.0.0.1:8786"

print("Connecting cleanly to Dataproc's Standalone Dask Scheduler...")

scheduler_address = resolve_scheduler_address()
print(f"Using scheduler address: {scheduler_address}")
client = Client(scheduler_address)

print("--------------------------------------------------")
print(f"Connected Successfully!")
print(f"Cluster Details: {client}")
print(f"Live Dashboard URL: {client.dashboard_link}")
print("--------------------------------------------------")

# --- YOUR NORMAL BENCHMARK PIPELINE RUNS HERE ---
# Data flows naturally from Cloud Storage to your cluster workers
# df = dd.read_parquet('gs://bdcc-assignment2-bucket/nyc_taxi/')
# print(df.groupby('passenger_count').trip_distance.mean().compute())

print("Pipeline completed successfully on native workers!")
