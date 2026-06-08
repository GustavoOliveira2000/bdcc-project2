# 1. Imports

import warnings
from collections.abc import Generator
from contextlib import contextmanager
from datetime import timedelta, datetime
from glob import glob
from pathlib import Path
from time import perf_counter
from typing import TypedDict

import dask.dataframe as dd
import numpy as np
import pandas as pd
import requests
from dask.distributed import Client
from dask_ml.linear_model import LogisticRegression as DaskLogisticRegression
from dask_ml.model_selection import KFold, train_test_split
from pydantic import ByteSize
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from xgboost.dask import DaskXGBRegressor

# 3. Configuration

CLUSTER_NAME = "final-v02"
BUCKET_NAME = "bdcc_code_v03"
DATA_PATH = "gs://{BUCKET_NAME}/data/taxi/{year}/yellow_tripdata_{year}-{month:02d}.parquet"

LOOKUP_PATH = "gs://{BUCKET_NAME}/data/taxi/lookup_tables/taxi_zone_lookup.csv".format(BUCKET_NAME=BUCKET_NAME)

SOURCE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"

TARGET = "fare_amount"

FEATURES = [
    "trip_distance",
    "passenger_count",
    "PULocationID",
    "DOLocationID",
    "payment_type",
]

SEED = 42
TEST_SIZE = 0.2

CLASSIFICATION_CV_FOLDS = 5
REGRESSION_CV_FOLDS = 5

LOGISTIC_MAX_ITER = 100
LOGISTIC_REGULARIZATION = 0.0

XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1

WORKERS = 1
THREADS_PER_WORKER = 32
MEMORY_LIMIT = "64GB"

RESULTS_PATH = (
    "gs://"
    "{BUCKET_NAME}/results/"
    "{months}_{processes}_{threads}_{memory}/"
    "{timestamp}_{filename}"
)

# Helper functions


def data_files(months: int) -> Generator[str, None, None]:
    if months <= 12:
        year = 2024
        for month in range(1, months + 1):
            yield DATA_PATH.format(BUCKET_NAME=BUCKET_NAME, year=year, month=month)
    elif months <= 24:
        i = 0
        for year in range(2024, 2026):
            for month in range(1, 13):
                i += 1
                if i > months:
                    return
                yield DATA_PATH.format(BUCKET_NAME=BUCKET_NAME, year=year, month=month)
    else:
        for year in range(2016, 2026):
            for month in range(1, 13):
                yield DATA_PATH.format(BUCKET_NAME=BUCKET_NAME, year=year, month=month)


class Duration(TypedDict):
    duration: timedelta


@contextmanager
def timeit(name: str) -> Generator[dict]:
    result = {"duration": timedelta()}
    start_time = perf_counter()
    yield result
    end_time = perf_counter()
    duration = timedelta(seconds=end_time - start_time)
    print(f"{name} took {duration}")
    result["duration"] = duration


def classify_fare_partition(partition):
    return partition.assign(
        fare_class=np.select(
            [
                partition[TARGET] < 15,
                partition[TARGET] < 40,
            ],
            [0, 1],
            default=2
        )
    )


# 2. Start Dask client

def start_client() -> Client:
    host = f"tcp://{CLUSTER_NAME}-m:8786"
    try:
        client = Client(host)
    except Exception as e:
        print(f"Failed to connect to Dask scheduler at {host}: {e}")
        client = Client(
            n_workers=1,
            threads_per_worker=32,
            memory_limit="64GB"
        )
    print(client)
    return client


# 4. Load data

def load_data(months: int = 1):
    df = dd.read_parquet(list(data_files(months=months)))
    location_lookup = dd.read_csv(LOOKUP_PATH)
    print(df.tail())
    return df, location_lookup


# 5. Basic dataset inspection

def inspect_data(df):
    print("Columns:")
    print(df.columns)

    print("Dtypes:")
    print(df.dtypes)

    print("Number of partitions:")
    print(df.npartitions)


# 6. Preprocessing

def preprocess(df):
    df = df[FEATURES + [TARGET]]
    df = df.dropna()
    df = df[
        (df["fare_amount"] > 0) &
        (df["trip_distance"] > 0) &
        (df["passenger_count"] > 0)
    ]

    df = df.repartition(npartitions=3)
    df = df.reset_index(drop=True)
    df = df.clear_divisions()

    print(df.tail())
    return df


# 7. Benchmarking

def benchmark_dask(data_path, lookup_path):
    metrics = {}

    # 1. Read Data
    with timeit("1. Read Data") as duration:
        df = dd.read_parquet(data_path)
        df_lookup = dd.read_csv(lookup_path)  # estamos a fazer leitura desse ficheiro para a operação de join/merge
        df_lookup["LocationID"] = df_lookup["LocationID"].astype(df["PULocationID"].dtype)
        # Dask reads lazily; force compute on length to evaluate reading execution
        _ = len(df)
    metrics['1. Read Data'] = duration["duration"]

    # 2. Count Index (Simple count)
    with timeit("2. Count Operation") as duration:
        _ = df.count().compute()
    metrics['2. Count Operation'] = duration["duration"]

    # 3. Complex Arithmetic Formula
    with timeit("3. Complex Arithmetic") as duration:
        arithmetic_res = (df['fare_amount'] + df['tip_amount']) * 1.5 / (df['passenger_count'] + 1)
        _ = arithmetic_res.compute()
    metrics['3. Complex Arithmetic'] = duration["duration"]

    # 4. Statistical Standard Deviation
    with timeit("4. Standard Deviation") as duration:
        _ = df['trip_distance'].std().compute()
    metrics['4. Standard Deviation'] = duration["duration"]

    # 5. GroupBy Aggregation
    with timeit("5. GroupBy Mean") as duration:
        _ = df.groupby('passenger_count')['fare_amount'].mean().compute()
    metrics['5. GroupBy Mean'] = duration["duration"]

    # 6. Merge/Join with Count
    with timeit("6. Join & Count") as duration:
        joined = dd.merge(df, df_lookup, left_on='PULocationID', right_on='LocationID', how='inner')
        _ = len(joined)
    metrics['6. Join & Count'] = duration["duration"]

    return metrics


# Regression training

def train_regression(df):
    X = df[FEATURES]
    y_reg = df[TARGET]

    X_train, X_test, y_train_reg, y_test_reg = train_test_split(
        X,
        y_reg,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True
    )

    # Specific for one year data:
    X_train = X_train.repartition(npartitions=12)
    X_test = X_test.repartition(npartitions=12)

    y_train_reg = y_train_reg.repartition(npartitions=12)
    y_test_reg = y_test_reg.repartition(npartitions=12)

    X_train_arr = X_train.to_dask_array(lengths=True)
    X_test_arr = X_test.to_dask_array(lengths=True)

    y_train_reg_arr = y_train_reg.to_dask_array(lengths=True)
    y_test_reg_arr = y_test_reg.to_dask_array(lengths=True)

    reg_model = DaskXGBRegressor(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        objective="reg:squarederror",
        random_state=SEED
    )

    reg_cv = KFold(
        n_splits=REGRESSION_CV_FOLDS,
        shuffle=True,
        random_state=SEED
    )

    reg_cv_rmse_scores = []

    with timeit("Regression cross-validation") as duration:
        for train_idx, val_idx in reg_cv.split(X_train_arr):
            X_fold_train = X_train_arr[train_idx]
            X_fold_val = X_train_arr[val_idx]

            y_fold_train = y_train_reg_arr[train_idx]
            y_fold_val = y_train_reg_arr[val_idx]

            fold_model = DaskXGBRegressor(
                n_estimators=XGB_N_ESTIMATORS,
                max_depth=XGB_MAX_DEPTH,
                learning_rate=XGB_LEARNING_RATE,
                objective="reg:squarederror",
                random_state=SEED
            )

            fold_model.fit(X_fold_train, y_fold_train)

            y_fold_pred = fold_model.predict(X_fold_val).compute()
            y_fold_val_local = y_fold_val.compute()

            fold_rmse = mean_squared_error(
                y_fold_val_local,
                y_fold_pred
            ) ** 0.5

            reg_cv_rmse_scores.append(fold_rmse)

    regression_cv_time = duration["duration"]
    regression_cv_rmse_mean = np.mean(reg_cv_rmse_scores)
    regression_cv_rmse_std = np.std(reg_cv_rmse_scores)

    print("Regression CV RMSE scores:", reg_cv_rmse_scores)
    print("Mean RMSE:", regression_cv_rmse_mean)
    print("Std RMSE:", regression_cv_rmse_std)

    with timeit("Regression training") as duration:
        fitted_reg_model = reg_model.fit(X_train_arr, y_train_reg_arr)

    regression_train_time = duration["duration"]

    with timeit("Regression prediction") as duration:
        y_pred_reg = fitted_reg_model.predict(X_test_arr).compute()

    regression_prediction_time = duration["duration"]

    y_test_reg_local = y_test_reg_arr.compute()

    regression_results = {
        "library": "Dask",
        "task": "regression",
        "model": "DaskXGBRegressor",
        "cv_folds": REGRESSION_CV_FOLDS,
        "cv_time": regression_cv_time,
        "cv_rmse_mean": regression_cv_rmse_mean,
        "cv_rmse_std": regression_cv_rmse_std,
        "train_time": regression_train_time,
        "prediction_time": regression_prediction_time,
        "mae": mean_absolute_error(y_test_reg_local, y_pred_reg),
        "mse": mean_squared_error(y_test_reg_local, y_pred_reg),
        "rmse": mean_squared_error(y_test_reg_local, y_pred_reg) ** 0.5,
        "r2": r2_score(y_test_reg_local, y_pred_reg),
    }

    return regression_results


# Classification training

def train_classification(df):
    warnings.filterwarnings("ignore")

    df_cls = df.map_partitions(classify_fare_partition)

    X_cls = df_cls[FEATURES]
    y_cls = df_cls["fare_class"]

    X_train, X_test, y_train_cls, y_test_cls = train_test_split(
        X_cls,
        y_cls,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True
    )

    X_train_arr = X_train.to_dask_array(lengths=True)
    X_test_arr = X_test.to_dask_array(lengths=True)

    y_train_arr = y_train_cls.to_dask_array(lengths=True)
    y_test_arr = y_test_cls.to_dask_array(lengths=True)

    cls_model = DaskLogisticRegression(
        max_iter=LOGISTIC_MAX_ITER,
        fit_intercept=True
    )

    cls_cv = KFold(
        n_splits=CLASSIFICATION_CV_FOLDS,
        shuffle=True,
        random_state=SEED
    )

    cls_cv_f1_scores = []

    with timeit("Classification cross-validation") as duration:
        for train_idx, val_idx in cls_cv.split(X_train_arr):
            X_fold_train = X_train_arr[train_idx]
            X_fold_val = X_train_arr[val_idx]
            y_fold_train = y_train_arr[train_idx]
            y_fold_val = y_train_arr[val_idx]

            fold_model = DaskLogisticRegression(
                max_iter=LOGISTIC_MAX_ITER,
                fit_intercept=True
            )

            fold_model.fit(X_fold_train, y_fold_train)
            y_fold_pred = fold_model.predict(X_fold_val).compute()
            y_fold_val_local = y_fold_val.compute()

            fold_f1 = f1_score(
                y_fold_val_local,
                y_fold_pred,
                average="macro"
            )

            cls_cv_f1_scores.append(fold_f1)

    classification_cv_time = duration["duration"]
    classification_cv_f1_mean = np.mean(cls_cv_f1_scores)
    classification_cv_f1_std = np.std(cls_cv_f1_scores)

    print("Classification CV F1 scores:", cls_cv_f1_scores)
    print("Mean F1:", classification_cv_f1_mean)
    print("Std F1:", classification_cv_f1_std)

    with timeit("Classification training") as duration:
        fitted_cls_model = cls_model.fit(X_train_arr, y_train_arr)

    classification_train_time = duration["duration"]

    with timeit("Classification prediction") as duration:
        y_pred_cls = fitted_cls_model.predict(X_test_arr).compute()

    classification_prediction_time = duration["duration"]

    y_test_cls_local = y_test_arr.compute()

    classification_results = {
        "library": "Dask",
        "task": "classification",
        "model": "DaskLogisticRegression",
        "cv_folds": CLASSIFICATION_CV_FOLDS,
        "cv_time": classification_cv_time,
        "cv_f1_macro_mean": classification_cv_f1_mean,
        "cv_f1_macro_std": classification_cv_f1_std,
        "train_time": classification_train_time,
        "prediction_time": classification_prediction_time,
        "accuracy": accuracy_score(y_test_cls_local, y_pred_cls),
        "precision_macro": precision_score(y_test_cls_local, y_pred_cls, average="macro"),
        "recall_macro": recall_score(y_test_cls_local, y_pred_cls, average="macro"),
        "f1_macro": f1_score(y_test_cls_local, y_pred_cls, average="macro"),
    }

    return classification_results


# Save Results

def save_results(benchmarks, regression_results, classification_results, env):
    benchmark_results_df = pd.DataFrame([
        {
            "library": "Dask",
            "operation": operation,
            "duration": duration,
            "duration_seconds": duration.total_seconds()
        }
        for operation, duration in benchmarks.items()
    ])

    results_path = RESULTS_PATH.format(
        BUCKET_NAME=BUCKET_NAME,
        months=env["month"],
        processes=env["processes"],
        threads=env["threads"],
        memory=env["memory"],
        timestamp=env["timestamp"],
        filename="dask_benchmark_results.csv"
    )
    benchmark_results_df.to_csv(results_path, index=False)

    results_df = pd.DataFrame([
        regression_results,
        classification_results
    ])

    results_path = RESULTS_PATH.format(
        BUCKET_NAME=BUCKET_NAME,
        months=env["month"],
        processes=env["processes"],
        threads=env["threads"],
        memory=env["memory"],
        timestamp=env["timestamp"],
        filename="dask_pipeline_results.csv"
    )
    results_df.to_csv(results_path, index=False)
    print(results_df)


def main():
    MONTHS_TO_LOAD = 2
    client = start_client()

    info = client.scheduler_info()
    n_processes = len(info["workers"])
    n_threads = sum(w["nthreads"] for w in info["workers"].values())
    total_memory = sum(w["memory_limit"] for w in info["workers"].values())

    env = {
        "month": MONTHS_TO_LOAD,
        "processes": n_processes,
        "threads": n_threads,
        "memory": ByteSize(total_memory).human_readable(),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }

    df, location_lookup = load_data(MONTHS_TO_LOAD)

    inspect_data(df)

    df = preprocess(df)

    benchmarks = benchmark_dask(
        list(data_files(months=MONTHS_TO_LOAD)),
        LOOKUP_PATH
    )

    regression_results = train_regression(df)

    classification_results = train_classification(df)

    save_results(benchmarks, regression_results, classification_results, env)

    client.close()


if __name__ == "__main__":
    main()
