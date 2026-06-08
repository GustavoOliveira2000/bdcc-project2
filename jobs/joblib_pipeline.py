# 1. Imports

import os
import math
import warnings
import requests
import pandas as pd
import numpy as np
import gcsfs

from joblib import Parallel, delayed
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor

from collections.abc import Generator
from contextlib import contextmanager
from datetime import timedelta, datetime
from pathlib import Path
from time import perf_counter
from typing import TypedDict


# 2. Configuration

BUCKET_NAME = "bdcc_code_v03"
DATA_PATH = "gs://{BUCKET_NAME}/data/taxi/{year}/yellow_tripdata_{year}-{month:02d}.parquet"

LOOKUP_PATH = "gs://{BUCKET_NAME}/data/taxi/lookup_tables/taxi_zone_lookup.csv".format(BUCKET_NAME=BUCKET_NAME)

SOURCE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

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

N_JOBS = -1

RESULTS_PATH = (
    "gs://"
    "{BUCKET_NAME}/results/"
    "{months}_{n_jobs}_{cpu_count}/"
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
    result: Duration = {"duration": timedelta()}
    start_time = perf_counter()

    try:
        yield result
    finally:
        duration = timedelta(seconds=perf_counter() - start_time)
        print(f"{name} took {duration}")
        result["duration"] = duration


def classify_fare(value):
    if value < 15:
        return 0
    elif value < 40:
        return 1
    else:
        return 2


def _gcs_read_parquet(paths: list[str]) -> pd.DataFrame:
    fs = gcsfs.GCSFileSystem(token="google_default")
    return pd.concat(
        [pd.read_parquet(fs.open(p)) for p in paths],
        ignore_index=True
    )


def _gcs_read_csv(path: str) -> pd.DataFrame:
    fs = gcsfs.GCSFileSystem(token="google_default")
    with fs.open(path) as f:
        return pd.read_csv(f)


def split_dataframe(df: pd.DataFrame, n_chunks: int) -> list[pd.DataFrame]:
    chunk_size = math.ceil(len(df) / n_chunks)
    return [
        df.iloc[i:i + chunk_size].copy()
        for i in range(0, len(df), chunk_size)
    ]


# 4. Load data

def load_data(months: int = 1):
    with timeit("Loading data"):
        df = _gcs_read_parquet(list(data_files(months=months)))
    location_lookup = _gcs_read_csv(LOOKUP_PATH)
    print(df.head())
    return df, location_lookup


# 5. Basic dataset inspection

def inspect_data(df):
    print("Columns:")
    print(df.columns)

    print("\nDtypes:")
    print(df.dtypes)

    print("\nShape:")
    print(df.shape)


# 6. Preprocessing

def preprocess(df):
    df = df[FEATURES + [TARGET]]
    df = df.dropna()

    df = df[
        (df["fare_amount"] > 0) &
        (df["trip_distance"] > 0) &
        (df["passenger_count"] > 0)
    ]

    print(df.tail())
    return df


# 7. Benchmarking

def benchmark_joblib(data_path: str | list[str], lookup_path: str, n_jobs: int = -1):
    metrics = {}

    # 1. Read Data
    with timeit("1. Read Data") as duration:
        df = _gcs_read_parquet(data_path if isinstance(data_path, list) else [data_path])
        df_lookup = _gcs_read_csv(lookup_path)
    metrics["1. Read Data"] = duration["duration"]

    if n_jobs == -1:
        effective_jobs = 4  # safe default for chunking; Joblib still uses all cores where applicable
    else:
        effective_jobs = max(1, n_jobs)

    chunks = split_dataframe(df, effective_jobs)

    # 2. Count Operation
    with timeit("2. Count Operation") as duration:
        partial_counts = Parallel(n_jobs=n_jobs)(
            delayed(lambda chunk: chunk.count())(chunk)
            for chunk in chunks
        )
        _ = pd.concat(partial_counts, axis=1).sum(axis=1)
    metrics["2. Count Operation"] = duration["duration"]

    # 3. Complex Arithmetic Formula
    with timeit("3. Complex Arithmetic") as duration:
        partial_results = Parallel(n_jobs=n_jobs)(
            delayed(lambda chunk: (chunk["fare_amount"] + chunk["tip_amount"])
                    * 1.5 / (chunk["passenger_count"] + 1))(chunk)
            for chunk in chunks
        )
        _ = pd.concat(partial_results)
    metrics["3. Complex Arithmetic"] = duration["duration"]

    # 4. Statistical Standard Deviation
    with timeit("4. Standard Deviation") as duration:
        # For simplicity and correctness, calculate std directly in pandas.
        # Joblib is less natural for global reductions than Dask/PySpark.
        _ = df["trip_distance"].std()
    metrics["4. Standard Deviation"] = duration["duration"]

    # 5. GroupBy Aggregation
    with timeit("5. GroupBy Mean") as duration:
        partial_grouped = Parallel(n_jobs=n_jobs)(
            delayed(lambda chunk: chunk.groupby("passenger_count")["fare_amount"].agg(["sum", "count"]))(chunk)
            for chunk in chunks
        )
        grouped = pd.concat(partial_grouped).groupby(level=0).sum()
        grouped["mean"] = grouped["sum"] / grouped["count"]
        _ = grouped["mean"]
    metrics["5. GroupBy Mean"] = duration["duration"]

    # 6. Merge/Join with Count
    with timeit("6. Join & Count") as duration:
        df_lookup["LocationID"] = df_lookup["LocationID"].astype(df["PULocationID"].dtype)
        joined = pd.merge(df, df_lookup, left_on="PULocationID", right_on="LocationID", how="inner")
        _ = len(joined)
    metrics["6. Join & Count"] = duration["duration"]

    return metrics


# Regression training

def train_regression(df):
    X = df[FEATURES]
    y_reg = df[TARGET]

    X_train, X_test, y_train_reg, y_test_reg = train_test_split(
        X,
        y_reg,
        test_size=TEST_SIZE,
        random_state=SEED
    )

    reg_model = XGBRegressor(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=N_JOBS
    )

    reg_cv = KFold(
        n_splits=REGRESSION_CV_FOLDS,
        shuffle=True,
        random_state=SEED
    )

    with timeit("Regression cross-validation") as duration:
        reg_cv_scores = cross_val_score(
            reg_model,
            X_train,
            y_train_reg,
            cv=reg_cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=N_JOBS
        )

    regression_cv_time = duration["duration"]
    regression_cv_rmse_scores = -reg_cv_scores
    regression_cv_rmse_mean = regression_cv_rmse_scores.mean()
    regression_cv_rmse_std = regression_cv_rmse_scores.std()

    print("Regression CV RMSE scores:", regression_cv_rmse_scores)
    print("Mean RMSE:", regression_cv_rmse_mean)
    print("Std RMSE:", regression_cv_rmse_std)

    with timeit("Regression training") as duration:
        reg_model.fit(X_train, y_train_reg)

    regression_train_time = duration["duration"]

    with timeit("Regression prediction") as duration:
        y_pred_reg = reg_model.predict(X_test)

    regression_prediction_time = duration["duration"]

    regression_results = {
        "library": "Joblib",
        "task": "regression",
        "model": "XGBRegressor",
        "cv_folds": REGRESSION_CV_FOLDS,
        "cv_time": regression_cv_time,
        "cv_rmse_mean": regression_cv_rmse_mean,
        "cv_rmse_std": regression_cv_rmse_std,
        "train_time": regression_train_time,
        "prediction_time": regression_prediction_time,
        "mae": mean_absolute_error(y_test_reg, y_pred_reg),
        "mse": mean_squared_error(y_test_reg, y_pred_reg),
        "rmse": mean_squared_error(y_test_reg, y_pred_reg) ** 0.5,
        "r2": r2_score(y_test_reg, y_pred_reg),
    }

    return regression_results


# Classification training

def train_classification(df):
    warnings.filterwarnings("ignore")

    X = df[FEATURES]
    y_cls = df[TARGET].apply(classify_fare)

    X_train, X_test, y_train_cls, y_test_cls = train_test_split(
        X,
        y_cls,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y_cls
    )

    cls_model = LogisticRegression(
        max_iter=LOGISTIC_MAX_ITER,
        penalty=None,
        fit_intercept=True,
        random_state=SEED,
        n_jobs=N_JOBS
    )

    cls_cv = StratifiedKFold(
        n_splits=CLASSIFICATION_CV_FOLDS,
        shuffle=True,
        random_state=SEED
    )

    with timeit("Classification cross-validation") as duration:
        cls_cv_scores = cross_val_score(
            cls_model,
            X_train,
            y_train_cls,
            cv=cls_cv,
            scoring="f1_macro",
            n_jobs=N_JOBS
        )

    classification_cv_time = duration["duration"]
    classification_cv_f1_scores = cls_cv_scores
    classification_cv_f1_mean = classification_cv_f1_scores.mean()
    classification_cv_f1_std = classification_cv_f1_scores.std()

    print("Classification CV F1 scores:", classification_cv_f1_scores)
    print("Mean F1:", classification_cv_f1_mean)
    print("Std F1:", classification_cv_f1_std)

    with timeit("Classification training") as duration:
        cls_model.fit(X_train, y_train_cls)

    classification_train_time = duration["duration"]

    with timeit("Classification prediction") as duration:
        y_pred_cls = cls_model.predict(X_test)

    classification_prediction_time = duration["duration"]

    classification_results = {
        "library": "Joblib",
        "task": "classification",
        "model": "LogisticRegression",
        "cv_folds": CLASSIFICATION_CV_FOLDS,
        "cv_time": classification_cv_time,
        "cv_f1_macro_mean": classification_cv_f1_mean,
        "cv_f1_macro_std": classification_cv_f1_std,
        "train_time": classification_train_time,
        "prediction_time": classification_prediction_time,
        "accuracy": accuracy_score(y_test_cls, y_pred_cls),
        "precision_macro": precision_score(y_test_cls, y_pred_cls, average="macro"),
        "recall_macro": recall_score(y_test_cls, y_pred_cls, average="macro"),
        "f1_macro": f1_score(y_test_cls, y_pred_cls, average="macro"),
    }

    return classification_results


# Save Results

def save_results(benchmarks, regression_results, classification_results, env):
    benchmark_results_df = pd.DataFrame([
        {
            "library": "Joblib",
            "operation": operation,
            "duration": duration,
            "duration_seconds": duration.total_seconds()
        }
        for operation, duration in benchmarks.items()
    ])

    benchmark_path = RESULTS_PATH.format(
        BUCKET_NAME=BUCKET_NAME,
        months=env["months"],
        n_jobs=env["n_jobs"],
        cpu_count=env["cpu_count"],
        timestamp=env["timestamp"],
        filename="joblib_benchmark_results.csv"
    )
    benchmark_results_df.to_csv(benchmark_path, index=False)

    results_df = pd.DataFrame([
        regression_results,
        classification_results
    ])

    pipeline_path = RESULTS_PATH.format(
        BUCKET_NAME=BUCKET_NAME,
        months=env["months"],
        n_jobs=env["n_jobs"],
        cpu_count=env["cpu_count"],
        timestamp=env["timestamp"],
        filename="joblib_pipeline_results.csv"
    )
    results_df.to_csv(pipeline_path, index=False)
    print(results_df)


def main():
    MONTHS_TO_LOAD = 18

    env = {
        "months": MONTHS_TO_LOAD,
        "n_jobs": N_JOBS,
        "cpu_count": os.cpu_count(),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }

    df, location_lookup = load_data(MONTHS_TO_LOAD)

    inspect_data(df)

    df = preprocess(df)

    benchmarks = benchmark_joblib(
        list(data_files(months=MONTHS_TO_LOAD)),
        LOOKUP_PATH,
        n_jobs=N_JOBS
    )

    regression_results = train_regression(df)

    classification_results = train_classification(df)

    save_results(benchmarks, regression_results, classification_results, env)


if __name__ == "__main__":
    main()
