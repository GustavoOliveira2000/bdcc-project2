# 1. Imports

import requests
import pandas as pd

from pydantic import ByteSize
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from collections.abc import Generator
from contextlib import contextmanager
from datetime import timedelta, datetime
from time import perf_counter
from typing import TypedDict


# 2. Configuration

CLUSTER_NAME = "final-v02"
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

FEATURES_COL = "features"
LABEL_COL = "label"

SEED = 42
TEST_SIZE = 0.2
TRAIN_RATIO = 0.8

CLASSIFICATION_CV_FOLDS = 5
REGRESSION_CV_FOLDS = 5

LOGISTIC_MAX_ITER = 100
LOGISTIC_REGULARIZATION = 0.0

XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1

RESULTS_PATH = (
    "gs://"
    "{BUCKET_NAME}/results/"
    "{months}_{executors}_{cores}_{memory}/"
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


# 2. Start Spark session

def start_spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("BDCC PySpark Taxi Pipeline")
        .config("spark.driver.extraJavaOptions", "--add-opens=java.base/javax.security.auth=ALL-UNNAMED")
        .config("spark.executor.extraJavaOptions", "--add-opens=java.base/javax.security.auth=ALL-UNNAMED")
        .getOrCreate()
    )
    print(spark)
    return spark


# 4. Load data

def load_data(spark: SparkSession, months: int = 1):
    with timeit("Loading data"):
        df = spark.read.parquet(*list(data_files(months=months)))

    location_lookup = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(LOOKUP_PATH)
    )

    df.show(5)
    return df, location_lookup


# 5. Basic dataset inspection

def inspect_data(df):
    print("Columns:")
    print(df.columns)

    print("\nSchema:")
    df.printSchema()

    print("\nNumber of partitions:")
    print(df.rdd.getNumPartitions())


# 6. Preprocessing

def preprocess(df):
    df = df.select(*(FEATURES + [TARGET]))
    df = df.dropna()
    df = df.filter(
        (F.col("fare_amount") > 0) &
        (F.col("trip_distance") > 0) &
        (F.col("passenger_count") > 0)
    )
    df.show(5)
    return df


# 7. Benchmarking

def benchmark_pyspark(spark: SparkSession, data_paths: list[str], lookup_path: str):
    metrics = {}

    # 1. Read Data
    with timeit("1. Read Data") as duration:
        df = spark.read.parquet(*data_paths)
        df_lookup = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(lookup_path)
        )
        _ = df.count()
    metrics["1. Read Data"] = duration["duration"]

    # 2. Count Operation
    with timeit("2. Count Operation") as duration:
        _ = df.count()
    metrics["2. Count Operation"] = duration["duration"]

    # 3. Complex Arithmetic Formula
    with timeit("3. Complex Arithmetic") as duration:
        arithmetic_df = df.select(
            (((F.col("fare_amount") + F.col("tip_amount")) * 1.5) / (F.col("passenger_count") + 1)).alias("result")
        )
        _ = arithmetic_df.count()
    metrics["3. Complex Arithmetic"] = duration["duration"]

    # 4. Statistical Standard Deviation
    with timeit("4. Standard Deviation") as duration:
        _ = df.select(F.stddev("trip_distance")).collect()
    metrics["4. Standard Deviation"] = duration["duration"]

    # 5. GroupBy Aggregation
    with timeit("5. GroupBy Mean") as duration:
        _ = (
            df.groupBy("passenger_count")
            .agg(F.mean("fare_amount").alias("mean_fare_amount"))
            .collect()
        )
    metrics["5. GroupBy Mean"] = duration["duration"]

    # 6. Merge/Join with Count
    with timeit("6. Join & Count") as duration:
        df_lookup = df_lookup.withColumn("LocationID", F.col("LocationID").cast(df.schema["PULocationID"].dataType))
        joined = df.join(df_lookup, df["PULocationID"] == df_lookup["LocationID"], how="inner")
        _ = joined.count()
    metrics["6. Join & Count"] = duration["duration"]

    return metrics


# Regression training

def train_regression(spark: SparkSession, df):
    assembler = VectorAssembler(
        inputCols=FEATURES,
        outputCol=FEATURES_COL,
        handleInvalid="skip"
    )

    ml_df = assembler.transform(df).select(
        F.col(FEATURES_COL),
        F.col(TARGET).alias(LABEL_COL)
    )

    train_df, test_df = ml_df.randomSplit([TRAIN_RATIO, TEST_SIZE], seed=SEED)

    print("Train rows:", train_df.count())
    print("Test rows:", test_df.count())

    spark.catalog.clearCache()

    reg_model = GBTRegressor(
        featuresCol=FEATURES_COL,
        labelCol=LABEL_COL,
        maxIter=XGB_N_ESTIMATORS,
        maxDepth=XGB_MAX_DEPTH,
        stepSize=XGB_LEARNING_RATE,
        seed=SEED
    )

    reg_evaluator = RegressionEvaluator(
        labelCol=LABEL_COL,
        predictionCol="prediction",
        metricName="rmse"
    )

    reg_param_grid = (
        ParamGridBuilder()
        .addGrid(reg_model.maxIter, [XGB_N_ESTIMATORS])
        .addGrid(reg_model.maxDepth, [XGB_MAX_DEPTH])
        .addGrid(reg_model.stepSize, [XGB_LEARNING_RATE])
        .build()
    )

    reg_cv = CrossValidator(
        estimator=reg_model,
        estimatorParamMaps=reg_param_grid,
        evaluator=reg_evaluator,
        numFolds=REGRESSION_CV_FOLDS,
        seed=SEED,
        parallelism=1
    )

    spark.catalog.clearCache()

    with timeit("Regression cross-validation") as duration:
        reg_cv_model = reg_cv.fit(train_df)

    regression_cv_time = duration["duration"]
    regression_cv_rmse_mean = min(reg_cv_model.avgMetrics)
    regression_cv_rmse_std = None

    print("Regression CV RMSE:", regression_cv_rmse_mean)

    spark.catalog.clearCache()

    with timeit("Regression training") as duration:
        fitted_reg_model = reg_model.fit(train_df)

    regression_train_time = duration["duration"]

    with timeit("Regression prediction") as duration:
        reg_predictions = fitted_reg_model.transform(test_df)
        reg_predictions.count()  # force execution

    regression_prediction_time = duration["duration"]

    reg_predictions.select("prediction", LABEL_COL).show(5)

    regression_results = {
        "library": "PySpark",
        "task": "regression",
        "model": "GBTRegressor",
        "cv_folds": REGRESSION_CV_FOLDS,
        "cv_time": regression_cv_time,
        "cv_rmse_mean": regression_cv_rmse_mean,
        "cv_rmse_std": regression_cv_rmse_std,
        "train_time": regression_train_time,
        "prediction_time": regression_prediction_time,
        "mae": RegressionEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName="mae").evaluate(reg_predictions),
        "mse": RegressionEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName="mse").evaluate(reg_predictions),
        "rmse": RegressionEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName="rmse").evaluate(reg_predictions),
        "r2": RegressionEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName="r2").evaluate(reg_predictions),
    }

    return regression_results


# Classification training

def train_classification(spark: SparkSession, df):
    assembler = VectorAssembler(
        inputCols=FEATURES,
        outputCol=FEATURES_COL,
        handleInvalid="skip"
    )

    classified_df = df.withColumn(
        "fare_class",
        F.when(F.col(TARGET) < 15, F.lit(0))
         .when(F.col(TARGET) < 40, F.lit(1))
         .otherwise(F.lit(2))
         .cast(IntegerType())
    )

    classification_ml_df = assembler.transform(classified_df).select(
        F.col(FEATURES_COL),
        F.col("fare_class").alias(LABEL_COL)
    )

    train_cls_df = (
        classification_ml_df
        .sampleBy(
            LABEL_COL,
            fractions={0: TRAIN_RATIO, 1: TRAIN_RATIO, 2: TRAIN_RATIO},
            seed=SEED
        )
    )

    test_cls_df = classification_ml_df.subtract(train_cls_df)

    cls_model = LogisticRegression(
        featuresCol=FEATURES_COL,
        labelCol=LABEL_COL,
        maxIter=LOGISTIC_MAX_ITER,
        regParam=LOGISTIC_REGULARIZATION,
        elasticNetParam=0.0,
        fitIntercept=True,
        family="multinomial"
    )

    cls_evaluator = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL,
        predictionCol="prediction",
        metricName="f1"
    )

    cls_param_grid = (
        ParamGridBuilder()
        .addGrid(cls_model.maxIter, [LOGISTIC_MAX_ITER])
        .addGrid(cls_model.regParam, [LOGISTIC_REGULARIZATION])
        .build()
    )

    cls_cv = CrossValidator(
        estimator=cls_model,
        estimatorParamMaps=cls_param_grid,
        evaluator=cls_evaluator,
        numFolds=CLASSIFICATION_CV_FOLDS,
        seed=SEED,
        parallelism=1
    )

    with timeit("Classification cross-validation") as duration:
        cls_cv_model = cls_cv.fit(train_cls_df)

    classification_cv_time = duration["duration"]
    classification_cv_f1_mean = max(cls_cv_model.avgMetrics)
    classification_cv_f1_std = None

    print("Classification CV F1:", classification_cv_f1_mean)

    with timeit("Classification training") as duration:
        fitted_cls_model = cls_model.fit(train_cls_df)

    classification_train_time = duration["duration"]

    with timeit("Classification prediction") as duration:
        cls_predictions = fitted_cls_model.transform(test_cls_df)
        cls_predictions.count()  # force execution

    classification_prediction_time = duration["duration"]

    cls_predictions.select("prediction", LABEL_COL).show(5)

    classification_results = {
        "library": "PySpark",
        "task": "classification",
        "model": "LogisticRegression",
        "cv_folds": CLASSIFICATION_CV_FOLDS,
        "cv_time": classification_cv_time,
        "cv_f1_macro_mean": classification_cv_f1_mean,
        "cv_f1_macro_std": classification_cv_f1_std,
        "train_time": classification_train_time,
        "prediction_time": classification_prediction_time,
        "accuracy": MulticlassClassificationEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName="accuracy").evaluate(cls_predictions),
        "precision_weighted": MulticlassClassificationEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName="weightedPrecision").evaluate(cls_predictions),
        "recall_weighted": MulticlassClassificationEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName="weightedRecall").evaluate(cls_predictions),
        "f1_weighted": MulticlassClassificationEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName="f1").evaluate(cls_predictions),
    }

    return classification_results


# Save Results

def save_results(benchmarks, regression_results, classification_results, env):
    benchmark_results_df = pd.DataFrame([
        {
            "library": "PySpark",
            "operation": operation,
            "duration": duration,
            "duration_seconds": duration.total_seconds()
        }
        for operation, duration in benchmarks.items()
    ])

    benchmark_path = RESULTS_PATH.format(
        BUCKET_NAME=BUCKET_NAME,
        months=env["months"],
        executors=env["executors"],
        cores=env["cores"],
        memory=env["memory"],
        timestamp=env["timestamp"],
        filename="pyspark_benchmark_results.csv"
    )
    benchmark_results_df.to_csv(benchmark_path, index=False, storage_options={"token": "google_default"})

    results_df = pd.DataFrame([
        regression_results,
        classification_results
    ])

    pipeline_path = RESULTS_PATH.format(
        BUCKET_NAME=BUCKET_NAME,
        months=env["months"],
        executors=env["executors"],
        cores=env["cores"],
        memory=env["memory"],
        timestamp=env["timestamp"],
        filename="pyspark_pipeline_results.csv"
    )
    results_df.to_csv(pipeline_path, index=False, storage_options={"token": "google_default"})
    print(results_df)


def main():
    MONTHS_TO_LOAD = 1

    spark = start_spark()

    sc = spark.sparkContext
    n_executors = len(sc._jsc.sc().statusTracker().getExecutorInfos())
    cores_per_executor = int(spark.conf.get("spark.executor.cores", str(sc.defaultParallelism)))
    executor_memory = spark.conf.get("spark.executor.memory", "unknown")

    env = {
        "months": MONTHS_TO_LOAD,
        "executors": n_executors,
        "cores": cores_per_executor,
        "memory": executor_memory,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }

    df, location_lookup = load_data(spark, MONTHS_TO_LOAD)

    inspect_data(df)

    df = preprocess(df)

    benchmarks = benchmark_pyspark(spark, list(data_files(months=MONTHS_TO_LOAD)), LOOKUP_PATH)

    regression_results = train_regression(spark, df)

    classification_results = train_classification(spark, df)

    save_results(benchmarks, regression_results, classification_results, env)

    spark.stop()


if __name__ == "__main__":
    main()
