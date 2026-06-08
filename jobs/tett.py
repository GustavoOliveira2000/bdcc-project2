# %%

# ============================================================
# pyspark_pipeline.ipynb
# PySpark ML Pipeline - NYC Taxi Dataset
# ============================================================

# 1. Imports

import time
import requests
import pandas as pd

from IPython.display import display

from pydantic import ByteSize
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from collections.abc import Generator
from contextlib import contextmanager
from datetime import timedelta, datetime
from pathlib import Path
from time import perf_counter
from typing import TypedDict


# %%
# 2. Start Spark session

spark = (
    SparkSession.builder
    .appName("BDCC PySpark Taxi Pipeline")
    # .master("local[10]")
    .master("local[*]")
    .config("spark.driver.memory", "16g")
    .getOrCreate()
)

spark

# %%

# 3. Configuration

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

N_JOBS = -1

_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_PATH = (
    "gs://"
    "{BUCKET_NAME}/results/"
    "{months}_{timestamp}_{filename}"
)


# %%

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


# %%

# 4. Load data

MONTHS_TO_LOAD = 1

with timeit("Loading data"):
    df = spark.read.parquet(*list(data_files(months=MONTHS_TO_LOAD)))

location_lookup = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(LOOKUP_PATH)
)

df.show(5)


# %%

df.orderBy(F.col("tpep_pickup_datetime").desc()).show(5)


# %%
# 5. Basic dataset inspection

print("Columns:")
print(df.columns)

print("\nSchema:")
df.printSchema()

print("\nNumber of partitions:")
print(df.rdd.getNumPartitions())


# %%
# 6. Preprocessing

df = df.select(*(FEATURES + [TARGET]))

df = df.dropna()

df = df.filter(
    (F.col("fare_amount") > 0) &
    (F.col("trip_distance") > 0) &
    (F.col("passenger_count") > 0)
)

df.show(5)


# %%

def benchmark_pyspark(data_paths: list[str], lookup_path: str):
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


# %%

benchmarks = benchmark_pyspark(list(data_files(months=MONTHS_TO_LOAD)), LOOKUP_PATH)

benchmarks


# %%
# 7. Feature vector preparation

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


# %%
# 8. Regression pipeline - GBTRegressor
# Note: PySpark MLlib does not use XGBRegressor by default.
# GBTRegressor is the closest native Spark tree boosting alternative.

reg_model = GBTRegressor(

    featuresCol=FEATURES_COL,
    labelCol=LABEL_COL,
    maxIter=XGB_N_ESTIMATORS,
    maxDepth=XGB_MAX_DEPTH,
    stepSize=XGB_LEARNING_RATE,
    seed=SEED

)

# Cross-Validation

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
    reg_predictions.count()  # because, sometimes spark doesnt execute predictions immediately...

regression_prediction_time = duration["duration"]

reg_predictions.select("prediction", LABEL_COL).show(5)


# %%
# 9. Regression metrics

mae_evaluator = RegressionEvaluator(
    labelCol=LABEL_COL,
    predictionCol="prediction",
    metricName="mae"
)

mse_evaluator = RegressionEvaluator(
    labelCol=LABEL_COL,
    predictionCol="prediction",
    metricName="mse"
)

rmse_evaluator = RegressionEvaluator(
    labelCol=LABEL_COL,
    predictionCol="prediction",
    metricName="rmse"
)

r2_evaluator = RegressionEvaluator(
    labelCol=LABEL_COL,
    predictionCol="prediction",
    metricName="r2"
)

regression_results = {
    "library": "PySpark",
    "task": "regression",
    "model": "GBTRegressor",
    "cv_folds": REGRESSION_CV_FOLDS,
    "cv_time": regression_cv_time,
    "cv_rmse_mean": regression_cv_rmse_mean,
    "cv_rmse_std": regression_cv_rmse_std,
    "train_time": regression_train_time,
    "regression_prediction_time": regression_prediction_time,
    "mae": mae_evaluator.evaluate(reg_predictions),
    "mse": mse_evaluator.evaluate(reg_predictions),
    "rmse": rmse_evaluator.evaluate(reg_predictions),
    "r2": r2_evaluator.evaluate(reg_predictions),
}

regression_results


# %%
# 10. Classification pipeline - LogisticRegression

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

TRAIN_RATIO = 0.8

train_cls_df = (
    classification_ml_df
    .sampleBy(
        LABEL_COL,
        fractions={0: TRAIN_RATIO, 1: TRAIN_RATIO, 2: TRAIN_RATIO},
        seed=42
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

# Cross Validation model

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
    cls_predictions.count()

classification_prediction_time = duration["duration"]

cls_predictions.select("prediction", LABEL_COL).show(5)


# %%
# 11. Classification metrics

accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol=LABEL_COL,
    predictionCol="prediction",
    metricName="accuracy"
)

precision_evaluator = MulticlassClassificationEvaluator(
    labelCol=LABEL_COL,
    predictionCol="prediction",
    metricName="weightedPrecision"
)

recall_evaluator = MulticlassClassificationEvaluator(
    labelCol=LABEL_COL,
    predictionCol="prediction",
    metricName="weightedRecall"
)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol=LABEL_COL,
    predictionCol="prediction",
    metricName="f1"
)

classification_results = {
    "library": "PySpark",
    "task": "classification",
    "model": "LogisticRegression",
    "cv_folds": CLASSIFICATION_CV_FOLDS,
    "cv_time": classification_cv_time,
    "cv_f1_macro_mean": classification_cv_f1_mean,
    "cv_f1_macro_std": classification_cv_f1_std,
    "train_time": classification_train_time,
    "classification_prediction_time": classification_prediction_time,
    "accuracy": accuracy_evaluator.evaluate(cls_predictions),
    "precision_weighted": precision_evaluator.evaluate(cls_predictions),
    "recall_weighted": recall_evaluator.evaluate(cls_predictions),
    "f1_weighted": f1_evaluator.evaluate(cls_predictions),
}

classification_results


# %%

# 12. Save results

benchmark_results_df = pd.DataFrame([
    {"operation": operation, "library": "PySpark", "duration": duration}
    for operation, duration in benchmarks.items()
])

ml_results_df = pd.DataFrame([
    regression_results,
    classification_results
])

benchmark_path = RESULTS_PATH.format(
    BUCKET_NAME=BUCKET_NAME,
    months=MONTHS_TO_LOAD,
    timestamp=_timestamp,
    filename="pyspark_benchmark_results.csv"
)

pipeline_path = RESULTS_PATH.format(
    BUCKET_NAME=BUCKET_NAME,
    months=MONTHS_TO_LOAD,
    timestamp=_timestamp,
    filename="pyspark_pipeline_results.csv"
)

benchmark_results_df.to_csv(benchmark_path, index=False, storage_options={"token": "google_default"})
ml_results_df.to_csv(pipeline_path, index=False, storage_options={"token": "google_default"})

print("Benchmark results:")
display(benchmark_results_df)

print("ML pipeline results:")
display(ml_results_df)
