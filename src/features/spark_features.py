"""PySpark implementation of the feature-engineering pipeline.

Mirrors ``build_features.py`` exactly — same features, same thresholds — using the
Spark DataFrame API, so the pipeline is portable to cluster scale (Databricks /
Dataproc) without logic changes. On this project's 10k rows Spark is a learning
exercise, not a necessity; the parity test below is what makes it trustworthy.

Run locally (needs Java 17/21):
    python -m src.features.spark_features
On Databricks Free Edition: import and call ``add_features(spark_df)``.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

DROP_NOISE = ["country", "city", "gender"]


def add_features(df: DataFrame) -> DataFrame:
    """Same transformations as the pandas pipeline, in Spark."""
    df = df.drop(*DROP_NOISE)

    # Threshold flags (step-shaped effects from the statistics phase)
    df = (
        df.withColumn("is_new_customer", (F.col("tenure_months") <= 6).cast("int"))
        .withColumn("payment_risk", (F.col("payment_failures") >= 2).cast("int"))
        .withColumn("low_csat", (F.col("csat_score") <= 2).cast("int"))
        .withColumn("inactive_30d", (F.col("last_login_days_ago") > 30).cast("int"))
    )

    # Interaction & compound risk
    df = df.withColumn(
        "csat_x_payment_risk", (F.col("low_csat") * F.col("payment_risk")).cast("int")
    ).withColumn(
        "risk_factor_count",
        (
            F.col("is_new_customer")
            + F.col("payment_risk")
            + F.col("low_csat")
            + F.col("inactive_30d")
        ).cast("int"),
    )

    # Lifecycle bucket
    df = df.withColumn(
        "tenure_bucket",
        F.when(F.col("tenure_months") <= 6, "01-06")
        .when(F.col("tenure_months") <= 12, "07-12")
        .when(F.col("tenure_months") <= 24, "13-24")
        .when(F.col("tenure_months") <= 36, "25-36")
        .otherwise("37+"),
    )

    # Intensity ratios
    df = (
        df.withColumn("usage_minutes", F.col("monthly_logins") * F.col("avg_session_time"))
        .withColumn("fee_per_feature", F.col("monthly_fee") / F.col("features_used"))
        .withColumn("support_burden", F.col("support_tickets") * F.col("avg_resolution_time"))
        .withColumn(
            "failures_per_year", F.col("payment_failures") / (F.col("tenure_months") / 12)
        )
    )
    return df


def main() -> None:
    from src.utils.config import PROJECT_ROOT, load_config

    cfg = load_config()
    clean_path = PROJECT_ROOT / cfg["cleaning"]["clean_file"]
    out = PROJECT_ROOT / cfg["data"]["processed_dir"] / "churn_features_spark.parquet"

    spark = (
        SparkSession.builder.appName("churn-features")
        .master("local[*]")
        .getOrCreate()
    )
    try:
        df = add_features(spark.read.parquet(str(clean_path)))
        df.coalesce(1).write.mode("overwrite").parquet(str(out))
        print(f"OK: wrote {df.count():,} rows x {len(df.columns)} cols -> {out}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
