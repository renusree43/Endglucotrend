import os
import pandas as pd
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "processed")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


DATASET_NAME = "GlucoBench_benchmark_dataset.csv"

# GlucoBench measurements are recorded at 5-minute intervals.
# 3 steps = approximately 15 minutes ahead.
FORECAST_HORIZON = 3


def get_dataset_path(filename=None):

    if filename:
        path = os.path.join(UPLOAD_FOLDER, filename)

        if os.path.exists(path):
            return path

    csv_files = [
        f for f in os.listdir(UPLOAD_FOLDER)
        if f.lower().endswith(".csv")
    ]

    if not csv_files:
        return None

    # Prefer GlucoBench
    for file in csv_files:
        if "glucobench" in file.lower():
            return os.path.join(UPLOAD_FOLDER, file)

    return os.path.join(UPLOAD_FOLDER, csv_files[0])


def load_dataset(filename=None):

    path = get_dataset_path(filename)

    if path is None:
        raise FileNotFoundError(
            "No CSV dataset found in the uploads folder."
        )

    df = pd.read_csv(path)

    return df, path


def numeric_columns(df):

    return df.select_dtypes(
        include=np.number
    ).columns.tolist()


def categorical_columns(df):

    return df.select_dtypes(
        exclude=np.number
    ).columns.tolist()


def create_forecasting_dataset(df, horizon=FORECAST_HORIZON):

    data = df.copy()

    # ---------------------------------------------------------
    # TIMESTAMP
    # ---------------------------------------------------------

    if "timestamp" in data.columns:

        data["timestamp"] = pd.to_datetime(
            data["timestamp"],
            errors="coerce"
        )

        data = data.dropna(subset=["timestamp"])

    # ---------------------------------------------------------
    # CGM QUALITY
    # ---------------------------------------------------------

    if "cgm_quality_flag" in data.columns:

        data["cgm_quality_flag"] = pd.to_numeric(
            data["cgm_quality_flag"],
            errors="coerce"
        )

        data = data[
            data["cgm_quality_flag"].fillna(1) == 1
        ]

    # ---------------------------------------------------------
    # SORT DATA
    # ---------------------------------------------------------

    if "user_id" in data.columns and "timestamp" in data.columns:

        data = data.sort_values(
            ["user_id", "timestamp"]
        )

    elif "timestamp" in data.columns:

        data = data.sort_values("timestamp")

    # ---------------------------------------------------------
    # NUMERIC CONVERSION
    # ---------------------------------------------------------

    numeric_candidates = [
        "glucose",
        "insulin_bolus",
        "insulin_basal",
        "carbs",
        "exercise_steps",
        "heart_rate",
        "skin_temp",
        "gsr",
        "stress_level",
        "alcohol",
        "hbA1c",
        "age",
        "weight",
        "carb_ratio",
        "insulin_sensitivity"
    ]

    for column in numeric_candidates:

        if column in data.columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    # ---------------------------------------------------------
    # GLUCOSE LAG FEATURES
    # ---------------------------------------------------------

    if "glucose" not in data.columns:

        raise ValueError(
            "The dataset must contain a 'glucose' column."
        )

    if "user_id" in data.columns:

        grouped = data.groupby("user_id")["glucose"]

        data["glucose_lag_1"] = grouped.shift(1)
        data["glucose_lag_2"] = grouped.shift(2)
        data["glucose_lag_3"] = grouped.shift(3)

        # Future glucose target
        data["future_glucose"] = grouped.shift(-horizon)

    else:

        data["glucose_lag_1"] = data["glucose"].shift(1)
        data["glucose_lag_2"] = data["glucose"].shift(2)
        data["glucose_lag_3"] = data["glucose"].shift(3)

        data["future_glucose"] = data["glucose"].shift(
            -horizon
        )

    # ---------------------------------------------------------
    # FEATURES
    # ---------------------------------------------------------

    preferred_features = [
        "glucose",
        "glucose_lag_1",
        "glucose_lag_2",
        "glucose_lag_3",
        "insulin_bolus",
        "insulin_basal",
        "carbs",
        "exercise_steps",
        "heart_rate",
        "skin_temp",
        "gsr",
        "stress_level",
        "alcohol",
        "hbA1c",
        "age",
        "weight",
        "carb_ratio",
        "insulin_sensitivity"
    ]

    features = [
        column
        for column in preferred_features
        if column in data.columns
    ]

    if len(features) < 4:

        raise ValueError(
            "Not enough usable numerical features were found."
        )

    # ---------------------------------------------------------
    # KEEP ONLY REQUIRED DATA
    # ---------------------------------------------------------

    model_data = data[
        features + ["future_glucose"]
    ].copy()

    # Replace infinity
    model_data = model_data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Drop rows where target is missing
    model_data = model_data.dropna(
        subset=["future_glucose"]
    )

    return model_data, features


def prepare_model_data(model_data, features):

    data = model_data.copy()

    X = data[features].copy()
    y = data["future_glucose"].copy()

    medians = {}

    for column in features:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

        median_value = X[column].median()

        if pd.isna(median_value):
            median_value = 0

        medians[column] = float(median_value)

        X[column] = X[column].fillna(
            median_value
        )

    y = pd.to_numeric(
        y,
        errors="coerce"
    )

    valid = y.notna()

    X = X.loc[valid]
    y = y.loc[valid]

    return X, y, medians


def create_prediction_input(
        values,
        features,
        medians
):

    row = {}

    for feature in features:

        value = values.get(feature)

        if value is None or value == "":

            row[feature] = medians.get(
                feature,
                0
            )

        else:

            try:
                row[feature] = float(value)

            except ValueError:

                row[feature] = medians.get(
                    feature,
                    0
                )

    return pd.DataFrame(
        [row],
        columns=features
    )