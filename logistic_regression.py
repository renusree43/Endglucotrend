import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def create_glucose_categories(df):

    data = df.copy()

    if "glucose" not in data.columns:

        raise ValueError(
            "The dataset does not contain a glucose column."
        )

    data["glucose"] = pd.to_numeric(
        data["glucose"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["glucose"]
    )

    # Simple project-level glucose categories.
    # These are used only for demonstrating
    # classification in the ML pipeline.

    def classify_glucose(value):

        if value < 70:

            return "Low"

        elif value <= 180:

            return "Normal"

        else:

            return "High"

    data["glucose_category"] = data[
        "glucose"
    ].apply(classify_glucose)

    return data


def run_logistic_regression(
        df,
        selected_columns
):

    data = create_glucose_categories(
        df
    )

    # Remove invalid selections

    selected_columns = [
        column
        for column in selected_columns
        if column in data.columns
        and column not in [
            "glucose_category",
            "timestamp",
            "user_id"
        ]
    ]

    if len(selected_columns) == 0:

        raise ValueError(
            "Please select at least one feature."
        )

    X = data[
        selected_columns
    ].copy()

    y = data[
        "glucose_category"
    ]

    # Convert numerical features

    usable_features = []

    for column in X.columns:

        converted = pd.to_numeric(
            X[column],
            errors="coerce"
        )

        if converted.notna().sum() > 0:

            X[column] = converted

            usable_features.append(
                column
            )

    X = X[
        usable_features
    ]

    if X.empty:

        raise ValueError(
            "No numerical features are available "
            "for Logistic Regression."
        )

    # Fill missing values

    for column in X.columns:

        X[column] = X[column].fillna(
            X[column].median()
        )

    # Remove rows still containing missing values

    valid_rows = X.notna().all(
        axis=1
    )

    X = X.loc[valid_rows]

    y = y.loc[valid_rows]

    if y.nunique() < 2:

        raise ValueError(
            "At least two glucose categories "
            "are required."
        )

    # Check class counts

    class_counts = y.value_counts()

    if class_counts.min() < 2:

        raise ValueError(
            "Each glucose category needs at least "
            "two observations."
        )

    # ---------------------------------------------------------
    # TRAIN TEST SPLIT
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ---------------------------------------------------------
    # LOGISTIC REGRESSION
    # ---------------------------------------------------------

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    return {

        "model": model,

        "features": usable_features,

        "target": "glucose_category",

        "train_rows": len(X_train),

        "test_rows": len(X_test),

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "classes": list(
            model.classes_
        )
    }