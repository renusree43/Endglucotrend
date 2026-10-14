import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from config import (
    create_forecasting_dataset,
    prepare_model_data
)


def train_linear_regression(df):

    model_data, features = create_forecasting_dataset(df)

    X, y, medians = prepare_model_data(
        model_data,
        features
    )

    if len(X) < 10:

        raise ValueError(
            "Not enough data for Linear Regression."
        )

    # Chronological split
    split_index = int(len(X) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return {
        "model": model,
        "features": features,
        "medians": medians,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "actual": y_test.tolist(),
        "predicted": predictions.tolist()
    }


def predict_glucose(
        model,
        feature_values,
        features,
        medians
):

    row = {}

    for feature in features:

        value = feature_values.get(feature)

        if value is None:

            row[feature] = medians.get(
                feature,
                0
            )

        else:

            try:
                row[feature] = float(value)

            except:
                row[feature] = medians.get(
                    feature,
                    0
                )

    input_df = pd.DataFrame(
        [row],
        columns=features
    )

    return float(
        model.predict(input_df)[0]
    )