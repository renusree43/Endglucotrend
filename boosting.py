import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from config import (
    create_forecasting_dataset,
    prepare_model_data
)


def train_boosting(df):

    model_data, features = create_forecasting_dataset(df)

    X, y, medians = prepare_model_data(
        model_data,
        features
    )

    split_index = int(
        len(X) * 0.80
    )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=1.0,
        random_state=42
    )

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

    feature_importance = dict(
        zip(
            features,
            model.feature_importances_
        )
    )

    staged_rmse = []

    for stage_prediction in model.staged_predict(
        X_test
    ):

        stage_rmse = np.sqrt(
            mean_squared_error(
                y_test,
                stage_prediction
            )
        )

        staged_rmse.append(
            stage_rmse
        )

    return {
        "model": model,
        "features": features,
        "medians": medians,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "feature_importance": feature_importance,
        "staged_rmse": staged_rmse,
        "train_rows": len(X_train),
        "test_rows": len(X_test)
    }


def predict_glucose(
        model,
        feature_values,
        features,
        medians
):

    row = {}

    for feature in features:

        value = feature_values.get(
            feature
        )

        if value is None or value == "":

            row[feature] = medians.get(
                feature,
                0
            )

        else:

            try:

                row[feature] = float(
                    value
                )

            except ValueError:

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