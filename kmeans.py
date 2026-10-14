import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def run_kmeans(df):

    features = [
        "glucose",
        "carbs",
        "exercise_steps",
        "heart_rate",
        "stress_level"
    ]

    features = [
        column
        for column in features
        if column in df.columns
    ]

    if len(features) < 2:
        raise ValueError(
            "At least two suitable numerical features are required for K-Means."
        )

    data = df[features].copy()

    for column in features:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.fillna(
        data.median()
    )

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        data
    )

    model = KMeans(
        n_clusters=3,
        n_init=10,
        random_state=42
    )

    clusters = model.fit_predict(
        scaled_data
    )

    clustered_data = data.copy()

    clustered_data["Cluster"] = clusters

    cluster_centers = pd.DataFrame(
        scaler.inverse_transform(
            model.cluster_centers_
        ),
        columns=features
    )

    return {
        "model": model,
        "scaler": scaler,
        "features": features,
        "clustered_data": clustered_data,
        "cluster_centers": cluster_centers,
        "clusters": clusters,
        "medians": data.median().to_dict()
    }


def predict_cluster(
    model,
    scaler,
    values,
    features,
    medians
):

    input_data = {}

    for feature in features:

        value = values.get(
            feature,
            ""
        )

        if value == "":

            input_data[feature] = medians.get(
                feature,
                0
            )

        else:

            try:

                input_data[feature] = float(
                    value
                )

            except ValueError:

                input_data[feature] = medians.get(
                    feature,
                    0
                )

    input_df = pd.DataFrame(
        [input_data],
        columns=features
    )

    scaled_input = scaler.transform(
        input_df
    )

    cluster = model.predict(
        scaled_input
    )[0]

    return int(cluster)