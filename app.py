import os
import base64
from io import BytesIO

import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session
)

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    Normalizer
)

from preprocessing import (
    handle_missing_values,
    encode_column
)

from linear_regression import (
    train_linear_regression,
    predict_glucose
)

from decision_tree import (
    train_decision_tree,
    predict_glucose as predict_tree
)

from random_forest import (
    train_random_forest,
    predict_glucose as predict_forest
)

from boosting import (
    train_boosting,
    predict_glucose as predict_boosting
)

from kmeans import (
    run_kmeans,
    predict_cluster
)

from logistic_regression import (
    run_logistic_regression
)

from pca import (
    run_pca
)


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

PROCESSED_FOLDER = os.path.join(
    BASE_DIR,
    "processed"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    PROCESSED_FOLDER,
    exist_ok=True
)


app = Flask(
    __name__,
    static_folder=os.path.join(
        BASE_DIR,
        "static"
    ),
    template_folder=os.path.join(
        BASE_DIR,
        "Templates"
    ),
    static_url_path="/static"
)

app.secret_key = "glucotrend_project"


def get_current_filename():

    filename = session.get(
        "dataset"
    )

    if filename:

        path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        if os.path.exists(path):

            return filename

    csv_files = [
        file
        for file in os.listdir(
            UPLOAD_FOLDER
        )
        if file.lower().endswith(".csv")
    ]

    if not csv_files:

        return None

    for file in csv_files:

        if "glucobench" in file.lower():

            return file

    return csv_files[0]


def get_current_dataframe():

    filename = get_current_filename()

    if filename is None:

        return None, None

    path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    if not os.path.exists(path):

        return None, None

    df = pd.read_csv(
        path
    )

    return df, filename


def dataframe_to_base64(
        figure
):

    buffer = BytesIO()

    figure.savefig(
        buffer,
        format="png",
        dpi=120,
        bbox_inches="tight"
    )

    buffer.seek(0)

    image = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    plt.close(
        figure
    )

    return image


def prediction_values_from_form(
        features
):

    values = {}

    for feature in features:

        values[feature] = request.form.get(
            feature,
            ""
        )

    return values


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route(
    "/upload",
    methods=["GET", "POST"]
)
def upload():

    error = None

    if request.method == "POST":

        file = request.files.get(
            "file"
        )

        if file is None or file.filename == "":

            error = "Please select a CSV file."

            return render_template(
                "upload.html",
                error=error
            )

        if not file.filename.lower().endswith(
                ".csv"
        ):

            error = "Please upload a CSV file."

            return render_template(
                "upload.html",
                error=error
            )

        file_path = os.path.join(
            UPLOAD_FOLDER,
            file.filename
        )

        file.save(
            file_path
        )

        session["dataset"] = file.filename

        return redirect(
            url_for("summary")
        )

    return render_template(
        "upload.html",
        error=error
    )


@app.route("/summary")
def summary():

    df, filename = get_current_dataframe()

    if df is None:

        return redirect(
            url_for("upload")
        )

    rows, columns = df.shape

    missing_values = int(
        df.isnull().sum().sum()
    )

    numerical_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        exclude="number"
    ).columns.tolist()

    return render_template(
        "summary.html",
        filename=filename,
        rows=rows,
        columns=columns,
        missing_values=missing_values,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        data=df.head(10).to_html(
            classes="data-table",
            index=False
        )
    )


@app.route(
    "/visualization",
    methods=["GET", "POST"]
)
def visualization():

    df, filename = get_current_dataframe()

    if df is None:

        return redirect(
            url_for("upload")
        )

    numerical_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        exclude="number"
    ).columns.tolist()

    columns = (
        numerical_columns +
        categorical_columns
    )

    graph = None
    selected_column = None
    graph_type = "histogram"
    error = None

    if request.method == "POST":

        graph_type = request.form.get(
            "graph_type"
        )

        selected_column = request.form.get(
            "column"
        )

        try:

            if selected_column not in columns:

                raise ValueError(
                    "Please select a valid column."
                )

            if graph_type == "histogram":

                if selected_column not in numerical_columns:

                    raise ValueError(
                        "Histogram requires a numerical column."
                    )

                figure = plt.figure(
                    figsize=(8, 5)
                )

                sns.histplot(
                    df[selected_column].dropna(),
                    kde=True
                )

                plt.title(
                    f"Distribution of {selected_column}"
                )

                plt.xlabel(
                    selected_column
                )

                plt.ylabel(
                    "Frequency"
                )

                graph = dataframe_to_base64(
                    figure
                )

            elif graph_type == "boxplot":

                if selected_column not in numerical_columns:

                    raise ValueError(
                        "Box Plot requires a numerical column."
                    )

                figure = plt.figure(
                    figsize=(8, 5)
                )

                sns.boxplot(
                    y=df[selected_column]
                )

                plt.title(
                    f"Box Plot of {selected_column}"
                )

                plt.ylabel(
                    selected_column
                )

                graph = dataframe_to_base64(
                    figure
                )

            elif graph_type == "bar":

                figure = plt.figure(
                    figsize=(9, 5)
                )

                df[selected_column].astype(
                    str
                ).value_counts().head(
                    15
                ).plot(
                    kind="bar"
                )

                plt.title(
                    f"Distribution of {selected_column}"
                )

                plt.xlabel(
                    selected_column
                )

                plt.ylabel(
                    "Count"
                )

                plt.xticks(
                    rotation=45,
                    ha="right"
                )

                graph = dataframe_to_base64(
                    figure
                )

            else:

                raise ValueError(
                    "Invalid graph type selected."
                )

        except Exception as e:

            error = str(e)

    return render_template(
        "visualization.html",
        filename=filename,
        columns=columns,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        selected_column=selected_column,
        graph_type=graph_type,
        graph=graph,
        error=error
    )


@app.route(
    "/scaling",
    methods=["GET", "POST"]
)
def scaling():

    df, filename = get_current_dataframe()

    if df is None:

        return redirect(
            url_for("upload")
        )

    numerical_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    result = None
    error = None

    if request.method == "POST":

        column = request.form.get(
            "column"
        )

        method = request.form.get(
            "method"
        )

        try:

            if column not in numerical_columns:

                raise ValueError(
                    "Please select a numerical column."
                )

            data = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            data = data.fillna(
                data.median()
            )

            values = data.values.reshape(
                -1,
                1
            )

            if method == "standard":

                scaler = StandardScaler()

                scaled = scaler.fit_transform(
                    values
                )

            elif method == "minmax":

                scaler = MinMaxScaler()

                scaled = scaler.fit_transform(
                    values
                )

            elif method == "normalizer":

                scaler = Normalizer()

                scaled = scaler.fit_transform(
                    values
                )

            else:

                raise ValueError(
                    "Please select a scaling method."
                )

            result_df = pd.DataFrame(
                {
                    "Original": data.head(10),
                    "Scaled": scaled[:10].flatten()
                }
            )

            result = result_df.to_html(
                classes="data-table",
                index=False
            )

        except Exception as e:

            error = str(e)

    return render_template(
        "scaling.html",
        filename=filename,
        columns=numerical_columns,
        numerical_columns=numerical_columns,
        result=result,
        error=error
    )


@app.route(
    "/missing-values",
    methods=["GET", "POST"]
)
def missing_values():

    df, filename = get_current_dataframe()

    if df is None:

        return redirect(
            url_for("upload")
        )

    columns = df.columns.tolist()

    result = None
    message = None
    code_used = None
    error = None

    if request.method == "POST":

        column = request.form.get(
            "column"
        )

        method = request.form.get(
            "method"
        )

        constant_value = request.form.get(
            "constant_value"
        )

        try:

            (
                updated_df,
                message,
                code_used
            ) = handle_missing_values(
                df.copy(),
                column,
                method,
                constant_value
            )

            result = updated_df.head(
                10
            ).to_html(
                classes="data-table",
                index=False
            )

        except Exception as e:

            error = str(e)

    return render_template(
        "missing_values.html",
        filename=filename,
        columns=columns,
        result=result,
        message=message,
        code_used=code_used,
        error=error
    )


@app.route(
    "/save-dataset",
    methods=["POST"]
)
def save_dataset():

    df, filename = get_current_dataframe()

    if df is None:

        return redirect(
            url_for("upload")
        )

    path = os.path.join(
        PROCESSED_FOLDER,
        "processed_dataset.csv"
    )

    df.to_csv(
        path,
        index=False
    )

    return redirect(
        url_for("summary")
    )


@app.route(
    "/encoding",
    methods=["GET", "POST"]
)
def encoding():

    df, filename = get_current_dataframe()

    if df is None:

        return redirect(
            url_for("upload")
        )

    categorical_columns = df.select_dtypes(
        exclude="number"
    ).columns.tolist()

    if "timestamp" in categorical_columns:

        categorical_columns.remove(
            "timestamp"
        )

    if "user_id" in categorical_columns:

        categorical_columns.remove(
            "user_id"
        )

    message = None
    code_used = None
    preview = None
    error = None

    if request.method == "POST":

        column = request.form.get(
            "column"
        )

        method = request.form.get(
            "method"
        )

        try:

            if column not in categorical_columns:

                raise ValueError(
                    "Please select a valid categorical column."
                )

            (
                updated_df,
                message,
                code_used
            ) = encode_column(
                df.copy(),
                column,
                method
            )

            preview = updated_df.head(
                10
            ).to_html(
                classes="data-table",
                index=False
            )

        except Exception as e:

            error = str(e)

    return render_template(
        "encoding.html",
        filename=filename,
        categorical_columns=categorical_columns,
        message=message,
        code_used=code_used,
        preview=preview,
        error=error
    )


@app.route(
    "/save-encoding-dataset",
    methods=["POST"]
)
def save_encoding_dataset():

    df, filename = get_current_dataframe()

    if df is None:

        return redirect(
            url_for("upload")
        )

    path = os.path.join(
        PROCESSED_FOLDER,
        "encoded_dataset.csv"
    )

    df.to_csv(
        path,
        index=False
    )

    return redirect(
        url_for("summary")
    )


@app.route("/pca")
def pca_page():

    df, filename = get_current_dataframe()

    if df is None:

        return redirect(
            url_for("upload")
        )

    results = None
    error = None

    try:

        results = run_pca(
            df
        )

    except Exception as e:

        error = str(e)

    return render_template(
        "pca.html",
        filename=filename,
        results=results,
        error=error
    )


@app.route(
    "/linear-regression",
    methods=["GET", "POST"]
)
def linear_regression():

    df, filename = get_current_dataframe()

    if df is None:

        return redirect(
            url_for("upload")
        )

    results = None
    error = None
    prediction = None

    try:

        results = train_linear_regression(
            df
        )

    except Exception as e:

        error = str(e)

    if request.method == "POST" and results:

        try:

            values = prediction_values_from_form(
                results["features"]
            )

            prediction = predict_glucose(
                results["model"],
                values,
                results["features"],
                results["medians"]
            )

        except Exception as e:

            error = str(e)

    return render_template(
        "linear_regression.html",
        filename=filename,
        results=results,
        prediction=prediction,
        error=error
    )


@app.route(
    "/decision-tree",
    methods=["GET", "POST"]
)
def decision_tree():

    df, filename = get_current_dataframe()

    if df is None:

        return redirect(
            url_for("upload")
        )

    results = None
    error = None
    prediction = None

    try:

        results = train_decision_tree(
            df
        )

    except Exception as e:

        error = str(e)

    if request.method == "POST" and results:

        try:

            values = prediction_values_from_form(
                results["features"]
            )

            prediction = predict_tree(
                results["model"],
                values,
                results["features"],
                results["medians"]
            )

        except Exception as e:

            error = str(e)

    return render_template(
        "decision_tree.html",
        filename=filename,
        results=results,
        prediction=prediction,
        error=error
    )


@app.route(
    "/random-forest",
    methods=["GET", "POST"]
)
def random_forest():

    df, filename = get_current_dataframe()

    if df is None:

        return redirect(
            url_for("upload")
        )

    results = None
    error = None
    prediction = None

    try:

        results = train_random_forest(
            df
        )

    except Exception as e:

        error = str(e)

    if request.method == "POST" and results:

        try:

            values = prediction_values_from_form(
                results["features"]
            )

            prediction = predict_forest(
                results["model"],
                values,
                results["features"],
                results["medians"]
            )

        except Exception as e:

            error = str(e)

    return render_template(
        "random_forest.html",
        filename=filename,
        results=results,
        prediction=prediction,
        error=error
    )


@app.route(
    "/boosting",
    methods=["GET", "POST"]
)
def boosting():

    df, filename = get_current_dataframe()

    if df is None:

        return redirect(
            url_for("upload")
        )

    results = None
    error = None
    prediction = None

    try:

        results = train_boosting(
            df
        )

    except Exception as e:

        error = str(e)

    if request.method == "POST" and results:

        try:

            values = prediction_values_from_form(
                results["features"]
            )

            prediction = predict_boosting(
                results["model"],
                values,
                results["features"],
                results["medians"]
            )

        except Exception as e:

            error = str(e)

    return render_template(
        "boosting.html",
        filename=filename,
        results=results,
        prediction=prediction,
        error=error
    )


@app.route(
    "/k_mean",
    methods=["GET", "POST"]
)
def k_mean():

    df, filename = get_current_dataframe()

    if df is None:

        return redirect(
            url_for("upload")
        )

    results = None
    error = None
    prediction = None

    try:

        results = run_kmeans(
            df
        )

    except Exception as e:

        error = str(e)

    if request.method == "POST" and results:

        try:

            values = prediction_values_from_form(
                results["features"]
            )

            prediction = predict_cluster(
                results["model"],
                results["scaler"],
                values,
                results["features"],
                results["medians"]
            )

        except Exception as e:

            error = str(e)

    return render_template(
        "k_mean.html",
        filename=filename,
        results=results,
        prediction=prediction,
        error=error
    )


@app.route(
    "/logistic-regression",
    methods=["GET", "POST"]
)
def logistic_regression():

    df, filename = get_current_dataframe()

    if df is None:

        return redirect(
            url_for("upload")
        )

    columns = df.columns.tolist()

    selected_features = []

    results = None
    error = None

    if request.method == "POST":

        selected_features = request.form.getlist(
            "features"
        )

        try:

            if len(selected_features) == 0:

                raise ValueError(
                    "Please select at least one feature."
                )

            results = run_logistic_regression(
                df,
                selected_features
            )

        except Exception as e:

            error = str(e)

    return render_template(
        "logistic_regression.html",
        filename=filename,
        columns=columns,
        selected_features=selected_features,
        results=results,
        error=error
    )


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True
    )