import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filepath):

    return pd.read_csv(
        filepath
    )


def create_histogram(
        df,
        column
):

    plt.figure(
        figsize=(8, 5)
    )

    sns.histplot(
        df[column].dropna(),
        kde=True
    )

    plt.title(
        f"Distribution of {column}"
    )

    plt.xlabel(column)

    plt.ylabel("Frequency")

    plt.tight_layout()

    path = "histogram.png"

    plt.savefig(
        path,
        dpi=120
    )

    plt.close()

    return path


def create_boxplot(
        df,
        column
):

    plt.figure(
        figsize=(8, 5)
    )

    sns.boxplot(
        y=df[column]
    )

    plt.title(
        f"Box Plot of {column}"
    )

    plt.tight_layout()

    path = "boxplot.png"

    plt.savefig(
        path,
        dpi=120
    )

    plt.close()

    return path


def create_bargraph(
        df,
        column
    ):

    plt.figure(
        figsize=(9, 5)
    )

    df[column].astype(
        str
    ).value_counts().head(15).plot(
        kind="bar"
    )

    plt.title(
        f"Bar Graph of {column}"
    )

    plt.xlabel(column)

    plt.ylabel("Count")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    path = "bargraph.png"

    plt.savefig(
        path,
        dpi=120
    )

    plt.close()

    return path


def create_heatmap(df):

    numeric_df = df.select_dtypes(
        include="number"
    )

    plt.figure(
        figsize=(12, 8)
    )

    sns.heatmap(
        numeric_df.corr(),
        annot=False,
        cmap="coolwarm"
    )

    plt.title(
        "Correlation Heatmap"
    )

    plt.tight_layout()

    path = "heatmap.png"

    plt.savefig(
        path,
        dpi=120
    )

    plt.close()

    return path