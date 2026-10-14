import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def run_pca(df):

    # Select numerical columns
    data = df.select_dtypes(
        include="number"
    ).copy()

    if len(data.columns) < 2:
        raise ValueError(
            "At least two numerical columns are required for PCA."
        )

    # Remove columns which contain only missing values
    data = data.dropna(
        axis=1,
        how="all"
    )

    if len(data.columns) < 2:
        raise ValueError(
            "Not enough numerical columns are available for PCA."
        )

    # Fill missing values using median
    data = data.fillna(
        data.median()
    )

    # Remove any remaining missing values
    data = data.dropna(
        axis=1,
        how="any"
    )

    if len(data.columns) < 2:
        raise ValueError(
            "Not enough usable numerical columns are available for PCA."
        )

    # Scale the data
    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        data
    )

    # Apply PCA
    pca = PCA(
        n_components=2
    )

    principal_components = pca.fit_transform(
        scaled_data
    )

    # Create PCA dataframe
    pca_data = pd.DataFrame(
        principal_components,
        columns=[
            "PC1",
            "PC2"
        ]
    )

    # Explained variance
    explained_variance = (
        pca.explained_variance_ratio_ * 100
    )

    return {
        "features": data.columns.tolist(),

        "pca_data": pca_data,

        "pc1_variance": round(
            explained_variance[0],
            2
        ),

        "pc2_variance": round(
            explained_variance[1],
            2
        ),

        "total_variance": round(
            explained_variance.sum(),
            2
        )
    }