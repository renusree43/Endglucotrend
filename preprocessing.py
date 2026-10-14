import pandas as pd
from sklearn.preprocessing import LabelEncoder


def handle_missing_values(
        df,
        column,
        method,
        constant_value=None
):

    updated_df = df.copy()

    if column not in updated_df.columns:

        return (
            updated_df,
            "Column not found.",
            ""
        )

    # ---------------------------------------------------------
    # MEAN
    # ---------------------------------------------------------

    if method == "mean":

        if not pd.api.types.is_numeric_dtype(
                updated_df[column]
        ):

            return (
                updated_df,
                "Mean can only be used for numerical columns.",
                ""
            )

        value = updated_df[column].mean()

        updated_df[column] = updated_df[column].fillna(
            value
        )

        message = (
            f"Missing values filled using mean: {value:.4f}"
        )

        code_used = (
            f"df['{column}'] = df['{column}'].fillna("
            f"df['{column}'].mean())"
        )

    # ---------------------------------------------------------
    # MEDIAN
    # ---------------------------------------------------------

    elif method == "median":

        if not pd.api.types.is_numeric_dtype(
                updated_df[column]
        ):

            return (
                updated_df,
                "Median can only be used for numerical columns.",
                ""
            )

        value = updated_df[column].median()

        updated_df[column] = updated_df[column].fillna(
            value
        )

        message = (
            f"Missing values filled using median: {value:.4f}"
        )

        code_used = (
            f"df['{column}'] = df['{column}'].fillna("
            f"df['{column}'].median())"
        )

    # ---------------------------------------------------------
    # MODE
    # ---------------------------------------------------------

    elif method == "mode":

        mode_values = updated_df[column].mode()

        if len(mode_values) == 0:

            return (
                updated_df,
                "Mode could not be calculated.",
                ""
            )

        value = mode_values.iloc[0]

        updated_df[column] = updated_df[column].fillna(
            value
        )

        message = (
            f"Missing values filled using mode: {value}"
        )

        code_used = (
            f"df['{column}'] = df['{column}'].fillna("
            f"df['{column}'].mode()[0])"
        )

    # ---------------------------------------------------------
    # CONSTANT
    # ---------------------------------------------------------

    elif method == "constant":

        if constant_value is None:

            constant_value = "0"

        updated_df[column] = updated_df[column].fillna(
            constant_value
        )

        message = (
            f"Missing values filled using constant: "
            f"{constant_value}"
        )

        code_used = (
            f"df['{column}'] = df['{column}'].fillna("
            f"{repr(constant_value)})"
        )

    # ---------------------------------------------------------
    # DROP ROW
    # ---------------------------------------------------------

    elif method == "drop_row":

        before = len(updated_df)

        updated_df = updated_df.dropna(
            subset=[column]
        )

        removed = before - len(updated_df)

        message = (
            f"{removed} rows removed."
        )

        code_used = (
            f"df = df.dropna(subset=['{column}'])"
        )

    # ---------------------------------------------------------
    # DROP COLUMN
    # ---------------------------------------------------------

    elif method == "drop_column":

        updated_df = updated_df.drop(
            columns=[column]
        )

        message = (
            f"Column '{column}' removed."
        )

        code_used = (
            f"df = df.drop(columns=['{column}'])"
        )

    else:

        return (
            updated_df,
            "Invalid method.",
            ""
        )

    return (
        updated_df,
        message,
        code_used
    )


def encode_column(
        df,
        column,
        method
):

    updated_df = df.copy()

    if column not in updated_df.columns:

        return (
            updated_df,
            "Column not found.",
            ""
        )

    # ---------------------------------------------------------
    # LABEL ENCODING
    # ---------------------------------------------------------

    if method == "label":

        encoder = LabelEncoder()

        updated_df[column] = updated_df[column].fillna(
            "Unknown"
        ).astype(str)

        updated_df[column] = encoder.fit_transform(
            updated_df[column]
        )

        code_used = (
            "from sklearn.preprocessing import LabelEncoder\n"
            f"encoder = LabelEncoder()\n"
            f"df['{column}'] = encoder.fit_transform("
            f"df['{column}'].astype(str))"
        )

        message = (
            f"Label encoding applied to '{column}'."
        )

    # ---------------------------------------------------------
    # ONE HOT
    # ---------------------------------------------------------

    elif method == "onehot":

        encoded = pd.get_dummies(
            updated_df[column].fillna("Unknown"),
            prefix=column,
            drop_first=False
        )

        updated_df = updated_df.drop(
            columns=[column]
        )

        updated_df = pd.concat(
            [updated_df, encoded],
            axis=1
        )

        code_used = (
            f"encoded = pd.get_dummies("
            f"df['{column}'], "
            f"prefix='{column}')"
        )

        message = (
            f"One-Hot Encoding applied to '{column}'."
        )

    else:

        return (
            updated_df,
            "Invalid encoding method.",
            ""
        )

    return (
        updated_df,
        message,
        code_used
    )