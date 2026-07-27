import pandas as pd
from pandera import Column, DataFrameSchema

MISSING_THRESHOLD = 0.3  # tune: above this, missingness is a signal, not noise


def clean_hpi_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Impute and dedupe the (small) HPI dataframe.

    Numeric columns are median-filled; categorical columns are mode-filled
    unless missingness exceeds MISSING_THRESHOLD, in which case a "Missing"
    sentinel is used instead (at that point absence is itself a signal).
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isna().any():
            missing_frac = df[col].isna().mean()
            if missing_frac > MISSING_THRESHOLD:
                df[col] = df[col].fillna("Missing")
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    return df.drop_duplicates()


def validate_no_nulls(df: pd.DataFrame) -> None:
    """Post-imputation contract: nothing should be null anymore."""
    schema = DataFrameSchema(
        {col: Column(nullable=False) for col in df.columns},
        strict=False,
    )
    schema.validate(df, lazy=True)


def normalize_ppd_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Light normalization for a Price Paid Data chunk.

    Blank saon/locality/street values in PPD are meaningful ("no data for
    this field"), not missingness to impute, so this only trims/normalizes
    text and dedupes - it never fills values in.
    """
    df = df.copy()

    text_cols = [
        "unique_id", "postcode", "property_type", "new_build", "estate_type",
        "saon", "paon", "street", "locality", "town", "district", "county",
        "ppd_category", "record_status",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].str.strip()

    if "postcode" in df.columns:
        df["postcode"] = df["postcode"].str.upper()

    if "price_paid" in df.columns:
        # Plain Python int/None (not numpy/pandas nullable-Int64 scalars) -
        # sqlite3 binds numpy integer scalars via the buffer protocol as a
        # BLOB instead of an INTEGER, silently corrupting the column.
        price_numeric = pd.to_numeric(df["price_paid"], errors="coerce")
        df["price_paid"] = [int(v) if pd.notna(v) else None for v in price_numeric]

    if "deed_date" in df.columns:
        df["deed_date"] = pd.to_datetime(
            df["deed_date"], format="%Y-%m-%d %H:%M", errors="coerce"
        ).dt.strftime("%Y-%m-%d")

    return df.drop_duplicates(subset=["unique_id"])
