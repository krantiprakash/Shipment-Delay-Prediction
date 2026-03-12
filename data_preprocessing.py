from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.2
DATE_FORMAT = "%d-%m-%Y"

TARGET_COL = "Delayed"
RAW_COLUMNS = [
    "Origin",
    "Destination",
    "Shipment Date",
    "Planned Delivery Date",
    "Vehicle Type",
    "Distance (km)",
    "Weather Conditions",
    "Traffic Conditions",
    TARGET_COL,
]

FINAL_FEATURES = [
    "Origin",
    "Destination",
    "Vehicle Type",
    "Distance (km)",
    "Weather Conditions",
    "Traffic Conditions",
    "planned_transit_days",
    "ship_month",
    "ship_weekday",
]


@dataclass
class PreparedData:
    X: pd.DataFrame
    y: pd.Series
    cleaned_df: pd.DataFrame


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_raw_data(csv_path: str | Path) -> pd.DataFrame:
    """Load raw CSV data."""
    return pd.read_csv(csv_path)


def validate_columns(df: pd.DataFrame, required_columns: List[str] | None = None) -> None:
    """Validate if all required columns exist in dataframe."""
    required = required_columns or RAW_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def prepare_features(df: pd.DataFrame) -> PreparedData:
    """
    Apply finalized preprocessing rules from notebook decisions:
    - Keep leakage-safe raw columns
    - Fill missing Vehicle Type with mode
    - Parse dates as DD-MM-YYYY
    - Engineer planned_transit_days, ship_month, ship_weekday
    - Remove rows with invalid/negative transit days
    """
    validate_columns(df, RAW_COLUMNS)
    data = df[RAW_COLUMNS].copy()

    # 1) Fill Vehicle Type with mode
    vehicle_mode = data["Vehicle Type"].mode()[0]
    data["Vehicle Type"] = data["Vehicle Type"].fillna(vehicle_mode)

    # 2) Parse dates
    data["Shipment Date"] = pd.to_datetime(
        data["Shipment Date"], format=DATE_FORMAT, errors="coerce"
    )
    data["Planned Delivery Date"] = pd.to_datetime(
        data["Planned Delivery Date"], format=DATE_FORMAT, errors="coerce"
    )

    # 3) Feature engineering
    data["planned_transit_days"] = (
        data["Planned Delivery Date"] - data["Shipment Date"]
    ).dt.days
    data["ship_month"] = data["Shipment Date"].dt.month
    data["ship_weekday"] = data["Shipment Date"].dt.weekday

    # 4) Drop invalid rows (failed date parsing or negative transit days)
    data = data.dropna(
        subset=["Shipment Date", "Planned Delivery Date", "planned_transit_days"]
    ).copy()
    data = data[data["planned_transit_days"] >= 0].copy()

    # 5) Final X/y
    X = data[FINAL_FEATURES].copy()
    y = data[TARGET_COL].copy()

    return PreparedData(X=X, y=y, cleaned_df=data)


def split_data(X: pd.DataFrame, y: pd.Series) -> SplitData:
    """Create stratified train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return SplitData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


def build_preprocessor(X_train: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Build preprocessing pipeline:
    - Numeric: median imputer + standard scaler
    - Categorical: most_frequent imputer + one-hot encoder
    """
    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    return preprocessor, numeric_features, categorical_features


def get_training_objects(csv_path: str | Path) -> Tuple[SplitData, ColumnTransformer]:
    """
    Convenience wrapper for train.py:
    - load
    - prepare features
    - split
    - build preprocessor
    """
    raw_df = load_raw_data(csv_path)
    prepared = prepare_features(raw_df)
    split = split_data(prepared.X, prepared.y)
    preprocessor, _, _ = build_preprocessor(split.X_train)
    return split, preprocessor


def _run_cli(csv_path: str) -> None:
    """CLI summary for quick sanity checks."""
    raw_df = load_raw_data(csv_path)
    print(f"Raw shape: {raw_df.shape}")

    prepared = prepare_features(raw_df)
    print(f"Cleaned shape after feature engineering: {prepared.cleaned_df.shape}")
    print(f"X shape: {prepared.X.shape}")
    print(f"y shape: {prepared.y.shape}")
    print("Final features:")
    for col in prepared.X.columns:
        print(f"- {col}")

    split = split_data(prepared.X, prepared.y)
    print(f"\nX_train: {split.X_train.shape}, X_test: {split.X_test.shape}")
    print(f"y_train: {split.y_train.shape}, y_test: {split.y_test.shape}")

    _, num_cols, cat_cols = build_preprocessor(split.X_train)
    print(f"\nNumeric features: {num_cols}")
    print(f"Categorical features: {cat_cols}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preprocessing for shipment delay model.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="Training Data.csv",
        help="Path to input CSV file",
    )
    args = parser.parse_args()
    _run_cli(args.data_path)
