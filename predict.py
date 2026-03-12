from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd


DATE_FORMAT = "%d-%m-%Y"
DEFAULT_DECISION_THRESHOLD = 0.50

# Raw input columns expected from user/batch file
REQUIRED_INPUT_COLUMNS = [
    "Origin",
    "Destination",
    "Shipment Date",
    "Planned Delivery Date",
    "Vehicle Type",
    "Distance (km)",
    "Weather Conditions",
    "Traffic Conditions",
]

# Categorical columns to normalize for inference consistency
CATEGORICAL_INPUT_COLUMNS = [
    "Origin",
    "Destination",
    "Vehicle Type",
    "Weather Conditions",
    "Traffic Conditions",
]

# Final model feature order used during training
MODEL_FEATURE_COLUMNS = [
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


def validate_input_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")


def build_inference_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build inference-time model features from raw input columns.
    """
    validate_input_columns(df_raw)
    data = df_raw[REQUIRED_INPUT_COLUMNS].copy()

    # Normalize categorical text inputs to reduce case/spacing mismatches
    for col in CATEGORICAL_INPUT_COLUMNS:
        data[col] = (
            data[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.title()
        )

    # Parse dates as DD-MM-YYYY
    data["Shipment Date"] = pd.to_datetime(
        data["Shipment Date"], format=DATE_FORMAT, errors="coerce"
    )
    data["Planned Delivery Date"] = pd.to_datetime(
        data["Planned Delivery Date"], format=DATE_FORMAT, errors="coerce"
    )

    # Feature engineering
    data["planned_transit_days"] = (
        data["Planned Delivery Date"] - data["Shipment Date"]
    ).dt.days
    data["ship_month"] = data["Shipment Date"].dt.month
    data["ship_weekday"] = data["Shipment Date"].dt.weekday

    # Basic validation for inference
    invalid_dates = data["Shipment Date"].isna() | data["Planned Delivery Date"].isna()
    if invalid_dates.any():
        bad_idx = data.index[invalid_dates].tolist()
        raise ValueError(
            f"Invalid date format at rows: {bad_idx}. Expected format: DD-MM-YYYY"
        )

    invalid_transit = data["planned_transit_days"].isna() | (data["planned_transit_days"] < 0)
    if invalid_transit.any():
        bad_idx = data.index[invalid_transit].tolist()
        raise ValueError(
            f"Invalid planned_transit_days at rows: {bad_idx}. "
            "Ensure Planned Delivery Date is on/after Shipment Date."
        )

    X = data[MODEL_FEATURE_COLUMNS].copy()
    return X


def risk_band(prob: float) -> str:
    """
    Convert delay probability into simple risk buckets.
    """
    if prob >= 0.70:
        return "High"
    if prob >= 0.40:
        return "Medium"
    return "Low"


def predict_batch(
    model_path: str,
    input_csv: str,
    output_csv: str,
) -> Path:
    model = joblib.load(model_path)
    df_input = pd.read_csv(input_csv)

    X_infer = build_inference_features(df_input)

    # Delay probability for positive class
    class_order: List[str] = list(model.classes_)
    pos_label = "Yes" if "Yes" in class_order else class_order[1]
    neg_label = [c for c in class_order if c != pos_label][0]
    pos_idx = class_order.index(pos_label)
    pred_proba = model.predict_proba(X_infer)[:, pos_idx]
    pred_labels = np.where(pred_proba >= DEFAULT_DECISION_THRESHOLD, pos_label, neg_label)

    # Build output
    output_df = df_input.copy()
    output_df["Predicted_Delayed"] = pred_labels
    output_df["Delay_Probability"] = np.round(pred_proba, 6)
    output_df["Risk_Band"] = output_df["Delay_Probability"].apply(risk_band)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(out_path, index=False)

    # Prediction summary (counts + percentages)
    total_rows = len(output_df)
    yes_count = int((output_df["Predicted_Delayed"] == "Yes").sum())
    no_count = int((output_df["Predicted_Delayed"] == "No").sum())
    yes_pct = round((yes_count / total_rows) * 100, 2) if total_rows else 0.0
    no_pct = round((no_count / total_rows) * 100, 2) if total_rows else 0.0

    summary_df = pd.DataFrame(
        [
            {"Label": "Yes", "Count": yes_count, "Percentage": yes_pct},
            {"Label": "No", "Count": no_count, "Percentage": no_pct},
            {"Label": "Total", "Count": total_rows, "Percentage": 100.0 if total_rows else 0.0},
        ]
    )
    summary_path = out_path.parent / "prediction_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nPrediction completed successfully.")
    print(f"Model loaded from: {model_path}")
    print(f"Input rows: {len(df_input)}")
    print(f"Output saved to: {out_path}")
    print(f"Probability column represents: P(Delayed = {pos_label})")
    print(f"Decision threshold used: {DEFAULT_DECISION_THRESHOLD:.2f}")
    print("\nPrediction Summary:")
    print(f"- Yes count: {yes_count} ({yes_pct}%)")
    print(f"- No count : {no_count} ({no_pct}%)")
    print(f"- Summary CSV saved to: {summary_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Run shipment delay predictions on new input CSV."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/model_pipeline.joblib",
        help="Path to trained model pipeline",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="sample_inference_input.csv",
        help="Path to raw input CSV for prediction",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="predictions_results/predictions.csv",
        help="Path to save predictions CSV",
    )
    args = parser.parse_args()

    predict_batch(
        model_path=args.model_path,
        input_csv=args.input_csv,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
