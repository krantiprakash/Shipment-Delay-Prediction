from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from data_preprocessing import RANDOM_STATE, get_training_objects


FINAL_MODEL_NAME = "logistic_regression_baseline"


def build_model() -> LogisticRegression:
    """Build finalized model selected from notebook experiments."""
    return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)


def get_pos_label(y_train, y_test):
    classes = sorted(list(set(y_train).union(set(y_test))))
    return "Yes" if "Yes" in classes else classes[1]


def compute_metrics(
    pipeline: Pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
    pos_label: str,
) -> Dict[str, float]:
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    metrics = {
        "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
        "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
        "train_precision": float(precision_score(y_train, y_train_pred, pos_label=pos_label)),
        "test_precision": float(precision_score(y_test, y_test_pred, pos_label=pos_label)),
        "train_recall": float(recall_score(y_train, y_train_pred, pos_label=pos_label)),
        "test_recall": float(recall_score(y_test, y_test_pred, pos_label=pos_label)),
        "train_f1": float(f1_score(y_train, y_train_pred, pos_label=pos_label)),
        "test_f1": float(f1_score(y_test, y_test_pred, pos_label=pos_label)),
    }

    # ROC-AUC if available
    metrics["train_roc_auc"] = np.nan
    metrics["test_roc_auc"] = np.nan
    if hasattr(pipeline, "predict_proba"):
        class_order = list(pipeline.classes_)
        pos_idx = class_order.index(pos_label)
        y_train_prob = pipeline.predict_proba(X_train)[:, pos_idx]
        y_test_prob = pipeline.predict_proba(X_test)[:, pos_idx]
        metrics["train_roc_auc"] = float(roc_auc_score(y_train, y_train_prob))
        metrics["test_roc_auc"] = float(roc_auc_score(y_test, y_test_prob))

    # Fit-gap quick diagnostics
    metrics["accuracy_gap_train_minus_test"] = float(
        metrics["train_accuracy"] - metrics["test_accuracy"]
    )
    metrics["f1_gap_train_minus_test"] = float(metrics["train_f1"] - metrics["test_f1"])
    return metrics


def train_and_save(
    data_path: str,
    output_dir: str,
) -> Tuple[Path, Path]:
    split_data, preprocessor = get_training_objects(data_path)
    model = build_model()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(split_data.X_train, split_data.y_train)

    pos_label = get_pos_label(split_data.y_train, split_data.y_test)
    metrics = compute_metrics(
        pipeline,
        split_data.X_train,
        split_data.X_test,
        split_data.y_train,
        split_data.y_test,
        pos_label,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_file = output_path / "model_pipeline.joblib"
    summary_file = output_path / "training_summary.json"

    joblib.dump(pipeline, model_file)

    training_summary = {
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_path": str(data_path),
        "model_name": FINAL_MODEL_NAME,
        "target_positive_label": pos_label,
        "train_shape": list(split_data.X_train.shape),
        "test_shape": list(split_data.X_test.shape),
        "features": split_data.X_train.columns.tolist(),
        "metrics": metrics,
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    print("\nTraining completed successfully.")
    print(f"Model saved to: {model_file}")
    print(f"Summary saved to: {summary_file}")
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"- {k}: {v}")

    return model_file, summary_file


def main():
    parser = argparse.ArgumentParser(
        description="Train finalized shipment delay model (Logistic Regression - Baseline) and save artifacts."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="Training Data.csv",
        help="Path to input training CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save model",
    )
    args = parser.parse_args()

    train_and_save(
        data_path=args.data_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
