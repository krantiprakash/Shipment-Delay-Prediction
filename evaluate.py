from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_preprocessing import get_training_objects


def get_pos_label(y_train, y_test):
    classes = sorted(list(set(y_train).union(set(y_test))))
    return "Yes" if "Yes" in classes else classes[1]


def compute_metrics(model, X_test, y_test, pos_label: str) -> Dict[str, float]:
    y_pred = model.predict(X_test)

    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, pos_label=pos_label)),
        "test_recall": float(recall_score(y_test, y_pred, pos_label=pos_label)),
        "test_f1": float(f1_score(y_test, y_pred, pos_label=pos_label)),
    }

    metrics["test_roc_auc"] = np.nan
    if hasattr(model, "predict_proba"):
        class_order = list(model.classes_)
        pos_idx = class_order.index(pos_label)
        y_prob = model.predict_proba(X_test)[:, pos_idx]
        metrics["test_roc_auc"] = float(roc_auc_score(y_test, y_prob))

    return metrics


def evaluate(
    data_path: str,
    model_path: str,
    output_dir: str,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Recreate the same split strategy used in training
    split_data, _ = get_training_objects(data_path)
    X_test = split_data.X_test
    y_test = split_data.y_test
    y_train = split_data.y_train

    model = joblib.load(model_path)
    pos_label = get_pos_label(y_train, y_test)

    metrics = compute_metrics(model, X_test, y_test, pos_label=pos_label)
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        class_order = list(model.classes_)
        pos_idx = class_order.index(pos_label)
        y_prob = model.predict_proba(X_test)[:, pos_idx]

    classes = sorted(list(set(y_test)))
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    # Save confusion matrix figure
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = output_path / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path, dpi=140)
    plt.close()

    # Save predictions sample
    eval_df = X_test.copy()
    eval_df["Actual_Delayed"] = y_test
    eval_df["Predicted_Delayed"] = y_pred
    if y_prob is not None:
        eval_df["Delay_Probability"] = y_prob
    eval_df["Is_Correct"] = eval_df["Actual_Delayed"] == eval_df["Predicted_Delayed"]
    preds_path = output_path / "test_predictions.csv"
    eval_df.to_csv(preds_path, index=True)

    # Class-wise correctness summary
    class_rows = []
    for cls in classes:
        total_actual = int((eval_df["Actual_Delayed"] == cls).sum())
        correct_pred = int(
            (
                (eval_df["Actual_Delayed"] == cls)
                & (eval_df["Predicted_Delayed"] == cls)
            ).sum()
        )
        incorrect_pred = total_actual - correct_pred
        recall_pct = round((correct_pred / total_actual) * 100, 2) if total_actual else 0.0
        class_rows.append(
            {
                "Class": cls,
                "Total_Actual": total_actual,
                "Correctly_Predicted": correct_pred,
                "Misclassified": incorrect_pred,
                "Correctness_Percentage": recall_pct,
            }
        )
    class_summary_df = pd.DataFrame(class_rows)
    class_summary_path = output_path / "class_wise_summary.csv"
    class_summary_df.to_csv(class_summary_path, index=False)

    # Misclassified rows for deeper analysis
    misclassified_df = eval_df[~eval_df["Is_Correct"]].copy()
    misclassified_path = output_path / "misclassified_rows.csv"
    misclassified_df.to_csv(misclassified_path, index=True)

    # Save JSON summary
    evaluation_summary = {
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_path": data_path,
        "model_path": model_path,
        "positive_label": pos_label,
        "metrics": metrics,
        "classification_report": report_dict,
        "artifacts": {
            "confusion_matrix_png": str(cm_path),
            "test_predictions_csv": str(preds_path),
            "class_wise_summary_csv": str(class_summary_path),
            "misclassified_rows_csv": str(misclassified_path),
        },
    }
    summary_path = output_path / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, indent=2)

    # Console output
    print("\nEvaluation completed successfully.")
    print(f"Model: {model_path}")
    print(f"Output directory: {output_dir}")
    print("\nTest Metrics:")
    for k, v in metrics.items():
        print(f"- {k}: {v}")

    print("\nClassification Report (Test):")
    print(report_text)
    print(f"Confusion matrix saved to: {cm_path}")
    print(f"Predictions CSV saved to: {preds_path}")
    print(f"Class-wise summary CSV saved to: {class_summary_path}")
    print(f"Misclassified rows CSV saved to: {misclassified_path}")
    print(f"Evaluation summary JSON saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained shipment delay model on test split.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="Training Data.csv",
        help="Path to input data CSV",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/model_pipeline.joblib",
        help="Path to saved model pipeline",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation artifacts",
    )
    args = parser.parse_args()

    evaluate(
        data_path=args.data_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
