import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from typing import List, Dict


class Evaluator:
    @staticmethod
    def compute_metrics(y_true: List[int], y_pred: List[int], y_prob: List[float]) -> Dict:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0, pos_label=1)
        recall = recall_score(y_true, y_pred, zero_division=0, pos_label=1)
        f1 = f1_score(y_true, y_pred, zero_division=0, pos_label=1)
        f1_macro = f1_score(y_true, y_pred, zero_division=0, average='macro')

        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = 0.5  # Fallback

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "f1_macro": round(f1_macro, 4),
            "auc": round(auc, 4),
            "confusion_matrix": {
                "TP": int(tp), "FP": int(fp),
                "FN": int(fn), "TN": int(tn)
            },
            "support": {
                "positive": int(sum(y_true)),
                "negative": int(len(y_true) - sum(y_true))
            }
        }

    @staticmethod
    def print_metrics(metrics: Dict, experiment_name: str = ""):
        """Print a metrics summary."""
        print(f"\n{'=' * 60}")
        print(f"Experiment: {experiment_name}")
        print(f"{'=' * 60}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        print(f"AUC:       {metrics['auc']:.4f}")
        print(f"\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"  TP: {cm['TP']:3d}  |  FP: {cm['FP']:3d}")
        print(f"  FN: {cm['FN']:3d}  |  TN: {cm['TN']:3d}")
        print(f"\nSupport: Positive={metrics['support']['positive']}, Negative={metrics['support']['negative']}")
        print(f"{'=' * 60}\n")