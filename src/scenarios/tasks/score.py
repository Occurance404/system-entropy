import argparse
import pandas as pd
import sys
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_ground_truth(task_path):
    # In a real setup, this would load from a hidden file not accessible to the agent easily
    # For simulation, we just look for ground_truth.csv in the root of the mounted data
    truth_path = os.path.join(task_path, "ground_truth.csv")
    if not os.path.exists(truth_path):
        print(f"ERROR: Oracle could not find ground truth at {truth_path}")
        sys.exit(1)
    return pd.read_csv(truth_path)

def score_submission(submission_path, task_path, metric="accuracy"):
    try:
        pred_df = pd.read_csv(submission_path)
        truth_df = load_ground_truth(task_path)
        
        # Basic validation
        if len(pred_df) != len(truth_df):
            print(f"ERROR: Prediction count ({len(pred_df)}) does not match Ground Truth ({len(truth_df)}).")
            return 0.0
            
        # Align by ID if present, otherwise assume order
        if "id" in pred_df.columns and "id" in truth_df.columns:
            pred_df = pred_df.sort_values("id").reset_index(drop=True)
            truth_df = truth_df.sort_values("id").reset_index(drop=True)
            
        y_true = truth_df["label"]
        y_pred = pred_df["label"]
        
        if metric == "accuracy":
            score = accuracy_score(y_true, y_pred)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, average='macro', zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, average='macro', zero_division=0)
        elif metric == "f1":
            score = f1_score(y_true, y_pred, average='macro', zero_division=0)
        else:
            print(f"ERROR: Unknown metric {metric}")
            return 0.0
            
        return score

    except Exception as e:
        print(f"ERROR: Scoring failed: {e}")
        return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True, help="Path to predictions.csv")
    parser.add_argument("--metric", default="accuracy", help="Metric to optimize")
    
    # We assume this script runs INSIDE the container at /workspace
    # and the data is at /workspace
    args = parser.parse_args()
    
    final_score = score_submission(args.submission, "/workspace", args.metric)
    print(f"SCORE: {final_score:.4f}")
