import os
import json
import numpy as np
import pandas as pd

# ==========================================
# ROOT CONFIG
# ==========================================
ROOT_DIR = r"D:\liver\CViMXAI"
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

DATASETS = ["CXRay", "Covid19_Ultrasound", "Lung_XRay"]
NUM_FOLDS = 5

# ==========================================
# HELPERS
# ==========================================
def load_fold_metrics(dataset_name):
    fold_metrics = []

    for fold in range(1, NUM_FOLDS + 1):
        metrics_path = os.path.join(
            RESULTS_DIR,
            dataset_name,
            f"fold_{fold}",
            "metrics.json"
        )

        if not os.path.exists(metrics_path):
            print(f"Missing file: {metrics_path}")
            continue

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        fold_metrics.append({
            "Dataset": dataset_name,
            "Fold": fold,
            "Accuracy": metrics.get("accuracy", None),
            "Precision": metrics.get("precision", None),
            "Recall": metrics.get("recall", None),
            "F1-score": metrics.get("f1_score", None)
        })

    return fold_metrics


def compute_summary(df, dataset_name):
    sub = df[df["Dataset"] == dataset_name]

    summary = {
        "Dataset": dataset_name,
        "Accuracy_mean": sub["Accuracy"].mean(),
        "Accuracy_std": sub["Accuracy"].std(),
        "Precision_mean": sub["Precision"].mean(),
        "Precision_std": sub["Precision"].std(),
        "Recall_mean": sub["Recall"].mean(),
        "Recall_std": sub["Recall"].std(),
        "F1_mean": sub["F1-score"].mean(),
        "F1_std": sub["F1-score"].std(),
    }

    return summary


def format_pm(mean_val, std_val):
    return f"{mean_val*100:.2f} ± {std_val*100:.2f}"


# ==========================================
# MAIN
# ==========================================
def main():
    all_fold_metrics = []

    for dataset in DATASETS:
        fold_metrics = load_fold_metrics(dataset)
        all_fold_metrics.extend(fold_metrics)

    if len(all_fold_metrics) == 0:
        print("No metrics files found.")
        return

    df = pd.DataFrame(all_fold_metrics)

    # Save fold-wise metrics
    fold_csv_path = os.path.join(RESULTS_DIR, "all_fold_metrics.csv")
    df.to_csv(fold_csv_path, index=False)

    summary_rows = []
    for dataset in DATASETS:
        if dataset not in df["Dataset"].unique():
            continue
        summary_rows.append(compute_summary(df, dataset))

    summary_df = pd.DataFrame(summary_rows)

    # Save raw summary
    summary_csv_path = os.path.join(RESULTS_DIR, "summary_metrics.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    # Save JSON summary
    summary_json_path = os.path.join(RESULTS_DIR, "summary_metrics.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_rows, f, indent=4)

    # Print fold-wise results
    print("\n================ Fold-wise Metrics ================\n")
    print(df.to_string(index=False))

    # Print paper-ready summary
    print("\n================ Mean ± Std (Paper Ready) ================\n")
    for _, row in summary_df.iterrows():
        print(f"Dataset: {row['Dataset']}")
        print(f"  Accuracy : {format_pm(row['Accuracy_mean'], row['Accuracy_std'])}")
        print(f"  Precision: {format_pm(row['Precision_mean'], row['Precision_std'])}")
        print(f"  Recall   : {format_pm(row['Recall_mean'], row['Recall_std'])}")
        print(f"  F1-score : {format_pm(row['F1_mean'], row['F1_std'])}")
        print("-" * 55)

    # Create final paper-ready table
    paper_table = []
    for _, row in summary_df.iterrows():
        paper_table.append({
            "Dataset": row["Dataset"],
            "Accuracy (%)": format_pm(row["Accuracy_mean"], row["Accuracy_std"]),
            "Precision (%)": format_pm(row["Precision_mean"], row["Precision_std"]),
            "Recall (%)": format_pm(row["Recall_mean"], row["Recall_std"]),
            "F1-score (%)": format_pm(row["F1_mean"], row["F1_std"])
        })

    paper_df = pd.DataFrame(paper_table)

    paper_csv_path = os.path.join(RESULTS_DIR, "paper_ready_metrics_table.csv")
    paper_df.to_csv(paper_csv_path, index=False)

    print("\n================ Final Table for Paper ================\n")
    print(paper_df.to_string(index=False))

    print("\nSaved files:")
    print(f"1. Fold-wise metrics: {fold_csv_path}")
    print(f"2. Summary metrics:   {summary_csv_path}")
    print(f"3. JSON summary:      {summary_json_path}")
    print(f"4. Paper-ready table: {paper_csv_path}")


if __name__ == "__main__":
    main()