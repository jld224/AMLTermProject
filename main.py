import subprocess
import os
import pandas as pd
from src.training.ExperimentRunner import ExperimentRunner

from EXP_CONFIG import EXP_CONFIG

def run_script(script_path):
    try:
        # Execute the script
        completed_process = subprocess.run(
            ["python", script_path], check=True, text=True, capture_output=True
        )

        # Improved output formatting
        if completed_process.stdout:
            print("=== Process Output ===")
            print(completed_process.stdout.strip())
        if completed_process.stderr:
            print("=== Process Errors ===")
            print(completed_process.stderr.strip())

    except subprocess.CalledProcessError as e:
        print(f"Error running script {script_path}: {e}\n{e.stderr}")


def train_and_evaluate(model_name):
    """Run the training and evaluation script for a given model."""
    print(f"Training and evaluating {model_name} model...")
    subprocess.run(
        ["python", "-m", "scripts.train_and_evaluate", "--model", model_name],
        check=True,
    )
    print(f"{model_name} model training and evaluation completed.")


def run_training(
    source_csv="./data/processed.csv",
    feature_selection_strategies=["Base", "RFE"],
    save_folder=None,
):

    runner = ExperimentRunner(EXP_CONFIG)

    data = pd.read_csv(source_csv).dropna(subset=["η / mPa s"])
    X, y = data.drop("η / mPa s", axis=1), data["η / mPa s"].values
    runner.run_cross_experiment(X, y, feature_selection_strategies, n_splits=5)


if __name__ == "__main__":
    # run_script("./src/preprocessing/preprocess.py")

    feature_selection_strategies = [
        {"name": "Base"},
        {"name": "SelectKBest"},
        {"name": "PCA"},
        {"name": "GiniImportance"}
    ]

    run_training("./data/processed.csv", feature_selection_strategies, "./results/")

    run_script("./src/evaluation/plots.py")
