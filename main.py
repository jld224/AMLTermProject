import subprocess
import os


def run_preprocessing():
    """Run the data preprocessing script."""
    print("Running preprocessing script...")
    subprocess.run(["python", "./scripts/run_preprocessing.py"], check=True)
    print("Preprocessing script completed.")


def train_and_evaluate(model_name):
    """Run the training and evaluation script for a given model."""
    print(f"Training and evaluating {model_name} model...")
    subprocess.run(
        ["python", "-m", "scripts.train_and_evaluate", "--model", model_name],
        check=True,
    )
    print(f"{model_name} model training and evaluation completed.")

def run_plotting():
    """Run the data plotting script."""
    print("Running plotting script...")
    subprocess.run(["python", "./scripts/run_plotting.py"], check=True)
    print("Plotting script completed.")

if __name__ == "__main__":
    # run_preprocessing()

    models = ["NN"]
    for model in models:
        train_and_evaluate(model)
        
    run_plotting()

