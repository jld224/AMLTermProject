import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2, venn3
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    r2_score,
)
from functools import wraps


def plot_wrapper(figsize=(8, 6), xlabel="", ylabel="", scale=None, directory="images", filename="image.svg"):
    def decorator(plot_func):
        def wrapper(*args, **kwargs):
            plt.figure(figsize=figsize)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if scale:
                plt.yscale(scale)
                plt.xscale(scale)
            plot_func(*args, **kwargs)  # args will include data and potentially other required information
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(os.path.join(directory, filename), format="svg")
            plt.close()
        return wrapper
    return decorator


def calculate_errors(data, techniques):
    """Calculate absolute errors for each feature selection technique."""
    for technique in techniques:
        predicted_column = f"{technique}_Predicted"
        data[f"{technique}_Error"] = np.abs(data["Actual"] - data[predicted_column])
    return data


def confusion_matrices(data, techniques):
    data["True Cluster"] = determine_cluster(data["Actual"]).astype(int)

    for technique in techniques:
        plot_confusion_matrix(data, technique)


def determine_cluster(values, num_clusters=5):
    """
    This divides a dataset into X clusters and returns the values corresponding to the values passed in.
    Done via quantile bins ranging from 0 - X.

    Args:
        values (pd.Series): Listing of values to bin.
        num_clusters (int, optional): Number of bins to divide into. Defaults to 5.

    Returns:
        np.array: Resultant corresponding list of cluster identifications.
    """
    if values.isnull().any():
        raise ValueError("NaN values present in input data for clustering.")
    if values.nunique() < num_clusters:
        raise ValueError("Not enough unique values to form distinct clusters.")

    quantiles = np.linspace(0, 1, num_clusters + 1)
    bin_edges = np.quantile(values, quantiles)
    clusters = np.digitize(values, bin_edges, right=False) - 1
    return clusters


@plot_wrapper(filename="confusion_matrix.svg", xlabel="Predicted Labels", ylabel="True Labels")
def plot_confusion_matrix(data, technique):
    if not data.empty and f'{technique}_Predicted' in data.columns and 'Actual' in data.columns:
        predicted_column = f"{technique}_Predicted"
        true_labels = data['Actual']
        predicted_labels = data[predicted_column]
        cm = confusion_matrix(true_labels, predicted_labels)
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
        plt.title(f'Confusion Matrix for {technique}')
    else:
        print(f"Data is empty or required columns are missing for {technique}.")

@plot_wrapper(filename="venn.svg")
def plot_feature_correspondance(data, techniques):
    """
    Plots correspondance between data features chosen by X techniques.
    This is done via Venn Diagram and is limited to 2-3 techniques/

    Args:
        data (pd.DataFrame): Input DF that details which features where chosen by what feature selection technique.
        techniques (List): Listing of the different techniques to iterate through.
    """
    data = data == True

    assert (
        1 < len(techniques) <= 3
    ), "This function supports 2 or 3 techniques for Venn diagrams."

    # Initialize selections and subsets for Venn diagram
    selections = {tech: data[tech].sum() for tech in techniques}
    subsets = [selections[tech] for tech in techniques]
    intersections = {}

    # Calculate intersections
    for i, tech1 in enumerate(techniques):
        for j, tech2 in enumerate(techniques[i + 1 :], i + 1):
            intersection_key = f"{tech1}_{tech2}"
            intersections[intersection_key] = data[data[tech1] & data[tech2]].shape[0]

            if len(techniques) == 3 and j < len(techniques) - 1:
                for k, tech3 in enumerate(techniques[j + 1 :], j + 1):
                    all_intersection_key = f"{tech1}_{tech2}_{tech3}"
                    intersections[all_intersection_key] = data[
                        data[tech1] & data[tech2] & data[tech3]
                    ].shape[0]

    # For 2 techniques, adjust subsets list directly
    if len(techniques) == 2:
        subsets.append(
            intersections[next(iter(intersections))]
        )  # Only intersection for 2 techniques
        venn_diagram = venn2(subsets=subsets, set_labels=techniques)
    # For 3 techniques, create subsets list based on the order required by venn3
    elif len(techniques) == 3:
        tech1, tech2, tech3 = techniques
        subsets = [
            selections[tech1],
            selections[tech2],
            intersections[f"{tech1}_{tech2}"],
            selections[tech3],
            intersections[f"{tech1}_{tech3}"],
            intersections[f"{tech2}_{tech3}"],
            intersections[f"{tech1}_{tech2}_{tech3}"],
        ]
        venn_diagram = venn3(subsets=subsets, set_labels=techniques)

    # Adjust font sizes for readability
    for text in venn_diagram.set_labels:
        if text:
            text.set_fontsize(14)
    for text in venn_diagram.subset_labels:
        if text:
            text.set_fontsize(12)


@plot_wrapper(ylabel="Mean Absolute Error", filename="MAE_Comparison_Techniques.svg")
def plot_mae(data, techniques):
    """Compare the mean absolute error of each technique with a bar chart."""

    mae_values = [data[f"{technique}_Error"].mean() for technique in techniques]
    plt.bar(techniques, mae_values)


@plot_wrapper(
    xlabel="Actual",
    ylabel="Predicted",
    scale="log",
    filename="Combined_Actual_vs_Predicted_R2.svg",
)
def plot_scatter(data, techniques):
    """Generate a combined scatter plot for actual vs. predicted values for each technique, including R^2 annotations."""

    for technique in techniques:
        predicted_column = f"{technique}_Predicted"
        r2 = r2_score(data["Actual"], data[predicted_column])  # Compute R^2
        plt.scatter(
            data["Actual"],
            data[predicted_column],
            alpha=0.5,
            label=f"{technique} (R² = {r2:.2f})",
        )

    # Plotting the identity line in a more compact form
    plt.plot(
        [min(data["Actual"]), max(data["Actual"])],
        [min(data["Actual"]), max(data["Actual"])],
        "k--",
        lw=2,
    )
    plt.legend(loc="best")  # Show legend to identify each technique

def read_data(file_path, sheet_name):
    try:
        return pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
    except Exception as e:
        print(f"Error reading {sheet_name} from {file_path}: {str(e)}")
        # Create an empty DataFrame with expected columns if necessary for the plots
        return pd.DataFrame(columns=['True Labels', 'Predicted Labels'])

def main():
    try:
        file_path = "./results.xlsx"
        pred_data = read_data(file_path, "Results")
        feature_data = read_data(file_path, "Selected_Features")

        if pred_data.empty:
            print("No data in 'Results' sheet, generating default plot with dummy data.")
            pred_data = pd.DataFrame({'Actual': [0, 1], 'Predicted Labels': [0, 1]})
        
        techniques = ['Technique1', 'Technique2']  # Example technique names
        for technique in techniques:
            plot_confusion_matrix(pred_data, technique)
        
    except FileNotFoundError as fnfe:
        print(f"File not found error: {fnfe}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
