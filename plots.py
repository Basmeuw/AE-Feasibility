import matplotlib.pyplot as plt
import json
import os
import numpy as np


def plot_metrics(train_data, val_data, metric_name, title, save_path=None, show=True):
    """
    General function to plot training and validation metrics over epochs.

    Args:
        train_data (list): List of training metric values (e.g., loss or accuracy).
        val_data (list): List of validation metric values.
        metric_name (str): Name of the metric (e.g., 'Loss', 'Accuracy').
        title (str): Title of the plot.
        save_path (str, optional): Path to save the plot. If None, it just displays.
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_data) + 1)

    # Plot Training Data
    plt.plot(epochs, train_data, 'b-o', label=f'Training {metric_name}', linewidth=2)
    # Plot Validation Data
    plt.plot(epochs, val_data, 'r-o', label=f'Validation {metric_name}', linewidth=2)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(epochs)  # Ensure all epochs are shown on the x-axis

    if save_path:
        plt.savefig(save_path)
        print(f"\nPlot saved to {save_path}")
    if show:
        plt.show()

# --- Helper Function: Data Loading (Kept separate as requested) ---

def load_run_data(parent_folder):
    """
    Helper function to iterate through run folders and load necessary data.

    Returns:
        list: A list of dictionaries, where each dictionary contains
              'title', 'val_accuracies', and 'final_test_accuracy' for a run.
    """
    run_data = []

    for run_folder_name in os.listdir(parent_folder):
        run_path = os.path.join(parent_folder, run_folder_name)

        if os.path.isdir(run_path):
            params_file = os.path.join(run_path, 'params.json')
            results_file = os.path.join(run_path, 'results.json')

            if os.path.exists(params_file) and os.path.exists(results_file):
                try:
                    with open(params_file, 'r') as f:
                        params = json.load(f)
                        run_title = params.get('title', run_folder_name)

                    with open(results_file, 'r') as f:
                        results = json.load(f)
                        val_accuracies = results.get('val_accuracies', [])
                        test_accuracy = results.get('final_test_accuracy')

                    if val_accuracies or test_accuracy is not None:
                        run_data.append({
                            'title': run_title,
                            'val_accuracies': val_accuracies,
                            'final_test_accuracy': test_accuracy
                        })

                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON in files for run: {run_folder_name}")
                except Exception as e:
                    print(f"An error occurred while processing run {run_folder_name}: {e}")

    return run_data


# --- Function 1: Line Plot for Epoch-based Metrics (No Change) ---

def plot_metric_from_runs(parent_folder, metric_name, title, save_path=None, show=True):
    """
    Plots a specified epoch-based metric for multiple runs using a line graph.
    """
    # Load all relevant data
    run_data = load_run_data(parent_folder)

    plt.figure(figsize=(12, 7))
    found_data = False
    max_epochs = 0

    for run in run_data:
        metric_data = run.get(metric_name, [])
        run_title = run['title']

        if metric_data:
            found_data = True
            epochs = range(1, len(metric_data) + 1)
            plt.plot(epochs, metric_data, marker='o', linestyle='-', label=run_title, linewidth=2)
            max_epochs = max(max_epochs, len(epochs))

    if found_data:
        plt.title(title, fontsize=18, fontweight='bold')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel(metric_name.replace('_', ' ').title(), fontsize=14)
        plt.legend(fontsize=12, loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)

        if max_epochs > 0:
            plt.xticks(range(1, max_epochs + 1))

        if save_path:
            plt.savefig(save_path)
            print(f"\nPlot saved to {save_path}")
        if show:
            plt.show()
    else:
        print(f"No metric data ('{metric_name}') found in the folder: {parent_folder}")


# --- Function 2: Bar Plot for Single Final Metric (Sorting Added)  ---

def plot_final_metric_bar_chart(parent_folder, final_metric_key, title, save_path=None, show=True):
    """
    Creates a bar chart to compare a single final metric across multiple runs,
    sorted alphabetically by run title.

    Args:
        parent_folder (str): Path to the parent directory containing run subfolders.
        final_metric_key (str): The key in results.json (e.g., 'final_test_accuracy') to plot.
        title (str): Title of the plot.
        save_path (str, optional): Path to save the plot.
        show (bool, optional): Whether to display the plot.
    """
    # Load all relevant data
    run_data = load_run_data(parent_folder)

    # Filter data to only include runs with the final metric
    plot_data = []
    for run in run_data:
        final_metric_value = run.get(final_metric_key)
        if final_metric_value is not None:
            plot_data.append({
                'title': run['title'],
                'value': final_metric_value
            })

    # --- Sorting Step ---
    # Sort the data alphabetically by the 'title' key
    plot_data.sort(key=lambda x: x['title'])

    # Separate sorted lists for plotting
    run_names = [d['title'] for d in plot_data]
    final_metrics = [d['value'] for d in plot_data]

    if final_metrics:
        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(run_names))

        plt.bar(y_pos, final_metrics, align='center', alpha=0.8, color='darkred')

        # Use the now-sorted run_names for the x-ticks
        plt.xticks(y_pos, run_names, rotation=45, ha='right', fontsize=10)

        plt.ylabel(final_metric_key.replace('_', ' ').title(), fontsize=14)
        plt.title(title, fontsize=16, fontweight='bold')

        # Add text labels on top of the bars
        for i, val in enumerate(final_metrics):
            plt.text(i, val * 1.01, f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"\nBar chart saved to {save_path}")
        if show:
            plt.show()
    else:
        print(f"No final metric data ('{final_metric_key}') found in the folder: {parent_folder}")
# plot_metric_from_runs("useful_runs/compression_vs_acc", "val_accuracies", "Compression vs Accuracy Tradeoff", save_path="useful_runs/compression_vs_acc/results_val")
# plot_metric_from_runs("useful_runs/compression_vs_acc", "train_accuracies", "Compression vs Accuracy Tradeoff", save_path="useful_runs/compression_vs_acc/results")
# 1. Plot the epoch-based validation accuracy (Line Plot)
# plot_metric_from_runs(
#     parent_folder="useful_runs/compression_vs_acc_CalTech",
#     metric_name="val_accuracies",
#     title="Caltech Final Test Accuracy",
#     save_path="useful_runs/compression_vs_acc_CalTech/results_val"
# )

# plot_metric_from_runs(
#     parent_folder="useful_runs/compression_vs_acc_cifar10",
#     metric_name="val_losses",
#     title="CIFAR10 Compression vs Val Loss Tradeoff",
#     save_path="useful_runs/compression_vs_acc_cifar10/results_val_loss"
# )

# # 2. Plot the final test accuracy (Bar Chart)
# plot_final_metric_bar_chart(
#     parent_folder="useful_runs/compression_vs_acc_CalTech",
#     final_metric_key="final_test_accuracy",
#     title="CalTech Final Test Accuracy Comparison",
#     save_path="useful_runs/compression_vs_acc_CalTech/final_test_plot"
# )

# 1. Plot the epoch-based validation accuracy (Line Plot)
plot_metric_from_runs(
    parent_folder="useful_runs/compression_vs_acc_TinyImageNet10",
    metric_name="val_accuracies",
    title="TinyImageNet Final Test Accuracy",
    save_path="useful_runs/compression_vs_acc_TinyImageNet10/results_val"
)

# plot_metric_from_runs(
#     parent_folder="useful_runs/compression_vs_acc_cifar10",
#     metric_name="val_losses",
#     title="CIFAR10 Compression vs Val Loss Tradeoff",
#     save_path="useful_runs/compression_vs_acc_cifar10/results_val_loss"
# )

# 2. Plot the final test accuracy (Bar Chart)
plot_final_metric_bar_chart(
    parent_folder="useful_runs/compression_vs_acc_TinyImageNet10",
    final_metric_key="final_test_accuracy",
    title="TinyImageNet Final Test Accuracy Comparison",
    save_path="useful_runs/compression_vs_acc_TinyImageNet10/final_test_plot"
)