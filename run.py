"""Main experiment script for Project 1."""

from pathlib import Path

import numpy as np

from helpers import create_csv_submission, load_csv_data
from implementations import (
    least_squares,
    logistic_regression,
    mean_squared_error_gd,
    mean_squared_error_sgd,
    reg_logistic_regression,
    ridge_regression,
)


def train_model(y_train, x_train, method="least_squares"):
    """Call one of the methods from implementations.py."""
    initial_w = np.zeros(x_train.shape[1])
    max_iters = 1000
    gamma = 0.01
    lambda_ = 0.01

    if method == "mean_squared_error_gd":
        return mean_squared_error_gd(y_train, x_train, initial_w, max_iters, gamma)
    if method == "mean_squared_error_sgd":
        return mean_squared_error_sgd(y_train, x_train, initial_w, max_iters, gamma)
    if method == "least_squares":
        return least_squares(y_train, x_train)
    if method == "ridge_regression":
        return ridge_regression(y_train, x_train, lambda_)
    if method == "logistic_regression":
        return logistic_regression(y_train, x_train, initial_w, max_iters, gamma)
    if method == "reg_logistic_regression":
        return reg_logistic_regression(
            y_train, x_train, lambda_, initial_w, max_iters, gamma
        )
    raise ValueError(f"Unknown method: {method}")


def make_predictions(x_data, weights):
    """Convert model scores into the required -1/1 predictions."""
    scores = x_data @ weights
    return np.where(scores >= 0, 1, -1)


def plot_results(results, output_path):
    """Save experiment results as a plot once results are available."""
    import matplotlib.pyplot as plt

    methods = list(results)
    losses = [results[method] for method in methods]
    plt.figure()
    plt.bar(methods, losses)
    plt.ylabel("Loss")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    """Load data, train a model, generate predictions, and save results."""
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    output_dir = project_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    x_train, x_test, y_train, _, test_ids = load_csv_data(data_dir)
    weights, loss = train_model(y_train, x_train, method="least_squares")
    predictions = make_predictions(x_test, weights)
    create_csv_submission(test_ids, predictions, output_dir / "submission.csv")
    plot_results({"least_squares": loss}, output_dir / "losses.png")


if __name__ == "__main__":
    main()
