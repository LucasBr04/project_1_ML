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
from preprocessing import clean_and_standardize, load_feature_names
from validation import best_threshold, f1_score, split_train_validation


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


def make_predictions(x_data, weights, threshold=0):
    """Convert model scores into the required -1/1 predictions."""
    scores = x_data @ weights
    return np.where(scores >= threshold, 1, -1)


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
    names = load_feature_names(data_dir)

    # 1. estimate the quality of the model: train on 80%, evaluate on 20%
    x_tr, x_val, y_tr, y_val = split_train_validation(y_train, x_train)
    tx_tr, tx_val = clean_and_standardize(x_tr, x_val, names)
    weights, _ = train_model(y_tr, tx_tr, method="least_squares")
    # the threshold is chosen on the training part only, not on the validation
    threshold = best_threshold(y_tr, tx_tr @ weights)
    f1_default = f1_score(y_val, make_predictions(tx_val, weights))
    f1 = f1_score(y_val, make_predictions(tx_val, weights, threshold))
    print(f"validation F1 = {f1_default:.4f} with threshold 0")
    print(f"validation F1 = {f1:.4f} with threshold {threshold:.3f}")

    # 2. final model: train on all the training data, predict the test set
    tx_train, tx_test = clean_and_standardize(x_train, x_test, names)
    weights, loss = train_model(y_train, tx_train, method="least_squares")
    threshold = best_threshold(y_train, tx_train @ weights)
    predictions = make_predictions(tx_test, weights, threshold)
    print(
        f"final threshold = {threshold:.3f}, {np.mean(predictions == 1):.1%} of the test set predicted sick"
    )
    create_csv_submission(test_ids, predictions, output_dir / "submission.csv")
    plot_results({"least_squares": loss}, output_dir / "losses.png")


if __name__ == "__main__":
    main()
