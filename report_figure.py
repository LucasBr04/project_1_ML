"""Figures of the report: python report_figure.py

outputs/report_threshold.png: precision, recall and F1 of the final logistic
    model versus the decision threshold, on the validation set (same setting
    as run.ipynb, section 4.1).
outputs/report_learning_curve.png: SGD learning curve, F1 on the first n
    training observations and on the validation set (same setting as
    sgd_tests.ipynb, test 5).
Both use the common 80/20 split (seed 1) and the one-hot preprocessing.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from helpers import load_csv_data
from implementations import mean_squared_error_sgd, reg_logistic_regression
from preprocessing import clean_and_standardize, load_feature_names
from validation import best_threshold, f1_score, split_train_validation


def precision_recall_f1(y, pred):
    """Precision, recall and F1 of the positive class; labels in {-1, 1}."""
    tp = np.sum((pred == 1) & (y == 1))
    precision = tp / max(np.sum(pred == 1), 1)
    recall = tp / np.sum(y == 1)
    return precision, recall, f1_score(y, pred)


def main():
    x, _, y, _, _ = load_csv_data("data")
    names = load_feature_names("data")
    x_tr, x_va, y_tr, y_va = split_train_validation(y, x)
    tx_tr, tx_va = clean_and_standardize(x_tr, x_va, names, one_hot=True)

    # (a) final logistic model (lambda = 0, gamma = 0.5, 500 iterations)
    w, _ = reg_logistic_regression(
        (y_tr == 1).astype(float), tx_tr, 0, np.zeros(tx_tr.shape[1]), 500, 0.5
    )
    scores = tx_va @ w
    chosen = best_threshold(y_va, scores)
    thresholds = np.quantile(scores, np.linspace(0.3, 0.995, 120))
    curves = np.array(
        [precision_recall_f1(y_va, np.where(scores >= t, 1, -1)) for t in thresholds]
    )

    # (b) SGD learning curve (gamma = 3e-5, 300,000 steps, seed 1)
    sizes = [1_000, 3_000, 10_000, 30_000, 100_000, len(y_tr)]
    f1_train, f1_val = [], []
    for n in sizes:
        tx_n, y_n = tx_tr[:n], y_tr[:n]  # rows already shuffled by the split
        np.random.seed(1)
        w_sgd, _ = mean_squared_error_sgd(
            y_n, tx_n, np.zeros(tx_n.shape[1]), 300_000, 3e-5
        )
        t = best_threshold(y_n, tx_n @ w_sgd)
        f1_train.append(f1_score(y_n, np.where(tx_n @ w_sgd >= t, 1, -1)))
        f1_val.append(f1_score(y_va, np.where(tx_va @ w_sgd >= t, 1, -1)))
        print(
            f"n = {n:>7}: F1 train {f1_train[-1]:.4f}, F1 validation {f1_val[-1]:.4f}"
        )

    # two separate one-column figures, so that they can be placed independently
    plt.rcParams.update({"font.size": 8})
    os.makedirs("outputs", exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.45, 2.2), dpi=300)
    ax.plot(thresholds, curves[:, 0], label="Precision")
    ax.plot(thresholds, curves[:, 1], label="Recall")
    ax.plot(thresholds, curves[:, 2], label="F1", lw=2)
    ax.axvline(chosen, color="black", ls="--", lw=0.9, label="Chosen threshold")
    ax.set_xlabel(r"Threshold $t$ on the score $x^\top w$")
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(alpha=0.3)
    fig.tight_layout(pad=0.3)
    fig.savefig("outputs/report_threshold.png")

    fig, ax = plt.subplots(figsize=(3.45, 2.2), dpi=300)
    ax.plot(sizes, f1_train, marker="o", ms=3, label="Training subset")
    ax.plot(sizes, f1_val, marker="s", ms=3, label="Validation set")
    ax.set_xscale("log")
    ax.set_xlabel("Number of training observations $n$")
    ax.set_ylabel("F1")
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(alpha=0.3)
    fig.tight_layout(pad=0.3)
    fig.savefig("outputs/report_learning_curve.png")
    print(f"chosen threshold {chosen:.3f}; saved the two figures in outputs/")


if __name__ == "__main__":
    main()
