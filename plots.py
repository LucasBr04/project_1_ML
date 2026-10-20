"""Figures for the report: convergence, cross validation and class separation."""

import matplotlib.pyplot as plt
import numpy as np

from implementations import logistic_gradient, logistic_loss
from validation import f1_score


def _finish(path, title):
    """Add the title, save the figure if a path is given, and show it."""
    plt.title(title)
    if path is not None:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_convergence(y, tx, lambda_, gamma, max_iters, every=10, path=None):
    """Plot the training log-loss along the iterations of gradient descent.

    Same update as reg_logistic_regression, but the loss is recorded every
    `every` iterations to check that the descent has converged.

    Args:
        y: shape=(N, ), labels in {-1, 1}
        tx: shape=(N, D), cleaned data with the bias column
        lambda_: scalar, the regularization parameter
        gamma: step size of gradient descent
        max_iters: number of iterations
        every: the loss is computed every `every` iterations (it is costly)
        path: where to save the figure (None: not saved)
    """
    y01 = (y == 1).astype(float)
    w = np.zeros(tx.shape[1])
    iters, losses = [], []
    for it in range(1, max_iters + 1):
        w = w - gamma * (logistic_gradient(y01, tx, w) + 2 * lambda_ * w)
        if it % every == 0:
            iters.append(it)
            losses.append(logistic_loss(y01, tx, w))

    plt.figure()
    plt.plot(iters, losses)
    plt.xlabel("itération")
    plt.ylabel("log-loss d'entraînement")
    _finish(path, f"Convergence de la descente de gradient (gamma = {gamma})")


def plot_cv_lambda(results, path=None):
    """Plot the mean F1 (± std over the folds) for each lambda.

    Args:
        results: list of (mean_f1, std_f1, lambda_, threshold), as built in
            the cross validation step of the notebook
        path: where to save the figure (None: not saved)
    """
    results = sorted(results, key=lambda r: r[2])
    lambdas = np.array([r[2] for r in results])
    means = [r[0] for r in results]
    stds = [r[1] for r in results]

    # lambda = 0 cannot be drawn on a log scale: put it one decade below the
    # smallest positive lambda
    positive = lambdas[lambdas > 0]
    zero_pos = positive.min() / 10 if len(positive) else 1e-6
    x = np.where(lambdas > 0, lambdas, zero_pos)

    plt.figure()
    plt.errorbar(x, means, yerr=stds, marker="o", capsize=4)
    plt.xscale("log")
    if (lambdas == 0).any():
        plt.annotate("λ = 0", (zero_pos, means[0]), textcoords="offset points", xytext=(5, 8))
    plt.xlabel("lambda")
    plt.ylabel("F1 moyen sur les plis de validation")
    _finish(path, "Cross validation de la régularisation L2")


def plot_score_distribution(y, scores, threshold, bins=100, path=None):
    """Plot the histograms of the scores x^T w of each class.

    The vertical line is the decision threshold: everyone on its right is
    predicted sick. It is the 1D equivalent of the decision boundary.

    Args:
        y: shape=(N, ), labels in {-1, 1}
        scores: shape=(N, ), tx @ w
        threshold: the decision threshold
        bins: number of bins of the histograms
        path: where to save the figure (None: not saved)
    """
    plt.figure()
    plt.hist(scores[y == -1], bins=bins, alpha=0.6, density=True, label="sains")
    plt.hist(scores[y == 1], bins=bins, alpha=0.6, density=True, label="malades")
    plt.axvline(threshold, color="k", linestyle="--", label=f"seuil = {threshold:.2f}")
    plt.xlabel("score $x^\\top w$")
    plt.ylabel("densité")
    plt.legend()
    _finish(path, "Séparation des classes sur la validation")


def plot_threshold_curves(y, scores, threshold=None, path=None):
    """Plot the F1, the precision and the recall as functions of the threshold.

    Args:
        y: shape=(N, ), labels in {-1, 1}
        scores: shape=(N, ), tx @ w
        threshold: the chosen threshold, drawn as a vertical line (optional)
        path: where to save the figure (None: not saved)
    """
    candidates = np.quantile(scores, np.linspace(0.50, 0.99, 50))
    f1s, precisions, recalls = [], [], []
    n_sick = np.sum(y == 1)
    for t in candidates:
        pred = np.where(scores >= t, 1, -1)
        tp = np.sum((pred == 1) & (y == 1))
        precisions.append(tp / max(np.sum(pred == 1), 1))
        recalls.append(tp / n_sick)
        f1s.append(f1_score(y, pred))

    plt.figure()
    plt.plot(candidates, f1s, label="F1")
    plt.plot(candidates, precisions, label="précision")
    plt.plot(candidates, recalls, label="rappel")
    if threshold is not None:
        plt.axvline(threshold, color="k", linestyle="--", label="seuil choisi")
    plt.xlabel("seuil sur $x^\\top w$")
    plt.legend()
    _finish(path, "Compromis précision / rappel")