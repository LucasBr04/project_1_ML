"""Evaluation of the models: F1 score and train/validation split."""

import numpy as np


def f1_score(y_true, y_pred):
    """Compute the F1 score of the positive class (label 1).

    F1 = 2 * precision * recall / (precision + recall), where
    precision = among the people predicted sick, the fraction really sick
    recall = among the people really sick, the fraction we found

    Args:
        y_true: shape=(N, ), true labels in {-1, 1}
        y_pred: shape=(N, ), predicted labels in {-1, 1}

    Returns:
        a scalar in [0, 1] (0 if no sick person is correctly found)
    """
    tp = np.sum((y_pred == 1) & (y_true == 1))  # sick, predicted sick
    fp = np.sum((y_pred == 1) & (y_true == -1))  # healthy, predicted sick
    fn = np.sum((y_pred == -1) & (y_true == 1))  # sick, predicted healthy
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def best_threshold(y, scores):
    """Find the decision threshold that gives the best F1 score.

    A person is predicted sick when its score is >= threshold. Since only ~9%
    of the people are sick, the best threshold is much lower than the default 0.

    Args:
        y: shape=(N, ), true labels in {-1, 1}
        scores: shape=(N, ), outputs of the model (tx @ w)

    Returns:
        the threshold (scalar) with the highest F1 on these data
    """
    # candidate thresholds: the values below which 50%, 51%, ..., 99% of the
    # scores fall, so that the candidates adapt to the scale of any model
    candidates = np.quantile(scores, np.linspace(0.50, 0.99, 50))
    f1s = [f1_score(y, np.where(scores >= t, 1, -1)) for t in candidates]
    return candidates[np.argmax(f1s)]


def split_train_validation(y, x, ratio=0.8, seed=1):
    """Randomly split the data into a training part and a validation part.

    The model is trained on the training part and evaluated on the validation
    part, which it has never seen: this estimates its error on new people.

    Args:
        y: shape=(N, )
        x: shape=(N, D)
        ratio: fraction of the data used for training
        seed: random seed, so that the split is the same at every run

    Returns:
        x_tr, x_val, y_tr, y_val
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))  # shuffle 0..N-1
    n_tr = int(ratio * len(y))
    tr, val = indices[:n_tr], indices[n_tr:]
    return x[tr], x[val], y[tr], y[val]
