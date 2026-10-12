"""Minimal cleaning of the data: missing values and standardization."""

import os

import numpy as np

# Columns that describe the phone survey itself (call date, ids, screening
# questions, questionnaire version) or the statistical weights of the survey,
# not the person: they carry no medical information.
USELESS_COLUMNS = [
    "_STATE", "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE",
    "SEQNO", "_PSU", "PVTRESD1", "CELLFON3", "QSTVER",
    "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_DUALUSE", "_DUALCOR", "_LLCPWT",
]  # fmt: skip


def load_feature_names(data_path):
    """Read the column names of x_train.csv, without the Id column."""
    with open(os.path.join(data_path, "x_train.csv")) as f:
        return f.readline().strip().split(",")[1:]


def clean_and_standardize(x_train, x_test, names, max_nan=0.5):
    """Make the data usable by the models of implementations.py.

    All the statistics (NaN rate, medians, means, stds) are computed on
    x_train only and then applied unchanged to x_test, so that the test data
    never influences the preprocessing.

    Args:
        x_train: shape=(N, D), raw training data (contains NaN)
        x_test: shape=(N_test, D), raw test data (contains NaN)
        names: list of the D column names, see load_feature_names
        max_nan: columns with a larger fraction of NaN in x_train are dropped

    Returns:
        tx_train: shape=(N, D'+1), cleaned training data with a bias column
        tx_test: shape=(N_test, D'+1), cleaned test data with a bias column
    """
    # 1. drop the useless columns and the columns that are almost empty
    nan_rate = np.isnan(x_train).mean(axis=0)
    keep = (nan_rate <= max_nan) & ~np.isin(names, USELESS_COLUMNS)
    x_train, x_test = x_train[:, keep], x_test[:, keep]

    # 2. replace the remaining NaN by the median of the column
    medians = np.nanmedian(x_train, axis=0)
    x_train = np.where(np.isnan(x_train), medians, x_train)
    x_test = np.where(np.isnan(x_test), medians, x_test)

    # 3. drop the constant columns (e.g. the survey year): they carry no
    # information and a column of zeros makes X^T X singular
    std = x_train.std(axis=0)
    x_train, x_test = x_train[:, std > 0], x_test[:, std > 0]

    # 4. standardize: mean 0 and std 1 for every column
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # 5. add the bias column (a column of ones) in front
    tx_train = np.c_[np.ones(x_train.shape[0]), x_train]
    tx_test = np.c_[np.ones(x_test.shape[0]), x_test]
    return tx_train, tx_test
