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


def is_categorical(column, max_categories):
    """True if the column has between 3 and max_categories integer values.

    These columns are answer codes (e.g. EMPLOY1: 1 = employed, ...,
    8 = retired; yes/no questions: 1 = yes, 2 = no, 7 = don't know,
    9 = refused), not quantities.
    """
    values = np.unique(column[~np.isnan(column)])
    return 3 <= len(values) <= max_categories and np.all(values == np.round(values))


def one_hot_encode(x_train, x_test):
    """Replace each categorical column by one 0/1 column per category.

    NaN is treated as its own category ("missing"). The most frequent category
    is dropped (it is the reference): otherwise the 0/1 columns of a question
    always sum to 1, like the bias column, and X^T X becomes singular.
    A value of x_test never seen in x_train gets 0 everywhere (= reference).

    Args:
        x_train: shape=(N, C), categorical columns of the training data
        x_test: shape=(N_test, C), the same columns for the test data

    Returns:
        onehot_train: shape=(N, K), 0/1 columns
        onehot_test: shape=(N_test, K), 0/1 columns
    """
    nan_code = -1  # NaN cannot be compared with ==, give it its own code
    x_train = np.where(np.isnan(x_train), nan_code, x_train)
    x_test = np.where(np.isnan(x_test), nan_code, x_test)

    cols_train, cols_test = [], []
    for j in range(x_train.shape[1]):
        categories, counts = np.unique(x_train[:, j], return_counts=True)
        reference = categories[np.argmax(counts)]
        for category in categories[categories != reference]:
            cols_train.append(x_train[:, j] == category)
            cols_test.append(x_test[:, j] == category)
    onehot_train = np.array(cols_train, dtype=float).T
    onehot_test = np.array(cols_test, dtype=float).T
    return onehot_train, onehot_test


def fix_special_codes(x):
    """Replace the BRFSS answer codes of the continuous columns.

    The code depends on the largest value of the column:
    - max 99:     77 = don't know, 99 = refused -> NaN;  88 = none -> 0
    - max 999:    777 / 999 -> NaN;  888 = none, 555 = never -> 0
    - max 9999:   7777 / 9999 -> NaN
    - computed variables: 99900 (_DRNKWEK) and 900 (DROCDY3_) = don't know -> NaN
    E.g. PHYSHLTH = 88 means 0 days of bad health, not 88 days.

    Args:
        x: shape=(N, D), continuous columns (NaN allowed)

    Returns:
        a corrected copy of x
    """
    x = x.copy()
    for j in range(x.shape[1]):
        col = x[:, j]  # a view: modifying col modifies x
        col_max = np.nanmax(col)
        if col_max == 99:
            col[np.isin(col, [77, 99])] = np.nan
            col[col == 88] = 0
        elif col_max == 999:
            col[np.isin(col, [777, 999])] = np.nan
            col[np.isin(col, [555, 888])] = 0
        elif col_max == 9999:
            col[np.isin(col, [7777, 9999])] = np.nan
        elif col_max in (900, 99900):
            col[col == col_max] = np.nan
    return x


def clean_and_standardize(
    x_train,
    x_test,
    names,
    max_nan=0.5,
    one_hot=False,
    max_categories=10,
    fix_codes=False,
    clip_quantile=None,
    poly_degree=1,
):
    """Make the data usable by the models of implementations.py.

    All the statistics (NaN rate, medians, means, stds, categories) are
    computed on x_train only and then applied unchanged to x_test, so that
    the test data never influences the preprocessing.

    Args:
        x_train: shape=(N, D), raw training data (contains NaN)
        x_test: shape=(N_test, D), raw test data (contains NaN)
        names: list of the D column names, see load_feature_names
        max_nan: columns with a larger fraction of NaN in x_train are dropped
        one_hot: if True, the categorical columns are one-hot encoded
        max_categories: a column with at most this many integer values is
            considered categorical
        fix_codes: if True, replace the BRFSS answer codes of the continuous
            columns (see fix_special_codes)
        clip_quantile: if e.g. 0.99, clip every continuous column between its
            1st and 99th percentile (outliers); None = no clipping
        poly_degree: add the powers 2..poly_degree of every continuous column

    Returns:
        tx_train: shape=(N, D'+1), cleaned training data with a bias column
        tx_test: shape=(N_test, D'+1), cleaned test data with a bias column
    """
    # 1. drop the useless columns and the columns that are almost empty
    nan_rate = np.isnan(x_train).mean(axis=0)
    keep = (nan_rate <= max_nan) & ~np.isin(names, USELESS_COLUMNS)
    x_train, x_test = x_train[:, keep], x_test[:, keep]

    # 1b. one-hot encode the categorical columns, the others continue below
    onehot_train = np.empty((x_train.shape[0], 0))
    onehot_test = np.empty((x_test.shape[0], 0))
    if one_hot:
        categorical = np.array(
            [
                is_categorical(x_train[:, j], max_categories)
                for j in range(x_train.shape[1])
            ]
        )
        onehot_train, onehot_test = one_hot_encode(
            x_train[:, categorical], x_test[:, categorical]
        )
        x_train, x_test = x_train[:, ~categorical], x_test[:, ~categorical]

    # 1c. special answer codes and outliers of the continuous columns
    if fix_codes:
        x_train, x_test = fix_special_codes(x_train), fix_special_codes(x_test)
    if clip_quantile is not None:
        low = np.nanquantile(x_train, 1 - clip_quantile, axis=0)
        high = np.nanquantile(x_train, clip_quantile, axis=0)
        x_train, x_test = np.clip(x_train, low, high), np.clip(x_test, low, high)

    # 2. replace the remaining NaN by the median of the column
    medians = np.nanmedian(x_train, axis=0)
    x_train = np.where(np.isnan(x_train), medians, x_train)
    x_test = np.where(np.isnan(x_test), medians, x_test)

    # 3. drop the constant columns (e.g. the survey year): they carry no
    # information and a column of zeros makes X^T X singular
    std = x_train.std(axis=0)
    x_train, x_test = x_train[:, std > 0], x_test[:, std > 0]

    # 3b. polynomial features: x^2, ..., x^degree of each continuous column,
    # computed on the values scaled to mean 0 / std 1 (otherwise x^3 of a
    # weight in pounds would be enormous); standardized again at step 4
    if poly_degree > 1:
        mean, std = x_train.mean(axis=0), x_train.std(axis=0)
        z_train, z_test = (x_train - mean) / std, (x_test - mean) / std
        powers = range(2, poly_degree + 1)
        x_train = np.c_[x_train, *[z_train**p for p in powers]]
        x_test = np.c_[x_test, *[z_test**p for p in powers]]

    # 4. standardize: mean 0 and std 1 for every column
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # 5. add the bias column (a column of ones) in front, then the one-hot
    # columns (kept as 0/1: standardizing a rare category would turn its 1
    # into a huge value)
    tx_train = np.c_[np.ones(x_train.shape[0]), x_train, onehot_train]
    tx_test = np.c_[np.ones(x_test.shape[0]), x_test, onehot_test]
    return tx_train, tx_test
