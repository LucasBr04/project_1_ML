# EPFL CS-433 — Project 1: BRFSS Heart Disease Classification

This project predicts the binary `_MICHD` target from BRFSS survey features for the EPFL Machine Learning course (Fall 2026). Labels are `-1` for the negative class and `+1` for the positive class. Since positive observations represent approximately 9% of the data, the main classification metric is **positive-class F1**.

The project provides six NumPy implementations, a submission notebook, a Least Squares/Ridge comparison notebook, and SGD experiments.

## Installation

Create an environment from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib notebook
```

Use Python 3.9 or later with compatible package versions. The project does not provide a pinned training environment. NumPy performs the numerical computations, Matplotlib handles visualization, and Jupyter runs the notebooks. The model code uses no Pandas, SciPy, scikit-learn, or other external machine learning libraries.

The course grading environment is separate and is described in `grading_tests/environment.yml`.

## Project structure

```text
.
├── run.ipynb                         # Logistic experiments and final submission export
├── Least_Squares_Ridge_BRFSS.ipynb   # LS/Ridge calibration and independent evaluation
├── sgd_tests.ipynb                   # SGD stability, convergence and learning curves
├── implementations.py               # Six required methods and numerical helpers
├── preprocessing.py                 # Cleaning, encoding and standardization
├── validation.py                    # Train/validation split, F1 and threshold search
├── helpers.py                       # CSV loading and submission export
├── plots.py                         # Logistic diagnostic plots
├── data/
│   ├── x_train.csv                  # Used by the LS/Ridge and SGD notebooks
│   ├── y_train.csv
│   ├── x_test.csv
│   └── dataset/
│       ├── x_train.csv              # Used by run.ipynb
│       ├── y_train.csv
│       ├── x_test.csv
│       └── sample_submission.csv
├── outputs/                         # Submission CSVs and logistic diagnostic figures
├── grading_tests/                   # Public course tests and their instructions
├── latex-example-paper/             # Course report template
└── project1_description.pdf         # Project instructions
```

There is currently no `run.py`; the submission entry point is **`run.ipynb`**. All notebook paths are relative to the current working directory, so launch Jupyter from the project root.

## Data

Use the competition's original `x_train.csv`, `y_train.csv`, and `x_test.csv`, keeping their filenames. The current local project contains copies in both `data/` and `data/dataset/`. To run all notebooks unchanged, provide the files in both locations and ensure the copies are identical.

`load_csv_data()` separates the first column of each feature CSV into observation identifiers and loads the second column of `y_train.csv` as labels. Test identifiers are preserved when writing submissions.

The existing `x_train_clean.csv` and `x_test_clean.csv` files are not used by these notebooks; preprocessing is recomputed from the raw CSVs. Cached Python files, notebook checkpoints and macOS metadata are not required.

## Generate the submission

From the project root:

```bash
jupyter notebook run.ipynb
```

Restart the kernel and run all cells in order. The notebook loads data from `data/dataset/` and creates `outputs/` if needed. It performs experiments before fitting the final model; it is not an inference-only notebook and does not load pretrained weights.

### Current execution procedure

1. Load the raw training and test data and apply `clean_and_standardize()` with its default options to the full labeled set and test set.
2. Split the preprocessed labeled data into 80% training and 20% validation using `np.random.default_rng(seed=1)`, without stratification.
3. Fit an initial regularized logistic model with `lambda_=1e-4`, `gamma=0.5`, 500 iterations and zero initial weights.
4. Run five-fold cross-validation for `lambda_` in `[0, 1e-5, 1e-4, 1e-3, 1e-2]`, using 500 iterations and `gamma=0.5`. Folds are built with `np.random.default_rng(seed=1)` and `np.array_split`.
5. Select the result tuple with the largest mean F1 and retain its lambda and mean fold threshold as `best_lambda` and `best_t`. Python tuple ordering breaks ties using the subsequent tuple fields.
6. Retrain regularized logistic regression on all labeled observations with the selected lambda, 500 iterations and `gamma=0.5`.
7. Classify test observations using `np.where(tx_test @ w >= best_t, 1, -1)` and export **`outputs/submission_reg_logistic.csv`**.

The single-split lambda sweep in section 4.1 is inside a triple-quoted string and is not executed. After final training, the notebook computes another threshold named `t` for a printed diagnostic; the submission uses **`best_t`**, not `t`.

### Outputs

| File | Contents |
|---|---|
| `outputs/submission_reg_logistic.csv` | Final predictions with columns `Id,Prediction` and labels in `{-1, 1}` |
| `outputs/convergence.png` | Training log-loss diagnostic on a 50,000-row training subsample |
| `outputs/cv_lambda.png` | Mean F1 and standard deviation across folds versus lambda |
| `outputs/scores.png` | Class score distributions on the validation subset |
| `outputs/threshold.png` | F1, precision and recall versus the decision threshold |

These files are overwritten on subsequent runs. Existing `outputs/submission.csv` and `outputs/losses.png` are not generated by the current `run.ipynb`.

The saved cross-validation output selects `lambda_=1e-4` and `best_t ≈ -1.396594`, with mean F1 approximately `0.411867`. These are stored notebook outputs, not guaranteed constants: the notebook recalculates them when executed.

## Preprocessing and feature generation

`clean_and_standardize()` applies the following steps:

1. Remove the 19 survey, administrative and weighting variables listed in `USELESS_COLUMNS`, and columns with more than 50% missing training values.
2. Optionally one-hot encode columns with 3–10 distinct observed integer values. Missing values form a category, the most frequent category is the reference, and an unseen category in the second dataset is encoded as all zeros. Binary columns remain in the numerical branch.
3. Optionally correct special BRFSS codes on the remaining numerical columns. The rules depend on each column's observed maximum: `77/99`, `777/999`, or `7777/9999` can become missing; `88`, `555`, or `888` can become zero; maxima `900` and `99900` can also become missing.
4. Optionally clip numerical columns using training quantiles, such as the 1st and 99th percentiles for `clip_quantile=0.99`.
5. Impute missing numerical values with training-column medians and remove columns with zero training standard deviation.
6. Optionally add individual polynomial powers from degree 2 to `poly_degree`, using standardized inputs. No feature interactions are added.
7. Standardize numerical columns using training means and standard deviations, then prepend a bias column of ones and append any categorical indicators, which remain 0/1.

All current notebooks call this function with its default options: `max_nan=0.5`, `one_hot=False`, `fix_codes=False`, `clip_quantile=None`, and `poly_degree=1`. Thus, one-hot encoding, special-code correction, clipping and polynomial expansion are available but **not enabled in the current workflows**. The saved baseline representation has 154 columns including the bias.

Selection masks, categories, medians, clipping bounds and scaling statistics are fitted on the first argument. If special-code correction is enabled, its maximum-based rules are inferred separately for each argument; this part is not fitted exclusively on the training argument.

## Required methods

`implementations.py` provides all six required functions. Each returns `(w, loss)`, with a one-dimensional weight vector and the corresponding unpenalized loss.

| Function | Method |
|---|---|
| `mean_squared_error_gd` | Full-batch gradient descent for half-MSE |
| `mean_squared_error_sgd` | Stochastic gradient descent, one randomly sampled observation per update |
| `least_squares` | Normal equations solved with `np.linalg.solve` |
| `ridge_regression` | Solve `(X.T @ X + 2*N*lambda_*I) @ w = X.T @ y` |
| `logistic_regression` | Full-batch gradient descent for mean logistic loss |
| `reg_logistic_regression` | Logistic descent with `2*lambda_*w` added to the gradient |

Logistic methods require labels in `{0, 1}`; the notebooks convert the original labels accordingly. Quadratic-loss experiments use `{-1, 1}`. All coefficients, including the bias, are penalized by the regularized methods. Logistic loss uses `np.logaddexp` for numerical stability.

## Threshold selection

All prediction thresholds apply to the continuous score **`X @ w`**, not directly to a probability. Scores can be negative. A score threshold of zero therefore does not classify every observation as positive; for logistic regression, it corresponds to a probability threshold of 0.5 after the sigmoid.

Two threshold searches are used:

- `validation.best_threshold()`, used in the logistic and SGD notebooks, tests 50 score quantiles from 50% through 99% and selects the first threshold maximizing positive-class F1.
- `best_f1_threshold()`, defined in the LS/Ridge notebook, sorts calibration scores and performs an exact search over distinct observed score thresholds, treating tied scores together.

## Least Squares and Ridge experiments

Run `Least_Squares_Ridge_BRFSS.ipynb` from a fresh kernel, executing all cells in order. It reads raw CSVs from `data/`.

The notebook first splits labeled observations into 80% training and 20% validation using `np.random.seed(42)`. The initial split is not stratified. Preprocessing is fitted on training data only. The validation portion is then split into two stratified halves with seed 43: 32,814 calibration observations and 32,813 evaluation observations.

For Least Squares, `pivoted_qr_numpy()` selects independent columns using NumPy QR reduction followed by Householder reflections with column pivoting. Numerical rank is estimated using singular values and a tolerance based on the original matrix dimensions. The same column selection is applied to calibration and evaluation data. This implementation does not require SciPy.

For Ridge, all preprocessed columns are retained. The notebook evaluates 30 lambda values from `np.logspace(-6, 2, 30)`. It reuses the training Gram matrix and right-hand side, solves the regularized normal equations for each lambda, and jointly selects lambda and threshold by calibration F1. Evaluation data are used after selection. No cross-validation is performed in this notebook.

### Recorded independent evaluation results

| Model | Threshold | F1 | Precision | Recall | Accuracy |
|---|---|---:|---:|---:|---:|
| Always-negative baseline | — | 0.0000 | 0.0000 | 0.0000 | 0.9101 |
| Least Squares | 0 | 0.0054 | 0.7273 | 0.0027 | 0.9102 |
| Least Squares | Calibrated | 0.4089 | 0.3280 | 0.5427 | 0.8589 |
| Ridge | 0 | 0.0047 | 0.7000 | 0.0024 | 0.9102 |
| Ridge | Calibrated | 0.4079 | 0.3235 | 0.5519 | 0.8560 |

The selected Ridge lambda is approximately `1.268961e-5`, with threshold `-0.582888`; the LS threshold is approximately `-0.576543`. These are local evaluation results, not competition scores.

Tables are displayed as aligned text using lists of dictionaries, and figures are displayed inline. The current notebook does **not** save figures, export result tables, retrain on the full dataset, or create LS/Ridge submission CSVs. Although the loader reads the test CSV, test observations are not used in fitting, calibration or evaluation.

## SGD experiments

`sgd_tests.ipynb` reads from `data/`, splits raw labeled observations 80/20 using `validation.split_train_validation()` with seed 1, and fits default preprocessing on the training portion. Thresholds are selected on training scores and F1 is measured on validation data.

The notebook compares SGD with Least Squares and explores:

- Learning rates `[3e-4, 1e-4, 3e-5, 1e-5]` and update counts `[10000, 100000, 300000, 1000000]`.
- Seeds 1–5 with `gamma=3e-5` and one million updates.
- Convergence in chunks, training/validation loss and F1, and learning curves with different training sizes.

The recorded five-seed result is validation F1 `0.4081 ± 0.0019` (population standard deviation). Because the same validation set is used during exploration, this is not an independent evaluation after configuration selection. The notebook displays figures and does not export a submission.

## Evaluation and reproducibility notes

The LS/Ridge notebook and the submission notebook use different evaluation procedures. Their scores should not be interpreted as a comparison on the same independent evaluation set.

In the current `run.ipynb`, preprocessing is fitted on all labeled data before splitting or constructing folds. Validation observations therefore influence preprocessing statistics. In addition, thresholds are tuned and scored on the same held-out fold. Cross-validation scores are tuning diagnostics rather than an evaluation independent of preprocessing and threshold selection. After full-data retraining, the subset still called validation is part of the training data, so the printed final F1 is not an independent validation score.

The report discussed alongside this project describes a final logistic model with one-hot preprocessing and a separate calibration/evaluation protocol. The current `run.ipynb` instead uses default preprocessing and regularized logistic regression with the cross-validation procedure documented above. Its saved outputs do not establish reproduction of the report's logistic F1 of 0.4337 or the reported competition score. Exact reproduction of the best competition CSV remains to be checked against that submitted file.

The code and saved notebook outputs were inspected for this README. The NumPy LS/Ridge implementation was previously checked on the supplied data against the archived result tables, reproducing their metrics with numerical differences around `2e-11` or smaller. The current full logistic cross-validation workflow and public grading tests were not rerun for this documentation update. Existing notebook outputs may reflect earlier execution states.

## Public course tests

Follow `grading_tests/INSTRUCTIONS.md` and use the dedicated grading environment:

```bash
conda env create --file grading_tests/environment.yml --name project1-grading
conda activate project1-grading
cd grading_tests
pytest --github_link /absolute/path/to/project .
```

Replace the path with the project root. A repository URL can be supplied instead. When using a local directory, the GitHub URL check may fail as described in the test instructions. Test-environment packages are not model dependencies, and passing the public tests does not establish full implementation correctness or submission reproducibility.
