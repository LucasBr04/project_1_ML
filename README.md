# EPFL CS-433 — Project 1: Heart Disease Prediction (BRFSS)

Team WEB: William Weber, Esther Deriaz, Lucas Brunschwick.

We predict the binary target `_MICHD` (reported coronary heart disease, labels `-1`/`+1`) from 321 BRFSS survey features. Only ~9% of the people are positive, so models are evaluated with the **F1-score of the positive class**.

**Final model:** logistic regression on one-hot encoded features, with a decision threshold tuned for F1.
**Score:** local evaluation F1 = 0.4337, AIcrowd F1 = 0.434 (accuracy 0.866).


## Data
Put the competition files, with their original names, in `data/`:
```
data/x_train.csv   data/y_train.csv   data/x_test.csv
```

## Reproduce the submission
From the project root, open `run.ipynb` and run all cells (Kernel → Restart & Run All).
It writes the submitted file `outputs/submission_reg_logistic_cv_one_hot_split_before.csv`.

## Pipeline (`run.ipynb`)
1. **Split** the raw labeled data: 80% training / 20% validation (seed 1).
2. **Preprocess** with `clean_and_standardize(..., one_hot=True)`, fitted on the training part only:
   drop administrative columns and columns with >50% missing values, one-hot encode the 95 categorical
   variables (NaN as its own category), impute the other columns with the median, standardize, add a bias.
3. **Train** regularized logistic regression by gradient descent (`gamma=0.5`, 500 iterations) for
   `lambda_` in {0, 1e-5, 1e-4, 1e-3, 1e-2}; keep the `lambda_` and the threshold with the best validation F1.
4. **Final model:** preprocessing refitted on all labeled data, retrained with the selected `lambda_`,
   predictions on the test set with the selected threshold.



## Files
| File | Content |
|---|---|
| `implementations.py` | The six required methods |
| `preprocessing.py` | `clean_and_standardize`: cleaning, one-hot encoding, standardization |
| `validation.py` | Train/validation split, F1-score, threshold search |
| `helpers.py` | CSV loading and submission writing (course helpers) |
| `plots.py` | Plots used in `run.ipynb` |
| `run.ipynb` | Logistic models and final submission |
| `Least_Squares_Ridge_BRFSS.ipynb` | Least squares and ridge experiments (report table) |
| `sgd_tests.ipynb` | GD/SGD experiments: step sizes, convergence, learning curves |

## Tests
```bash
cd grading_tests
pytest --github_link <repo-url> .
```
