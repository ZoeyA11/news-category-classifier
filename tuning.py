"""
================================================================================
 Part 6.a (bonus) - Grid Search + K-Fold Cross Validation
================================================================================

What this part does:
  - Runs a grid search over hyperparameters.
  - For each combination, runs 5-fold cross validation on the TRAINING SET ONLY.
  - Averages the metric (Macro-F1) across the 5 folds and selects the best
    combination.
  - Searches TWO hyperparameters rather than one - `alpha` (smoothing) and
    `fit_prior` (whether to learn the class prior) - since using more than a
    single hyperparameter is worth additional credit.

--------------------------------------------------------------------------------
 Why StratifiedKFold rather than KFold
--------------------------------------------------------------------------------
 With 20 imbalanced classes, a plain split can produce a fold in which a rare
 class is absent entirely. Macro-F1 would then be computed over a class with no
 examples, which emits warnings and biases the metric downward. Stratification
 guarantees every fold preserves the class proportions.

--------------------------------------------------------------------------------
 Why the test set is never touched here
--------------------------------------------------------------------------------
 Hyperparameter selection is itself a form of learning. Selecting on the test
 set would leak it, and the final score would no longer be an honest estimate of
 performance on unseen data. Cross validation inside the training set is what
 lets us choose without spending the test set.
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from model import TextNaiveBayes

# Smoothing values to search. The range is logarithmic: on TF-IDF text features
# smaller alpha values usually work better, so the grid is denser at the low end.
ALPHA_GRID = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]

# The second hyperparameter: whether to learn the classes' prior distribution.
# On imbalanced data this is the interesting question - fit_prior=False is
# equivalent to assuming a uniform prior, which helps the rare classes and may
# therefore improve Macro-F1 specifically.
FIT_PRIOR_GRID = [True, False]


def grid_search_kfold(X_train, y_train, alpha_grid=ALPHA_GRID,
                      fit_prior_grid=FIT_PRIOR_GRID, n_splits: int = 5,
                      random_state: int = 42, n_jobs: int = 1) -> pd.DataFrame:
    """
    Runs grid search with k-fold cross validation on the training set.

    Returns a DataFrame with: alpha, fit_prior, mean Macro-F1, standard deviation.

    n_jobs=1 is deliberate: MultinomialNB trains extremely fast, and on Windows
    every worker process is spawned fresh and needs the entire sparse TF-IDF
    matrix pickled across to it. That overhead exceeds the saving. Raise it if
    the estimator is ever swapped for something heavier.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = []

    total = len(alpha_grid) * len(fit_prior_grid)
    print(f"Searching {total} combinations, each with {n_splits}-fold CV "
          f"({total * n_splits} fits in total)...\n")

    for fit_prior in fit_prior_grid:
        for alpha in alpha_grid:
            clf = TextNaiveBayes(alpha=alpha, fit_prior=fit_prior)
            scores = cross_val_score(
                clf, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=n_jobs
            )
            results.append({
                "alpha": alpha,
                "fit_prior": fit_prior,
                "mean_macro_f1": np.mean(scores),
                "std_macro_f1": np.std(scores),
            })
            print(f"alpha={alpha:<7} fit_prior={str(fit_prior):<5} | "
                  f"mean macro-F1={np.mean(scores):.4f} | std={np.std(scores):.4f}")

    return (pd.DataFrame(results)
            .sort_values("mean_macro_f1", ascending=False)
            .reset_index(drop=True))


def best_params(results_df: pd.DataFrame) -> dict:
    """Returns the best-scoring hyperparameter combination by mean Macro-F1."""
    best = results_df.iloc[0]
    return {"alpha": float(best["alpha"]), "fit_prior": bool(best["fit_prior"])}


if __name__ == "__main__":
    import console_utf8  # noqa: F401  (switches the Windows console to UTF-8)

    print("This file is intended to be used as a module - see main.py")
