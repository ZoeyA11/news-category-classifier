"""
================================================================================
 Part 6.b (bonus) - Dataset-specific adaptation: merging near-duplicate labels
================================================================================

 Usage: python src/experiment_merge.py

--------------------------------------------------------------------------------
 Background
--------------------------------------------------------------------------------
 In the tuned model's per-class F1 table, the two worst classes by a wide margin
 are PARENTS (F1 ~ 0.30) and HEALTHY LIVING (F1 ~ 0.29) - which are precisely
 the semantic duplicates of PARENTING and WELLNESS. HuffPost renamed sections
 over the years, so the same topic appears under two labels.

 This script tests the hypothesis that the failure is a LABELLING problem rather
 than a model failure, in two complementary ways:

   Experiment A - "how much of the error is duplicate labelling in the first
                   place?"
       Take the SAME model without retraining, and collapse the duplicate pairs
       in both y_true and y_pred. If F1 jumps, the model was effectively right
       and the label was the arbitrary part.

   Experiment B - "does merging before training help?"
       Merge before the top-20 reduction, then retrain. This is the real fix.
       Note the caveat: the number of classes is still 20, but their COMPOSITION
       differs, so Macro-F1 between Experiment B and the baseline is not a
       strict apples-to-apples comparison - it measures a better-defined task,
       not only a better model.
================================================================================
"""

import console_utf8  # noqa: F401  (switches the Windows console to UTF-8)

import os

import numpy as np
import pandas as pd

from data_loader import (load_raw_data, reduce_to_top_classes, split_train_test,
                         merge_duplicate_categories, DUPLICATE_GROUPS, LABEL_COL)
from feature_engineering import build_tfidf_features
from model import TextNaiveBayes
from evaluate import evaluate_model, per_class_f1

# The hyperparameters chosen by the grid search. Read from the grid results if
# main.py has already run, with a declared fallback otherwise - so the values
# cannot go quietly stale if the grid or the features are ever changed.
GRID_RESULTS = "outputs/grid_search_results.csv"
FALLBACK = {"alpha": 0.5, "fit_prior": False}


def load_best_params() -> dict:
    if os.path.exists(GRID_RESULTS):
        row = (pd.read_csv(GRID_RESULTS)
               .sort_values("mean_macro_f1", ascending=False).iloc[0])
        params = {"alpha": float(row["alpha"]), "fit_prior": bool(row["fit_prior"])}
        print(f"Loaded hyperparameters from {GRID_RESULTS}: {params}")
        return params

    print(f"{GRID_RESULTS} not found (run main.py first) - "
          f"using the declared fallback: {FALLBACK}")
    return FALLBACK


def header(text: str) -> None:
    print("\n" + "=" * 78)
    print(f" {text}")
    print("=" * 78)


def pipeline(df, label, best):
    """Trains and evaluates the tuned model on a given DataFrame."""
    train_df, test_df = split_train_test(df)
    X_tr, X_te, vec, train_df, test_df = build_tfidf_features(train_df, test_df)
    y_tr, y_te = train_df[LABEL_COL], test_df[LABEL_COL]

    model = TextNaiveBayes(**best).fit(X_tr, y_tr)
    preds = model.predict(X_te)
    results = evaluate_model(y_te, preds, label=label, verbose=False)
    return results, y_te, preds


def main():
    best = load_best_params()
    raw = load_raw_data()

    # ---------- Baseline: no merging ----------
    header("Baseline: top 20 classes, no merging")
    df_plain = reduce_to_top_classes(raw)
    base_res, y_te, preds = pipeline(df_plain, "Tuned, unmerged (20 classes)", best)

    # ---------- Experiment A: merge at evaluation time only ----------
    header("Experiment A: collapse duplicate pairs at evaluation only "
           "(same model, no retraining)")

    pairs = {"PARENTS": "PARENTING", "HEALTHY LIVING": "WELLNESS",
             "THE WORLDPOST": "WORLDPOST"}
    pairs = {k: v for k, v in pairs.items() if k in set(y_te)}

    # How much of each "duplicate" true class went to its twin?
    print("Leakage between the duplicate pairs (same model):")
    y_te_arr = np.asarray(y_te)
    for dup, canon in pairs.items():
        mask = y_te_arr == dup
        if not mask.any():
            continue
        correct = (preds[mask] == dup).mean()
        to_twin = (preds[mask] == canon).mean()
        print(f"  true={dup:<16} -> predicted correctly {correct:5.1%} | "
              f"predicted as twin '{canon}' {to_twin:5.1%}")

    merged_eval = evaluate_model(
        pd.Series(y_te_arr).replace(pairs),
        pd.Series(preds).replace(pairs),
        label="Same model, duplicates collapsed at eval",
        verbose=False)

    # ---------- Experiment B: merge before training ----------
    header("Experiment B: merge before reduction and training (the real fix)")
    df_merged = reduce_to_top_classes(merge_duplicate_categories(raw))
    merged_res, y_te2, preds2 = pipeline(df_merged,
                                         "Tuned, merged labels (20 classes)", best)

    # ---------- Summary ----------
    header("Comparison summary")
    summary = pd.DataFrame([base_res, merged_eval, merged_res])
    print(summary.to_string(index=False))
    summary.to_csv("outputs/merge_experiment.csv", index=False)
    print("\nsaved: outputs/merge_experiment.csv")

    print("\n--- Per-class F1 after merging before training (5 worst) ---")
    print(per_class_f1(y_te2, preds2).tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
