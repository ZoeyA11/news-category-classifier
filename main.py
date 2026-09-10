"""
================================================================================
 main.py - runs the entire assignment pipeline end to end
================================================================================

 Usage:  python src/main.py

 Must be run from the PROJECT ROOT, not from inside src/, because the data path
 is relative. Requires the dataset - run `python src/download_data.py` first,
 or see README.md.

 Pipeline:
   Part 1   - load, describe, stratified train/test split
   Part 2   - text cleaning and TF-IDF (fitted on train only)
   Part 3-4 - Naive Bayes, trained across different parameter settings
   Part 5   - evaluation on the held-out test set
   Part 6.a - grid search with 5-fold cross validation
   Part 6.b - imbalanced-data handling via oversampling
   Part 6.c - explainability through per-class log-odds
================================================================================
"""

# Must come first: switches the console to UTF-8 before anything is printed.
# The dataset text contains typographic quotes and em dashes that would
# otherwise crash on the Windows cp1252 console.
import console_utf8  # noqa: F401

import os

import pandas as pd

from data_loader import (load_raw_data, reduce_to_top_classes, describe_dataset,
                         split_train_test, show_split_samples,
                         verify_stratification, LABEL_COL)
from feature_engineering import (build_tfidf_features, show_before_after_examples,
                                 describe_matrix)
from model import TextNaiveBayes, save_bundle
from tuning import grid_search_kfold, best_params
from evaluate import (evaluate_model, plot_confusion_matrix, plot_per_class_f1,
                      plot_model_comparison, summarize_results, per_class_f1,
                      first_predictions_table)
from extensions import (show_class_imbalance, plot_class_distribution,
                        apply_oversampling, top_features_per_class,
                        plot_top_features_grid, plot_grid_search)

OUT = "outputs"


def header(text: str) -> None:
    print("\n" + "=" * 78)
    print(f" {text}")
    print("=" * 78)


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 140)

    # ===== Part 1: Data loading and overview =====
    header("Part 1: Loading and describing the dataset")
    df = load_raw_data()
    print(f"Loaded {len(df):,} records with {df[LABEL_COL].nunique()} categories "
          f"(before reduction).")
    df = reduce_to_top_classes(df)
    print()
    describe_dataset(df)

    train_df, test_df = split_train_test(df)
    print()
    show_split_samples(train_df, test_df, n=5)
    print()
    verify_stratification(train_df, test_df)   # prints its own summary line

    # ===== Part 2: Feature engineering =====
    header("Part 2: Feature Engineering")
    show_before_after_examples(train_df, n=3)
    X_train, X_test, vectorizer, train_df, test_df = build_tfidf_features(train_df, test_df)
    y_train = train_df[LABEL_COL]
    y_test = test_df[LABEL_COL]
    print()
    describe_matrix(X_train, X_test, vectorizer)

    # ===== Parts 3-4: Naive Bayes and training =====
    header("Parts 3-4: Training Naive Bayes (baseline, alpha=1.0)")
    baseline_model = TextNaiveBayes(alpha=1.0).fit(X_train, y_train)
    print("Baseline model trained with scikit-learn's default alpha=1.0.")

    # ===== Part 5: Evaluation on the test set =====
    header("Part 5: Evaluating the baseline model on the test set")
    baseline_preds = baseline_model.predict(X_test)
    baseline_results = evaluate_model(y_test, baseline_preds,
                                      label="Baseline (alpha=1.0)")

    # ===== Part 6.a: Grid search + k-fold cross validation =====
    header("Part 6.a (bonus): Grid Search + 5-Fold Cross Validation")
    cv_results_df = grid_search_kfold(X_train, y_train)
    print("\nGrid search results (best to worst):")
    print(cv_results_df.to_string(index=False))

    params = best_params(cv_results_df)
    print(f"\nSelected hyperparameters: {params}")

    tuned_model = TextNaiveBayes(**params).fit(X_train, y_train)
    tuned_preds = tuned_model.predict(X_test)
    tuned_results = evaluate_model(
        y_test, tuned_preds,
        label=f"Tuned (alpha={params['alpha']}, fit_prior={params['fit_prior']})")

    # Did cross validation actually predict test performance? This is the check
    # that shows we did not overfit the tuning process itself.
    cv_best = cv_results_df.iloc[0]["mean_macro_f1"]
    print(f"\nCV-predicted macro-F1 (on train): {cv_best:.4f}")
    print(f"Actual macro-F1 (on test)       : {tuned_results['macro_f1']:.4f}")
    print(f"Difference                      : "
          f"{abs(tuned_results['macro_f1'] - cv_best):.4f}")

    print("\nFirst 5 test-set predictions (predicted vs. true):")
    print(first_predictions_table(tuned_model, X_test, y_test,
                                  test_df["headline"], n=5).to_string(index=False))

    # Persist the tuned model with its vectorizer, so predictions can be made
    # later without retraining.
    save_bundle(tuned_model, vectorizer)

    # ===== Part 6.b: Imbalanced data =====
    header("Part 6.b (bonus): Handling imbalanced data")
    show_class_imbalance(train_df)
    print()
    X_train_bal, y_train_bal = apply_oversampling(X_train, y_train)

    balanced_model = TextNaiveBayes(**params).fit(X_train_bal, y_train_bal)
    balanced_preds = balanced_model.predict(X_test)
    balanced_results = evaluate_model(y_test, balanced_preds,
                                      label="Balanced (oversampling)")

    # ===== Consolidated comparison =====
    header("Comparison of the three models")
    all_results = [baseline_results, tuned_results, balanced_results]
    comparison_df = summarize_results(all_results)
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv(f"{OUT}/model_comparison.csv", index=False)
    cv_results_df.to_csv(f"{OUT}/grid_search_results.csv", index=False)

    # ===== Part 6.c: Explainability =====
    header("Part 6.c (bonus): The words that distinguish each category")
    explain_classes = ["POLITICS", "SPORTS", "FOOD & DRINK", "TRAVEL"]
    explain_classes = [c for c in explain_classes if c in list(tuned_model.classes_)]
    for cls in explain_classes:
        top_features_per_class(tuned_model, vectorizer, class_label=cls, top_n=10)

    # ===== Charts =====
    header("Generating charts")
    labels = sorted(y_test.unique())
    plot_class_distribution(train_df, save_path=f"{OUT}/class_distribution.png")
    plot_grid_search(cv_results_df, save_path=f"{OUT}/grid_search.png")
    plot_confusion_matrix(y_test, tuned_preds, labels=labels,
                          save_path=f"{OUT}/confusion_matrix.png")
    plot_per_class_f1(y_test, tuned_preds, save_path=f"{OUT}/per_class_f1.png")
    plot_model_comparison(all_results, save_path=f"{OUT}/model_comparison.png")
    if explain_classes:
        plot_top_features_grid(tuned_model, vectorizer, explain_classes,
                               top_n=10, save_path=f"{OUT}/top_features.png")

    per_class_f1(y_test, tuned_preds).to_csv(f"{OUT}/per_class_f1.csv", index=False)

    print("\nRun completed successfully.")


if __name__ == "__main__":
    main()
