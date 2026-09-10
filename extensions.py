"""
================================================================================
 Parts 6.b and 6.c (bonus)
================================================================================

--------------------------------------------------------------------------------
 6.b - Dataset-specific adaptations: handling imbalanced data
--------------------------------------------------------------------------------
 The dataset is not balanced across the 20 categories: POLITICS holds roughly
 10x the smallest class. This module provides oversampling with
 RandomOverSampler and compares its effect on Macro-F1.

 The result is reported honestly: oversampling made things *worse* here, and the
 reason is explainable - the grid search had already selected fit_prior=False,
 which neutralises the imbalance by assuming a uniform prior. Treating the same
 problem twice adds nothing. The genuine fix for this dataset turned out to be
 the duplicate-label merge in data_loader.py - see src/experiment_merge.py.

--------------------------------------------------------------------------------
 6.c - Explainability: understanding and presenting the influential features
--------------------------------------------------------------------------------
 Naive Bayes is not a black box. Its decision follows directly from
 log P(word | class), and those log-probabilities are the model's entire set of
 learned parameters. So no SHAP or LIME is needed - the explanation can be read
 straight out of the model.

 The ranking method matters, though, and that distinction is the substance of
 this section. See `top_features_per_class`.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

from viz import apply_style, despine, save, BAR_COLOR, SERIES, TEXT_SECONDARY

apply_style()

LABEL_COL = "category"


# --------------------------------------------------------------------------
#  6.b - Imbalanced data
# --------------------------------------------------------------------------

def show_class_imbalance(df: pd.DataFrame, label_col: str = LABEL_COL) -> pd.Series:
    """Prints the class distribution, to make the imbalance explicit."""
    counts = df[label_col].value_counts()
    print("Class distribution (largest to smallest):")
    print(counts.to_string())
    print(f"\nLargest class : {counts.idxmax()} ({counts.max():,} examples)")
    print(f"Smallest class: {counts.idxmin()} ({counts.min():,} examples)")
    print(f"Imbalance ratio: {counts.max() / counts.min():.1f}x")
    return counts


def plot_class_distribution(df: pd.DataFrame, label_col: str = LABEL_COL,
                            save_path: str = None):
    """
    Class distribution as sorted horizontal bars.
    A single hue: the bar encodes size (magnitude), not identity.
    """
    counts = df[label_col].value_counts().sort_values()

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(counts.index, counts.to_numpy(), color=BAR_COLOR, height=0.72)
    for bar, v in zip(bars, counts.to_numpy()):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{v:,}", va="center", fontsize=8, color=TEXT_SECONDARY)

    ax.set_xlabel("Number of articles")
    ax.set_title(f"Class distribution - {counts.max() / counts.min():.0f}x imbalance")
    ax.set_xlim(0, counts.max() * 1.12)
    ax.grid(axis="y", visible=False)
    despine(ax)
    fig.tight_layout()
    if save_path:
        save(fig, save_path)
    return fig


def apply_oversampling(X_train, y_train, random_state: int = 42):
    """
    Applies random over-sampling to the TRAINING SET ONLY.

    This restriction is not cosmetic. Oversampling the test set would corrupt
    the evaluation, and oversampling before the split would leak duplicates of
    the same row into both train and test - so the model would be tested on rows
    it had already memorised.
    """
    ros = RandomOverSampler(random_state=random_state)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

    print(f"Training set size before oversampling: {X_train.shape[0]:,}")
    print(f"Training set size after oversampling : {X_resampled.shape[0]:,}")
    print(f"Every class now holds: "
          f"{pd.Series(y_resampled).value_counts().iloc[0]:,} examples")

    return X_resampled, y_resampled


# --------------------------------------------------------------------------
#  6.c - Explainability
# --------------------------------------------------------------------------

def top_features_per_class(model, vectorizer, class_label: str, top_n: int = 15,
                           discriminative: bool = True,
                           verbose: bool = True) -> pd.DataFrame:
    """
    Returns the features that most influence classification into a given class.

    There are two ways to rank, and the difference between them is the whole
    point of this section:

    discriminative=False -> rank by the highest log P(w | c).
        This returns the words most FREQUENT in the class. The problem is that
        generic words common across every category rise to the top of every
        class's list, so they do not explain why *this* class was chosen.

    discriminative=True (the default) -> rank by log-odds:
        log P(w | c) minus the mean of log P(w | c') over all other classes.
        This isolates what makes the class distinctive, which is what actually
        drives the decision when Naive Bayes chooses between classes.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    classes = list(model.classes_)
    if class_label not in classes:
        raise ValueError(f"'{class_label}' is not a known class. Classes: {classes}")

    class_idx = classes.index(class_label)
    log_probs = model.feature_log_prob_          # shape: (n_classes, n_features)

    if discriminative:
        others = np.delete(log_probs, class_idx, axis=0).mean(axis=0)
        score = log_probs[class_idx] - others
        score_name = "log_odds_vs_rest"
    else:
        score = log_probs[class_idx]
        score_name = "log_prob"

    top_indices = np.argsort(score)[::-1][:top_n]
    result = pd.DataFrame({
        "feature": feature_names[top_indices],
        score_name: score[top_indices],
    })

    if verbose:
        kind = "most distinctive" if discriminative else "most frequent"
        print(f"\n{kind.capitalize()} words for '{class_label}':")
        print(result.to_string(index=False))

    return result


def plot_top_features_grid(model, vectorizer, class_labels, top_n: int = 10,
                           save_path: str = None):
    """
    Small multiples: one panel per category showing the words that distinguish
    it. Small multiples rather than a single colourful chart, because the
    interesting comparison is WITHIN each category, not between categories.
    """
    n = len(class_labels)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.1 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, label in zip(axes, class_labels):
        d = top_features_per_class(model, vectorizer, label, top_n=top_n,
                                   verbose=False).iloc[::-1]
        ax.barh(d["feature"], d["log_odds_vs_rest"], color=BAR_COLOR, height=0.7)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("log-odds vs rest", fontsize=8)
        ax.grid(axis="y", visible=False)
        ax.tick_params(labelsize=8)
        despine(ax)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Most distinctive words per category (Naive Bayes log-odds)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        save(fig, save_path)
    return fig


def plot_grid_search(results_df: pd.DataFrame, save_path: str = None):
    """
    Grid search results: Macro-F1 as a function of alpha, one line per
    fit_prior value. Two series, so two categorical hues plus a legend. The
    x axis is logarithmic because the alpha grid itself is logarithmic.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (fp, grp) in enumerate(results_df.groupby("fit_prior")):
        grp = grp.sort_values("alpha")
        ax.errorbar(grp["alpha"], grp["mean_macro_f1"], yerr=grp["std_macro_f1"],
                    marker="o", markersize=6, capsize=3, color=SERIES[i],
                    label=f"fit_prior={fp}")

    best = results_df.iloc[0]
    ax.annotate(f"best: alpha={best['alpha']}, fit_prior={best['fit_prior']}\n"
                f"macro-F1={best['mean_macro_f1']:.4f}",
                xy=(best["alpha"], best["mean_macro_f1"]),
                # Placed down-left into empty space: sitting flush against the
                # point collided with the descending fit_prior=False line.
                xytext=(-40, -95), textcoords="offset points", fontsize=9,
                color=TEXT_SECONDARY, ha="right",
                arrowprops=dict(arrowstyle="->", color=TEXT_SECONDARY, lw=1,
                                connectionstyle="arc3,rad=0.15"))

    ax.set_xscale("log")
    ax.set_xlabel("alpha (smoothing)")
    ax.set_ylabel("Mean macro-F1 (5-fold CV)")
    ax.set_title("Grid search - 5-fold cross-validation on the train set")
    ax.legend(loc="lower left")
    despine(ax)
    fig.tight_layout()
    if save_path:
        save(fig, save_path)
    return fig
