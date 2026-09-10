"""
================================================================================
 Part 5 - Prediction and Model Quality Estimation on the Test Set
================================================================================

Primary evaluation metric: Macro-Average F1.

Why Macro-F1 and not accuracy. This is a 20-class problem on imbalanced data
(the largest class holds roughly 10x the smallest). Accuracy would reward a
model that predicts the large classes well and ignores the small ones entirely,
because the large classes make up most of the data.

  Precision = of everything I predicted as class c, how much really was c?
  Recall    = of everything that really was c, how much did I catch?
  F1        = their harmonic mean, so it penalises an imbalance between them.

Macro averaging computes F1 for each of the 20 classes separately and then takes
an unweighted mean, so the smallest class counts exactly as much as the largest.
That is the property we want: it forces the model to be good everywhere, not
just where the data is dense.

Weighted F1 is reported alongside it for contrast - it weights by class size and
therefore drifts back toward accuracy's blind spot.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from viz import (apply_style, despine, save, BLUE_SEQ, BAR_COLOR, SERIES,
                 TEXT_SECONDARY)

apply_style()


def evaluate_model(y_true, y_pred, label: str = "", verbose: bool = True) -> dict:
    """
    Computes and prints the main quality metrics.
    Returns a dict so results from several runs can be collected into a table.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n=== Results: {label} ===")
    print(f"Accuracy         : {acc:.4f}")
    print(f"Macro-Average F1 : {macro_f1:.4f}   <-- primary metric")
    print(f"Weighted F1      : {weighted_f1:.4f}")

    if verbose:
        print("\nFull classification report:")
        print(classification_report(y_true, y_pred, zero_division=0))

    return {"label": label, "accuracy": acc,
            "macro_f1": macro_f1, "weighted_f1": weighted_f1}


def first_predictions_table(model, X_test, y_true, text_series,
                            n: int = 5) -> pd.DataFrame:
    """
    Returns the first n test-set predictions side by side with the ground truth -
    a required test deliverable.

    Includes the model's confidence, because a wrong prediction made at 40%
    confidence is a very different kind of error from a wrong prediction made at
    99%, and the raw label alone hides that distinction.
    """
    X_head = X_test[:n]
    preds = model.predict(X_head)
    probs = model.predict_proba(X_head)
    classes = list(model.classes_)

    truth = np.asarray(y_true)[:n]
    texts = np.asarray(text_series)[:n]

    rows = []
    for i in range(len(preds)):
        confidence = probs[i][classes.index(preds[i])]
        rows.append({
            "headline": (texts[i][:60] + "...") if len(texts[i]) > 60 else texts[i],
            "true": truth[i],
            "predicted": preds[i],
            "confidence": f"{confidence:.1%}",
            "correct": "yes" if preds[i] == truth[i] else "NO",
        })

    return pd.DataFrame(rows)


def per_class_f1(y_true, y_pred) -> pd.DataFrame:
    """Returns per-class F1 and support, best class first."""
    labels = sorted(pd.unique(y_true))
    f1s = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    support = pd.Series(y_true).value_counts().reindex(labels).to_numpy()
    return (pd.DataFrame({"category": labels, "f1": f1s, "support": support})
            .sort_values("f1", ascending=False)
            .reset_index(drop=True))


# --------------------------------------------------------------------------
#  Charts
# --------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, labels, save_path: str = None,
                          normalize: bool = True):
    """
    Confusion matrix as a heatmap.

    normalize=True normalizes per row, so each cell is the share of a true class
    that was predicted as each class. This is essential here: without it the
    POLITICS row - about 10x larger than the smallest class - saturates the whole
    colour scale and every small class looks empty.

    Colour: a single hue light-to-dark, because a cell encodes MAGNITUDE, not
    identity.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, cmap=BLUE_SEQ, vmin=0, vmax=1 if normalize else None,
                   aspect="auto")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix"
                 + (" (normalized per true class)" if normalize else ""))
    ax.grid(False)
    despine(ax, keep=())

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Share of true class" if normalize else "Count",
                   color=TEXT_SECONDARY)
    cbar.outline.set_visible(False)

    fig.tight_layout()
    if save_path:
        save(fig, save_path)
    return fig


def plot_per_class_f1(y_true, y_pred, save_path: str = None):
    """
    Per-class F1, sorted. Horizontal bars because the class names are long.
    A single hue, because the bar encodes magnitude and the axis already
    carries the identity.
    """
    df = per_class_f1(y_true, y_pred).sort_values("f1")

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(df["category"], df["f1"], color=BAR_COLOR, height=0.72)

    # Direct labels on every bar: with only 20 classes this stays readable, and
    # it saves the reader from tracking back to the axis and forward again.
    for bar, (f1, sup) in zip(bars, zip(df["f1"], df["support"])):
        ax.text(bar.get_width() + 0.012, bar.get_y() + bar.get_height() / 2,
                f"{f1:.2f}  (n={sup:,})", va="center", fontsize=8,
                color=TEXT_SECONDARY)

    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F1 score")
    ax.set_title("Per-class F1 - where the model succeeds and fails")
    ax.grid(axis="y", visible=False)
    despine(ax)
    fig.tight_layout()
    if save_path:
        save(fig, save_path)
    return fig


def plot_model_comparison(results: list, save_path: str = None):
    """
    Model comparison: accuracy against Macro-F1.
    Two series, so two categorical hues in a fixed order plus a legend (always
    present from two series up). Both metrics live on the same 0-1 scale, so a
    single axis - never a dual axis.
    """
    df = pd.DataFrame(results)
    x = np.arange(len(df))
    w = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w / 2, df["accuracy"], w, label="Accuracy", color=SERIES[0])
    b2 = ax.bar(x + w / 2, df["macro_f1"], w, label="Macro-F1", color=SERIES[1])

    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{bar.get_height():.3f}", ha="center", fontsize=8,
                    color=TEXT_SECONDARY)

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=15, ha="right")
    ax.set_ylim(0, max(df[["accuracy", "macro_f1"]].max()) * 1.18)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison - Accuracy vs Macro-F1")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    despine(ax)
    fig.tight_layout()
    if save_path:
        save(fig, save_path)
    return fig


def summarize_results(results: list) -> pd.DataFrame:
    """
    Collects results from several runs into a single table - required by
    Part 6.a (presenting results in a DataFrame).
    """
    return pd.DataFrame(results)
