"""
================================================================================
 Part 1 - Data Loading, Dataset Overview, and Train/Test Split
================================================================================

Machine Learning Project: News Category Classification (HuffPost News Dataset)

--------------------------------------------------------------------------------
 Student Information
--------------------------------------------------------------------------------
 Name                      : Zoey A
 Student ID (last 4 digits): 1269

--------------------------------------------------------------------------------
 AI Tools & Prompt Engineering
--------------------------------------------------------------------------------
 AI Tool : Claude (Opus) via Claude Code
 Purpose : Architectural planning for the NLP pipeline, designing the label
           reduction and duplicate-category merging logic, and defining the
           multi-class evaluation strategy (Macro-Average F1).

 Key prompts used:
   - "Find me an NLP-tagged dataset on Kaggle suitable for multi-class text
      classification with Naive Bayes."
   - "Design a TF-IDF + Multinomial Naive Bayes pipeline where the vectorizer is
      fitted on the training set only, with grid-search over more than one
      hyperparameter using stratified k-fold cross validation."
   - "The oversampling made Macro-F1 worse - find out why instead of working
      around it."

 The full prompt history is documented in PROMPT.md.

--------------------------------------------------------------------------------
 Problem Description & Dataset Overview
--------------------------------------------------------------------------------
 This project addresses a supervised multi-class text classification task using
 the News Category Dataset (209,527 HuffPost headlines, 2012-2022).

 Rather than classifying into the original 42 fine-grained editorial sections,
 we aggregate the targets down to the 20 most frequent categories. This
 reformulation is deliberate and serves three purposes: it aligns the task with
 the assignment specification (multi-class, 20 classes), it guarantees a higher
 sample density per class (every retained class holds at least ~3,400 examples,
 whereas the discarded tail holds fewer than 1,100 each), and it yields a more
 reliable Macro-Average F1 estimate, since that metric weights every class
 equally regardless of size.

 A second, dataset-specific adaptation is also provided here
 (`merge_duplicate_categories`): HuffPost renamed several sections over the
 years, so semantically identical topics appear under two different labels
 (PARENTS vs. PARENTING, HEALTHY LIVING vs. WELLNESS). This is a systematic
 labelling artifact rather than random noise - see Part 6.b for the evidence
 that the model predicts the twin label roughly twice as often as the
 "correct" one.

 Dataset  : https://www.kaggle.com/datasets/rmisra/news-category-dataset
 License  : CC BY 4.0 - attribution is a licence condition; see README.md
 Raw format: JSON Lines - one JSON object per line, NOT a single JSON array:
   {"link": "...", "headline": "Over 4 Million Americans Roll Up Sleeves...",
    "category": "U.S. NEWS", "short_description": "Health experts said...",
    "authors": "Carla K. Johnson, AP", "date": "2022-09-23"}
================================================================================
"""

import json
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
#  Configuration
# --------------------------------------------------------------------------

DATA_PATH = "data/news_category/News_Category_Dataset_v3.json"

# The target column we are predicting.
LABEL_COL = "category"

# The two free-text fields that form the model input.
TEXT_COLS = ["headline", "short_description"]

# Number of classes to retain. See "Problem Description" above for the rationale.
N_CLASSES = 20

# Categories that are different names for the same topic. Merging is applied
# BEFORE the top-N reduction, so a merged category competes for a place in the
# top 20 with its combined size.
DUPLICATE_GROUPS = {
    "PARENTING": ["PARENTING", "PARENTS"],
    "WELLNESS": ["WELLNESS", "HEALTHY LIVING"],
    "WORLDPOST": ["WORLDPOST", "THE WORLDPOST"],
    "ARTS & CULTURE": ["ARTS & CULTURE", "CULTURE & ARTS", "ARTS"],
    "STYLE & BEAUTY": ["STYLE & BEAUTY", "STYLE"],
}


def _show(df: pd.DataFrame) -> None:
    """
    Renders a DataFrame as a formatted table in Jupyter, or as plain text when
    running from a terminal. Keeps this module usable in both contexts without
    the caller having to care which one it is.

    Note the detection: it is not enough to check whether IPython can be
    imported, because it is installed alongside Jupyter and imports fine from a
    plain terminal too - where `display()` degrades to a width-limited repr that
    elides columns. Checking for a live kernel is what actually distinguishes
    the two contexts, so `to_string()` (which prints every column) is used in
    the terminal.
    """
    if "ipykernel" in sys.modules:
        from IPython.display import display

        display(df)
    else:
        print(df.to_string())


# --------------------------------------------------------------------------
#  Loading
# --------------------------------------------------------------------------

def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Loads the raw JSON Lines file and returns a DataFrame.

    The file is deliberately parsed line by line rather than with a single
    `json.load`: it is JSON Lines, so the file as a whole is not valid JSON and
    a single parse would raise `JSONDecodeError`.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            f"Run 'python src/download_data.py' to fetch it "
            f"(no Kaggle credentials required), or see README.md."
        )

    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return pd.DataFrame(records)


# --------------------------------------------------------------------------
#  Label preparation
# --------------------------------------------------------------------------

def merge_duplicate_categories(df: pd.DataFrame, groups: dict = None) -> pd.DataFrame:
    """
    Collapses categories that are alternative names for the same topic.

    Must be called BEFORE `reduce_to_top_classes` so that the merged category
    is ranked by its combined size.
    """
    groups = groups if groups is not None else DUPLICATE_GROUPS
    mapping = {old: new for new, olds in groups.items() for old in olds}

    df = df.copy()
    before = df[LABEL_COL].nunique()
    df[LABEL_COL] = df[LABEL_COL].replace(mapping)

    print(f"Merged duplicate categories: {before} -> {df[LABEL_COL].nunique()} classes")
    for new, olds in groups.items():
        print(f"  {' + '.join(olds)}  ->  {new}")

    return df


def reduce_to_top_classes(df: pd.DataFrame, n_classes: int = N_CLASSES) -> pd.DataFrame:
    """
    Keeps only the `n_classes` most frequent categories.
    Returns a new DataFrame; the input is left untouched.
    """
    top = df[LABEL_COL].value_counts().head(n_classes).index
    reduced = df[df[LABEL_COL].isin(top)].reset_index(drop=True)

    retained = 100 * len(reduced) / len(df)
    print(f"\nClass reduction: {df[LABEL_COL].nunique()} -> "
          f"{reduced[LABEL_COL].nunique()} classes")
    print(f"Rows: {len(df):,} -> {len(reduced):,} ({retained:.1f}% of the data retained)")
    print(f"Classes retained: {sorted(top)}")

    return reduced


# --------------------------------------------------------------------------
#  Overview
# --------------------------------------------------------------------------

def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a per-class count and share table, largest class first."""
    counts = df[LABEL_COL].value_counts()
    return pd.DataFrame({
        "category": counts.index,
        "count": counts.to_numpy(),
        "share_%": (100 * counts / counts.sum()).round(2).to_numpy(),
    }).reset_index(drop=True)


def describe_dataset(df: pd.DataFrame) -> None:
    """Prints a general overview of the dataset - required by Part 1."""
    print(f"Rows (news articles) : {len(df):,}")
    print(f"Columns              : {df.shape[1]}")
    print(f"Column names         : {list(df.columns)}")
    print(f"Unique '{LABEL_COL}' classes: {df[LABEL_COL].nunique()}")

    counts = df[LABEL_COL].value_counts()
    print(f"\nLargest class : {counts.idxmax()} ({counts.max():,} articles)")
    print(f"Smallest class: {counts.idxmin()} ({counts.min():,} articles)")
    print(f"Imbalance ratio: {counts.max() / counts.min():.1f}x")
    print("\nClass distribution:")
    _show(class_distribution(df))

    print("\nMissing values (NaN) per column:")
    print(df.isnull().sum().to_string())

    # In this dataset the text fields are never NaN - they are empty strings
    # instead, which `isnull()` alone silently misses. Checking explicitly
    # because an empty description is a real modelling consideration.
    print("\nEmpty text fields (empty string, not NaN):")
    for col in TEXT_COLS:
        empty = (df[col].fillna("").str.strip() == "").sum()
        print(f"  {col:<18}: {empty:,} ({100 * empty / len(df):.1f}%)")


# --------------------------------------------------------------------------
#  Train / test split
# --------------------------------------------------------------------------

def split_train_test(df: pd.DataFrame, test_size: float = 0.2,
                     random_state: int = 42):
    """
    Splits the dataset into train and test sets.

    Stratified on the label so that the class proportions are preserved in both
    sets. Without stratification a small class could end up with a very
    different share in the test set, which would make Macro-F1 noisy and hard
    to compare between runs.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[LABEL_COL],
    )

    print(f"\nTrain size: {len(train_df):,} | Test size: {len(test_df):,} "
          f"({100 * (1 - test_size):.0f}/{100 * test_size:.0f} split, stratified)")

    return train_df, test_df


def show_split_samples(train_df: pd.DataFrame, test_df: pd.DataFrame,
                       n: int = 5) -> None:
    """
    Displays the first n rows of the train and test sets separately - required
    by the assignment guidelines.

    The free-text columns are truncated for display only, so that the table
    stays readable in a terminal instead of every column being elided to "...".
    """
    def head_view(df: pd.DataFrame) -> pd.DataFrame:
        view = df[TEXT_COLS + [LABEL_COL]].head(n).reset_index(drop=True).copy()
        for col in TEXT_COLS:
            view[col] = (view[col].fillna("").str.slice(0, 55) +
                         view[col].fillna("").str.len().gt(55).map({True: "...", False: ""}))
        return view

    print(f"=== Train Set: First {n} Rows ===")
    _show(head_view(train_df))

    print(f"\n=== Test Set: First {n} Rows ===")
    _show(head_view(test_df))


def verify_stratification(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Confirms the split preserved class proportions, by comparing each class's
    share in train vs. test. Evidence that the stratification actually worked
    rather than an assumption that it did.
    """
    train_share = 100 * train_df[LABEL_COL].value_counts(normalize=True)
    test_share = 100 * test_df[LABEL_COL].value_counts(normalize=True)

    comparison = pd.DataFrame({
        "train_%": train_share.round(2),
        "test_%": test_share.reindex(train_share.index).round(2),
    })
    comparison["diff"] = (comparison["train_%"] - comparison["test_%"]).abs().round(3)

    print(f"Largest class-share difference between train and test: "
          f"{comparison['diff'].max():.3f} percentage points")

    return comparison.reset_index(names="category")


# --------------------------------------------------------------------------
#  Convenience wrapper
# --------------------------------------------------------------------------

def load_and_prepare(path: str = DATA_PATH, n_classes: int = N_CLASSES,
                     merge_duplicates: bool = False, test_size: float = 0.2,
                     random_state: int = 42):
    """
    Runs the full Part 1 pipeline: load -> (optionally merge) -> reduce -> split.

    `merge_duplicates` defaults to False so that the headline result stays
    directly comparable to the baseline; Part 6.b turns it on to quantify the
    effect of the labelling artifact.

    Returns: (train_df, test_df)
    """
    df = load_raw_data(path)
    print(f"Loaded {len(df):,} records with {df[LABEL_COL].nunique()} categories.")

    if merge_duplicates:
        df = merge_duplicate_categories(df)

    df = reduce_to_top_classes(df, n_classes)
    return split_train_test(df, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    import console_utf8  # noqa: F401  (switches the Windows console to UTF-8)

    print("=" * 78)
    print(" Part 1 - Data Loading, Overview, and Train/Test Split")
    print("=" * 78)

    df = load_raw_data()
    print(f"Loaded {len(df):,} records with {df[LABEL_COL].nunique()} categories "
          f"(before reduction).")

    df = reduce_to_top_classes(df)

    print("\n" + "-" * 78)
    describe_dataset(df)

    print("\n" + "-" * 78)
    train_df, test_df = split_train_test(df)

    print()
    show_split_samples(train_df, test_df, n=5)

    print("\n" + "-" * 78)
    print("Stratification check:")
    _show(verify_stratification(train_df, test_df).head(5))
