"""
================================================================================
 Part 2 - Feature Engineering & Preprocessing
================================================================================

Turns the news text (headline + short description) into a numeric TF-IDF vector.

--------------------------------------------------------------------------------
 Transformation pipeline
--------------------------------------------------------------------------------
 1. Field concatenation : headline + short_description are joined into a single
                          text field. They carry complementary information - the
                          headline is short and focused, the description adds
                          context - so combining them improves classification.
 2. Text normalization  : lowercase, strip every non-letter character, collapse
                          repeated whitespace.
 3. TF-IDF vectorization: term frequency weighted by inverse document frequency,
                          so a word scores highly when it is frequent in *this*
                          document but rare across the corpus. Those are exactly
                          the words that discriminate between categories.

--------------------------------------------------------------------------------
 The critical methodological rule
--------------------------------------------------------------------------------
 The vectorizer is fitted on the TRAINING SET ONLY. The test set is transformed,
 never re-fitted. Fitting on both would let the vocabulary and the IDF weights
 "see" the test data, which is data leakage and would inflate the reported
 scores into something meaningless.
================================================================================
"""

import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Name of the combined text column this module builds.
TEXT_COL = "text"


def clean_text(text: str) -> str:
    """
    Normalizes a single news text string:
      - lowercases it
      - removes every character that is not a letter (digits, punctuation,
        typographic quotes, emoji)
      - collapses repeated whitespace

    Note on dropping digits: in news headlines, numbers are usually dates or
    figures specific to one individual article, which do not generalize to other
    articles. Removing them reduces overfitting to noise.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def row_to_text(row) -> str:
    """Joins the headline and the short description into one cleaned text field."""
    headline = clean_text(row.get("headline", ""))
    description = clean_text(row.get("short_description", ""))
    return f"{headline} {description}".strip()


def build_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the cleaned text column to a DataFrame, without mutating the input."""
    df = df.copy()
    df[TEXT_COL] = df.apply(row_to_text, axis=1)
    return df


def build_tfidf_features(train_df: pd.DataFrame, test_df: pd.DataFrame,
                         max_features: int = 20000, ngram_range=(1, 2),
                         min_df: int = 2):
    """
    Builds the TF-IDF matrices: fit on train only, transform both.

    Returns: X_train, X_test, vectorizer, train_df, test_df

    Design choices:
      - stop_words="english": news headlines are full of function words
        ("the", "is", "of") that carry no information about the category.
      - ngram_range=(1, 2): bigrams capture meaningful multi-word units such as
        "donald trump" or "tom brady", which unigrams would split apart.
      - min_df=2: drops terms occurring in only one document across the whole
        corpus (mostly proper nouns and typos), shrinking the vocabulary
        substantially at no cost in signal.
      - sublinear_tf=True: dampens the effect of a term repeating many times
        within a single document.
    """
    train_df = build_text_column(train_df)
    test_df = build_text_column(test_df)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words="english",
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_df[TEXT_COL])   # fit ONLY on train
    X_test = vectorizer.transform(test_df[TEXT_COL])         # transform only

    vocab_size = len(vectorizer.get_feature_names_out())
    print(f"Vocabulary size (features): {vocab_size:,}"
          + ("   <-- max_features cap is binding" if vocab_size == max_features else ""))

    return X_train, X_test, vectorizer, train_df, test_df


def show_before_after_examples(df: pd.DataFrame, n: int = 3) -> None:
    """
    Shows n before/after feature engineering examples - required by Part 2.
    """
    sample = df.sample(n=n, random_state=1)
    for _, row in sample.iterrows():
        print("BEFORE (headline)   :", row["headline"])
        print("BEFORE (description):", row["short_description"])
        print("AFTER  (combined)   :", row_to_text(row))
        print("-" * 78)


def describe_matrix(X_train, X_test, vectorizer) -> None:
    """Prints the shape and sparsity of the resulting feature matrices."""
    density = X_train.nnz / (X_train.shape[0] * X_train.shape[1])
    print(f"Train matrix shape: {X_train.shape}")
    print(f"Test matrix shape : {X_test.shape}")
    print(f"Matrix density    : {density:.5f}  "
          f"({100 * (1 - density):.2f}% of cells are zero - a very sparse matrix)")
    print(f"\nExample features: {list(vectorizer.get_feature_names_out()[5000:5015])}")


if __name__ == "__main__":
    import console_utf8  # noqa: F401  (switches the Windows console to UTF-8)

    from data_loader import load_raw_data, reduce_to_top_classes, split_train_test

    print("=" * 78)
    print(" Part 2 - Feature Engineering")
    print("=" * 78)

    df = reduce_to_top_classes(load_raw_data())
    train_df, test_df = split_train_test(df)

    print("\n=== Feature Engineering Examples (train set) ===")
    show_before_after_examples(train_df)

    X_train, X_test, vectorizer, train_df, test_df = build_tfidf_features(train_df, test_df)
    print()
    describe_matrix(X_train, X_test, vectorizer)
