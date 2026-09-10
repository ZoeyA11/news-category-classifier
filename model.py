"""
================================================================================
 Part 3 - Learning Algorithm Implementation (Multinomial Naive Bayes)
================================================================================

Wraps scikit-learn's MultinomialNB so hyperparameters can be swapped easily.

Hyperparameters exposed:
  - alpha     : Laplace/Lidstone smoothing. Without it, a word never seen in a
                class during training would get probability exactly 0 - and
                because Naive Bayes multiplies probabilities, a single zero
                wipes out the whole score for that class.
  - fit_prior : whether to learn the classes' prior probability from the data,
                or assume a uniform distribution. This matters a great deal on
                imbalanced data - see Part 6.a.

Why Naive Bayes suits this task: the input is a high-dimensional sparse
bag-of-words vector (20,000 features). The independence assumption between
features given the class is plainly wrong for natural language, yet the model
still ranks classes well, trains in seconds, resists overfitting in high
dimensions, and - most usefully here - is completely transparent, which is what
makes the explainability work in Part 6.c possible without SHAP or LIME.
================================================================================
"""

import os

import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import MultinomialNB

DEFAULT_MODEL_PATH = "models/naive_bayes_bundle.joblib"


class TextNaiveBayes(BaseEstimator, ClassifierMixin):
    """
    A thin wrapper around MultinomialNB, in order to:
      - make hyperparameters easy to vary
      - stay compatible with the scikit-learn API (Pipeline, GridSearchCV,
        cross_val_score)

    Important note on scikit-learn compatibility:
    by convention, `__init__` must *only* store the parameters exactly as
    received - it must not build objects or transform values. The reason is that
    `cross_val_score` and `GridSearchCV` call `clone()`, which reconstructs the
    estimator through `get_params()` and expects the values to come back
    unchanged. Building the underlying model therefore happens in `fit()`.
    """

    def __init__(self, alpha: float = 1.0, fit_prior: bool = True):
        self.alpha = alpha
        self.fit_prior = fit_prior

    def fit(self, X, y):
        self.model_ = MultinomialNB(alpha=self.alpha, fit_prior=self.fit_prior)
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    @property
    def feature_log_prob_(self):
        """
        Exposes the per-class feature log-probabilities. These are the model's
        entire set of learned parameters, and they are what Part 6.c reads to
        explain the model's decisions.
        """
        return self.model_.feature_log_prob_


def save_bundle(model, vectorizer, path: str = DEFAULT_MODEL_PATH) -> str:
    """
    Persists the trained model together with its fitted vectorizer.

    Both are saved as one bundle deliberately. A Naive Bayes model is useless
    without the exact vectorizer it was trained on: the feature indices are
    positional, so pairing the model with a differently-fitted vectorizer would
    silently produce garbage predictions rather than raising an error.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump({"model": model, "vectorizer": vectorizer}, path)
    print(f"Model bundle saved to: {path} "
          f"({os.path.getsize(path) / 1e6:.2f} MB)")
    return path


def load_bundle(path: str = DEFAULT_MODEL_PATH):
    """Loads a saved bundle. Returns (model, vectorizer)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No model bundle at '{path}'. Run 'python src/main.py' first."
        )
    bundle = joblib.load(path)
    print(f"Loaded model bundle from: {path}")
    return bundle["model"], bundle["vectorizer"]


if __name__ == "__main__":
    import console_utf8  # noqa: F401  (switches the Windows console to UTF-8)

    # Quick self-check on synthetic data.
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.integers(0, 5, size=(100, 20))   # MultinomialNB requires non-negative input
    y = rng.integers(0, 3, size=100)

    clf = TextNaiveBayes(alpha=0.5).fit(X, y)
    print("Sample predictions:", clf.predict(X[:5]))
    print("Classes           :", clf.classes_)
