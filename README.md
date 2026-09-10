# News Category Classification (NLP + Naive Bayes)

## Student Information
- **Name:** Zoey A
- **Student ID (last 4 digits):** 1269

## Submission Info
| | |
|---|---|
| **Repository URL** | https://github.com/ZoeyA11/news-category-classifier |
| **AI Tool Disclosure** | See [`PROMPT.md`](./PROMPT.md) for the full prompt history used with the AI chatbot during this project |

## Assignment Parameters
| | |
|---|---|
| **Assignment Type** | NLP |
| **Learning Type** | Classification (multi-class, 20 classes) |
| **Algorithm** | Multinomial Naive Bayes |
| **Dataset** | [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) (HuffPost) — 209,527 headlines, 42 original categories |
| **Evaluation Metric** | Macro-Average F1 |

## AI Tools & Prompt Engineering
**AI Tool:** Claude (Opus) via Claude Code

**Purpose:** Architectural planning for the NLP pipeline, designing the label reduction and duplicate-category merging logic, and defining the multi-class evaluation strategy (Macro-Average F1).

**Key prompts used:**
- "Find me an NLP-tagged dataset on Kaggle suitable for multi-class text classification with Naive Bayes."
- "Design a TF-IDF + Multinomial Naive Bayes pipeline where the vectorizer is fitted on the training set only, with grid-search over more than one hyperparameter using stratified k-fold cross validation."
- "The oversampling made Macro-F1 worse — find out why instead of working around it."

The full prompt history is documented in [`PROMPT.md`](./PROMPT.md).

## Dataset Attribution
This dataset is licensed under **CC BY 4.0** — attribution is a license condition, not a courtesy:

> Misra, Rishabh. "News Category Dataset." *arXiv preprint arXiv:2209.11429* (2022).
>
> Misra, Rishabh and Jigyasa Grover. "Sculpting Data for ML: The first act of Machine Learning." (2021).

## Problem Description & Dataset Overview
This project addresses a supervised multi-class text classification task using the News Category Dataset: 209,527 HuffPost headlines published between 2012 and 2022. Given only the headline and its short description, the model predicts which editorial section the article belongs to.

Rather than classifying into the original 42 fine-grained sections, the targets are aggregated down to the **20 most frequent categories**. This reformulation is deliberate and serves three purposes:
- It aligns the task with the assignment specification (multi-class, 20 classes)
- It guarantees higher sample density per class (every retained class holds at least ~3,400 examples, whereas the discarded tail holds fewer than 1,100 each)
- It yields a more reliable Macro-Average F1, since that metric weights every class equally regardless of size

### Why Naive Bayes suits this task
The text is represented as a sparse, high-dimensional bag-of-words TF-IDF vector (20,000 features). Naive Bayes assumes the features are conditionally independent given the class — an assumption that is plainly wrong for natural language, since "trump" and "president" are obviously correlated. It nevertheless works well on text: it trains in seconds, resists overfitting in high dimensions, and — most importantly here — it is completely transparent. Every decision follows directly from `log P(word | class)`, so the explanation can be read straight out of the model's parameters, which is what makes Part 6.c possible without SHAP or LIME.

## Headline Result
**Macro-F1 = 0.6105** on the test set (33,212 headlines, 20 classes), versus 0.4888 for the baseline — a 25% improvement achieved entirely through hyperparameter tuning.

| Model | Accuracy | **Macro-F1** |
|---|---|---|
| Baseline (`alpha=1.0`) | 0.6354 | 0.4888 |
| **Tuned (`alpha=0.5, fit_prior=False`)** | **0.6741** | **0.6105** |
| Balanced (oversampling) | 0.6421 | 0.5910 |
| Same model, duplicate labels merged at evaluation | 0.7165 | 0.6580 |
| Duplicate labels merged before training | 0.6977 | 0.6418 |

## Installation & Usage
```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python src/download_data.py     # downloads the dataset into data/
python src/main.py              # runs the full pipeline end-to-end
python src/experiment_merge.py  # label-merging experiment (Part 6.b)
```

Each module in `src/` can also be run standalone, to test a single stage without running everything:
```bash
python src/data_loader.py          # Part 1 only
python src/feature_engineering.py  # Part 2 only
```

For the notebook:
```bash
python -m ipykernel install --user --name news-category-nlp \
       --display-name "Python (news-category-nlp)"
jupyter notebook notebooks/analysis.ipynb
```

The notebook already points to a kernel named `news-category-nlp`, so it opens and runs without prompting for an environment. If the kernel is not registered yet, run the command above once.

### Two environment notes worth knowing in advance
**Downloading from Kaggle requires no credentials.** The public endpoint for downloading public *datasets* works without an API token or accepting any terms. Note that this behavior is unique to datasets — downloading data from a **competition** does require both a token and accepting the competition's terms on the site, so it cannot be fully automated.

**Network with TLS inspection.** If the request to Kaggle fails with `CERTIFICATE_VERIFY_FAILED`, the reason is that a corporate proxy re-signs the certificate chain with a CA that Python's `certifi` bundle doesn't recognize. This project uses `truststore`, which routes verification through the system's certificate store (where the corporate CA is installed). This **preserves** certificate verification — do not use `verify=False`, which would disable protection against man-in-the-middle attacks instead of solving the problem.

## Mapping to the Assignment Structure
| Assignment Part | Implementation | Result |
|---|---|---|
| 1. Introduction, loading, train/test split | `data_loader.py` | 209,527 → 166,057 rows, 20 classes, 80/20 stratified split |
| 2. Feature Engineering | `feature_engineering.py` | TF-IDF, 20,000 features, `fit` on train only |
| 3. Naive Bayes implementation | `model.py` | `MultinomialNB` with `alpha` and `fit_prior` |
| 4. Training with flow over different hyperparameters | `main.py` | baseline / tuned / balanced |
| 5. Evaluation on the test set | `evaluate.py` | **Macro-F1 = 0.6105**, accuracy = 0.6741 |
| 6.a. Grid-search + k-fold | `tuning.py` | 2 hyperparameters, 14 combinations × 5 folds |
| 6.b. Imbalanced data | `extensions.py`, `experiment_merge.py` | oversampling + label merging |
| 6.c. Explainability | `extensions.py` | log-odds per class + single-prediction explanation |

## Design Decisions and Their Rationale

### Why Macro-F1 and not Accuracy
The data is imbalanced by roughly a 10x ratio (POLITICS: 28,481 vs. IMPACT: 2,787 in train). Accuracy "rewards" a model that only guesses the large class well and ignores the small ones. Macro-F1 gives equal weight to every class regardless of its size, so it measures what actually matters here.

### Why reduce to 20 classes
The original dataset contains 42 categories. Reducing to the 20 largest aligns with the assignment specification, and additionally the tail classes contain fewer than 1,100 examples each — too few for reliable learning.

### Why StratifiedKFold and not KFold
With 20 imbalanced classes, a plain split could produce a fold where a rare class is completely absent. In that case macro-F1 is computed over a class with no examples — which produces warnings and skews the metric downward.

### Why fitting only happens on the train set
The TF-IDF and grid search are learned on the train set only. The test set is not touched until the final selection is made. Validation that this worked: the CV predicted a macro-F1 of 0.6077, and the actual result was 0.6105 — a difference of 0.003, meaning the tuning generalized rather than overfitting to the train set.

## Key Findings

### 1. All the gain comes from `fit_prior=False`
In the grid search, **all the top-ranked combinations use `fit_prior=False`**, and the worst are `fit_prior=True` with high alpha (macro-F1 of only 0.378).

Explanation: `fit_prior=True` learns the prior probability from the data, and since it is imbalanced, the model is biased toward POLITICS from the start and erases the small classes — exactly what macro-F1 penalizes. `fit_prior=False` assumes a uniform distribution and therefore treats all classes equally. **This is itself a form of imbalance handling, done through the hyperparameter rather than the data.**

### 2. Oversampling hurt performance, and this is self-explanatory
Macro-F1 dropped from 0.6105 to 0.5910. The reason: `fit_prior=False` already neutralizes the imbalance, so oversampling adds no new information — it only duplicates existing rows (132,845 → 569,620, a 4.3x increase) and inflates training time.

The takeaway: when the same problem is addressed twice in two different ways, the second doesn't add anything. It's worth checking *why* a method should help before applying it.

### 3. The performance ceiling here is label quality, not the model
The two worst classes by a wide margin were `PARENTS` (F1≈0.30) and `HEALTHY LIVING` (F1≈0.29) — exactly the semantic duplicates of `PARENTING` and `WELLNESS`. HuffPost renamed sections over the years, so the same topic appears under two different names.

The numbers are unambiguous:
```
true = PARENTS          → correctly predicted 24.5%  |  predicted as twin 'PARENTING' 48.8%
true = HEALTHY LIVING   → correctly predicted 23.7%  |  predicted as twin 'WELLNESS'  47.9%
```

**The model predicts the twin twice as often as the "correct" label.** This is not a model error — it identifies the topic correctly, and the labeling is what's arbitrary. Merging the pairs at evaluation only, **without retraining**, raises macro-F1 to 0.6580. Merging before training gives 0.6418 and additionally retains 85.1% of the data instead of 79.3%.

After merging, the worst classes are `IMPACT` and `WOMEN` — vague, catch-all categories that can contain almost any topic. These, too, are a labeling limitation rather than a model limitation.

### 4. Explainability ranking must be discriminative
Ranking by raw `feature_log_prob_` returns the words most *frequent* in each class, so the same generic words rise to the top in every class and explain nothing. This implementation uses log-odds — `log P(w|c)` minus the average over the other classes — which isolates what actually distinguishes the class:

| Category | Distinctive Words |
|---|---|
| POLITICS | `gop`, `trump`, `republican`, `donald`, `clinton` |
| SPORTS | `nba`, `nfl`, `player`, `football`, `brady` |
| FOOD & DRINK | `recipes`, `recipe`, `butter`, `kitchen`, `cheese` |
| TRAVEL | `travel`, `hotel`, `destinations`, `vacation`, `hotels` |

This is a qualitative confirmation that the model learns real signal rather than corpus artifacts.

## Outputs in `outputs/`
| File | Content |
|---|---|
| `class_distribution.png` | Distribution of the 20 classes (the imbalance) |
| `grid_search.png` | Macro-F1 as a function of alpha, one line per `fit_prior` |
| `confusion_matrix.png` | Row-normalized confusion matrix |
| `per_class_f1.png` | F1 per class, sorted |
| `model_comparison.png` | Accuracy vs. macro-F1 for the three models |
| `top_features.png` | Distinctive words per category (small multiples) |
| `grid_search_results.csv` | Full grid search results table |
| `model_comparison.csv` | Model comparison table |
| `per_class_f1.csv` | F1 and support per class |
| `merge_experiment.csv` | Label-merging experiment results |
| `full_run.log` | Full output log of `main.py` |

## What the Next Step Would Be
- `max_features=20000` is an **active constraint** — the vocabulary hit exactly the cap, meaning it's blocking features that could help. Worth adding it to the grid too.
- `ComplementNB` — a Naive Bayes variant designed specifically for imbalanced data.
- A comparison against Linear SVM or Logistic Regression as a stronger baseline.
