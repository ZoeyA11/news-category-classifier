# Original Prompt (English translation)

English translation of the prompts used to build this project with Claude.
The original prompts were written in Hebrew; this is a faithful translation,
not a cleaned-up rewrite.

---

## Prompt 1 — the initial request

> I'm working on an assignment for a machine learning course. I already have a
> project skeleton ready in this folder (`cuisine_classification`) that was built
> ahead of time in another chat with Claude. I need you to continue with me and
> actually build and run it.
>
> ### Assignment details
> - **Assignment Type:** NLP
> - **Learning Type:** Classification (multi-class, 20 classes)
> - **Algorithm:** Multinomial Naive Bayes
> - **Dataset:** "What's Cooking?" (Yummly) — predicting the type of cuisine from
>   a list of ingredients
>   https://www.kaggle.com/competitions/whats-cooking-kernels-only
>
> ### Existing project structure
> - `src/data_loader.py` — loading `train.json`, overview, train/test split
> - `src/feature_engineering.py` — cleaning the ingredient text + TF-IDF (fit on train only)
> - `src/model.py` — Naive Bayes wrapper
> - `src/tuning.py` — grid-search + 5-fold cross validation over alpha
> - `src/evaluate.py` — Macro-F1, classification report, confusion matrix
> - `src/extensions.py` — handling imbalanced data (oversampling) + explainability
> - `src/main.py` — runs everything end to end
> - `README.md` — includes a full explanation of the project and a mapping to the
>   assignment sections
>
> ⚠️ **Important:** this code was written without actually being tested (it was
> written in an environment with no internet), so it may contain minor errors that
> will need fixing once we run it on real data.
>
> ### What I need from you now, in this order:
> 1. Help me download `train.json` from Kaggle (competition:
>    `whats-cooking-kernels-only`) and put it in the `data/` folder.
> 2. Run `src/main.py` and fix every error that comes up (imports, paths, library
>    versions, etc.).
> 3. Verify that the results are reasonable (Accuracy, Macro-F1) and show them to me.
> 4. Help me complete and improve section 6 (the bonus sections):
>    - **6.a:** grid-search + k-fold — verify that the alpha grid runs correctly, and
>      expand it if needed
>    - **6.b:** imbalanced data — run the oversampling and show an
>      improvement/comparison
>    - **6.c:** explainability — show the most influential ingredients for a few
>      cuisines, and maybe also a nice chart/visualization for presentation
> 5. At the end, help me create a final, well-organized README + a Jupyter notebook
>    with all the charts and results, ready to present in the assignment video.
>
> Let's start with step 1.

---

## Prompt 2 — switching the dataset

> You need to find me a dataset from this link
> https://www.kaggle.com/datasets?tags=13204-NLP

---

## Prompt 3 — choosing the alternative

> I'm choosing option B. Before we continue, send me the full link to this dataset.

*(Option B was `rmisra/news-category-dataset` — the News Category Dataset from
HuffPost. Option A was a re-upload of the original What's Cooking data, which is
not tagged `nlp` on Kaggle and therefore does not appear on the NLP tag page.)*

---

## Prompt 4 — adapting the project

> Adapt the whole project to the chosen dataset
> https://www.kaggle.com/datasets/rmisra/news-category-dataset

---

## Prompt 5 — running it

> Run it for me, for the video?

---

## Prompt 6 — preparing to present

> I need to record a video about the project. What I need from you is to teach me
> the project from scratch — what I need to know, what the next step is, what
> exactly to go over in the README, and what to talk about for five minutes in
> this video.

---

## Note on the dataset change

Prompt 1 specifies the **What's Cooking** dataset (predicting cuisine from
ingredients). Prompts 2–4 changed direction to the **News Category Dataset**
(predicting a news section from a headline), because the course required a dataset
carrying Kaggle's `NLP` tag — and What's Cooking is a *competition* rather than a
tagged dataset, so it does not appear on that tag page.

The assignment parameters themselves did not change: NLP, multi-class
classification with 20 classes, and Multinomial Naive Bayes. What changed was the
data layer (`data_loader.py` and `feature_engineering.py`); the model, tuning,
evaluation, and extension modules carried over.

---

## Reusable version of this prompt

A generalized version of Prompt 1, for a future assignment — with the
dataset-specific details left as placeholders:

> I'm working on an assignment for a machine learning course. I already have a
> project skeleton in this folder. I need you to continue with me and actually
> build and run it.
>
> ### Assignment details
> - **Assignment Type:** `<NLP / vision / tabular>`
> - **Learning Type:** `<classification / regression>` (`<binary / multi-class, N classes>`)
> - **Algorithm:** `<algorithm>`
> - **Dataset:** `<name>` — `<one-line description of the prediction task>`
>   `<URL>`
>
> The dataset must carry Kaggle's `<required tag>` tag.
>
> ### Existing project structure
> `<list the files and what each one does>`
>
> ⚠️ **Important:** this code was written without actually being tested, so it may
> contain errors that need fixing once we run it on real data.
>
> ### What I need from you now, in this order:
> 1. Help me download the data and put it in `data/`.
> 2. Run `src/main.py` and fix every error that comes up (imports, paths, library
>    versions, encoding, etc.).
> 3. Verify that the results are reasonable and show me the metrics.
> 4. Help me complete and improve the bonus sections:
>    - grid-search + k-fold cross validation, over more than one hyperparameter
>    - handling imbalanced data, with a before/after comparison
>    - explainability — the most influential features per class, plus a chart
> 5. At the end, produce a final README + a Jupyter notebook with all charts and
>    results embedded, ready to present in a video.
>
> Tell me if you find a problem with the approach rather than working around it
> silently, and flag any result that looks wrong even if it looks good on paper.
>
> Let's start with step 1.
