# סיווג קטגוריות חדשות מטקסט — NLP + Naive Bayes

מטלה בלמידת מכונה. המשימה: לחזות את קטגוריית המדור של כתבת חדשות על סמך
הכותרת והתקציר שלה בלבד.

| | |
|---|---|
| **Assignment Type** | NLP |
| **Learning Type** | Classification (multi-class, 20 מחלקות) |
| **Algorithm** | Multinomial Naive Bayes |
| **Dataset** | [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) (HuffPost) — 209,527 כותרות, 42 קטגוריות |
| **מדד ההצלחה** | Macro-Average F1 |

## התוצאה בשורה אחת

**Macro-F1 = 0.6105** על ה-test set (33,212 כותרות, 20 מחלקות), לעומת 0.4888
ב-baseline — שיפור של 25% שהושג כולו מכיול hyperparameters.

| מודל | Accuracy | **Macro-F1** |
|---|---|---|
| Baseline (`alpha=1.0`) | 0.6354 | 0.4888 |
| **Tuned (`alpha=0.5, fit_prior=False`)** | **0.6741** | **0.6105** |
| Balanced (oversampling) | 0.6421 | 0.5910 |
| אותו מודל, כפילויות תיוג ממוזגות בהערכה | 0.7165 | 0.6580 |
| מיזוג תיוג לפני האימון | 0.6977 | 0.6418 |

## הייחוס ל-Dataset

הרישיון הוא **CC BY 4.0**, כלומר ייחוס הוא תנאי של הרישיון ולא נימוס:

> Misra, Rishabh. "News Category Dataset." *arXiv preprint arXiv:2209.11429* (2022).
>
> Misra, Rishabh and Jigyasa Grover. "Sculpting Data for ML: The first act of Machine Learning." (2021).

## התקנה והרצה

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  ב-Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python src/download_data.py     # מוריד את ה-dataset אל data/
python src/main.py              # מריץ את כל ה-flow מקצה לקצה
python src/experiment_merge.py  # ניסוי מיזוג התיוג (חלק 6.ב)
```

כל מודול ב-`src/` רץ גם באופן עצמאי, כדי לבדוק שלב בודד בלי להריץ את הכול:

```bash
python src/data_loader.py          # חלק 1 בלבד
python src/feature_engineering.py  # חלק 2 בלבד
```

ל-notebook:

```bash
python -m ipykernel install --user --name news-category-nlp \
       --display-name "Python (news-category-nlp)"
jupyter notebook notebooks/analysis.ipynb
```

ה-notebook כבר מצביע על ה-kernel בשם `news-category-nlp`, כך שהוא נפתח
ורץ בלי לבקש לבחור סביבה. אם ה-kernel לא רשום עדיין, הריצו את הפקודה
הראשונה פעם אחת.

### שתי נקודות סביבה ששוות לדעת מראש

**הורדה מ-Kaggle לא דורשת credentials.** נקודת הקצה הציבורית להורדת
*datasets* ציבוריים עובדת בלי API token ובלי אישור תקנון. שווה לדעת
שההתנהגות הזאת ייחודית ל-datasets: הורדת נתונים של **תחרות** כן דורשת
גם טוקן וגם אישור תקנון התחרות באתר, ולכן היא לא ניתנת לאוטומציה מלאה.

**רשת עם TLS inspection.** אם הבקשה ל-Kaggle נכשלת ב-`CERTIFICATE_VERIFY_FAILED`,
הסיבה היא שפרוקסי ארגוני חותם מחדש את שרשרת התעודות ב-CA שחבילת `certifi`
של Python לא מכירה. הפרויקט משתמש ב-`truststore`, שמפנה את האימות ל-certificate
store של המערכת (שבו ה-CA הארגוני מותקן). זה **שומר** על אימות התעודות —
אין להשתמש ב-`verify=False`, שהיה מבטל את ההגנה מפני man-in-the-middle
במקום לפתור את הבעיה.

## מבנה הפרויקט

```
llm_task/
├── README.md
├── requirements.txt
├── data/
│   └── news_category/
│       └── News_Category_Dataset_v3.json   <- נוצר ע"י download_data.py
├── src/
│   ├── download_data.py        # הורדת ה-dataset מ-Kaggle
│   ├── console_utf8.py         # תיקון קידוד פלט ב-Windows
│   ├── data_loader.py          # חלק 1 — טעינה, סקירה, train/test split, מיזוג תיוג
│   ├── feature_engineering.py  # חלק 2 — ניקוי טקסט + TF-IDF
│   ├── model.py                # חלק 3 — עוטף MultinomialNB
│   ├── evaluate.py             # חלק 5 — Macro-F1, report, confusion matrix
│   ├── tuning.py               # חלק 6.א — grid-search + StratifiedKFold
│   ├── extensions.py           # חלק 6.ב + 6.ג — imbalance ו-explainability
│   ├── experiment_merge.py     # חלק 6.ב — ניסוי מיזוג קטגוריות כפולות
│   ├── viz.py                  # פלטה וסטייל משותפים לכל הגרפים
│   └── main.py                 # מריץ את כל ה-flow
├── notebooks/
│   ├── analysis.ipynb          # ה-notebook המלא עם כל הגרפים (להצגה)
│   └── analysis.pct.py         # מקור ה-notebook בפורמט jupytext (ידידותי ל-diff)
└── outputs/                    # גרפים, טבלאות CSV, ולוג הריצה
```

## התאמה למבנה המטלה

| חלק במטלה | מימוש | תוצאה |
|---|---|---|
| 1. הקדמה, טעינה, train/test split | `data_loader.py` | 209,527 → 166,057 שורות, 20 מחלקות, חלוקה 80/20 מסטרטפת |
| 2. Feature Engineering | `feature_engineering.py` | TF-IDF, 20,000 features, `fit` על train בלבד |
| 3. מימוש Naive Bayes | `model.py` | `MultinomialNB` עם `alpha` ו-`fit_prior` |
| 4. אימון עם flow לפרמטרים שונים | `main.py` | baseline / tuned / balanced |
| 5. הערכה על test set | `evaluate.py` | **Macro-F1 = 0.6105**, accuracy = 0.6741 |
| 6.א. grid-search + k-fold | `tuning.py` | 2 hyperparameters, 14 שילובים × 5 folds |
| 6.ב. imbalanced data | `extensions.py`, `experiment_merge.py` | oversampling + מיזוג תיוג |
| 6.ג. explainability | `extensions.py` | log-odds per class + הסבר לתחזית בודדת |

## החלטות תכנון והנימוק להן

### למה Macro-F1 ולא Accuracy

ה-data לא מאוזן ביחס של כ-10x (POLITICS: 28,481 מול IMPACT: 2,787 ב-train).
Accuracy "מתגמל" מודל שמנחש טוב רק את המחלקה הגדולה ומתעלם מהקטנות.
Macro-F1 נותן משקל שווה לכל מחלקה ללא תלות בגודלה, ולכן הוא המדד שמודד
את מה שבאמת מעניין כאן.

### למה צמצום ל-20 מחלקות

ה-dataset המקורי מכיל 42 קטגוריות. הצמצום ל-20 הגדולות מיישר קו עם מפרט
המטלה, ובנוסף מחלקות הזנב מכילות פחות מ-1,100 דוגמאות כל אחת — מעט מדי
ללמידה אמינה.

### למה StratifiedKFold ולא KFold

עם 20 מחלקות לא מאוזנות, חלוקה רגילה עלולה לייצר fold שבו מחלקה נדירה
נעדרת לחלוטין. במקרה כזה macro-F1 מחושב על מחלקה בלי דוגמאות — מה שמייצר
אזהרות ומטה את המדד כלפי מטה.

### למה `fit`-ים רק על ה-train

ה-TF-IDF וה-grid search נלמדים על ה-train בלבד. ה-test אינו נוגע בתהליך
עד הבחירה הסופית. התיקוף שזה עבד: ה-CV חזה macro-F1 של 0.6077, ובפועל
יצא 0.6105 — הפרש של 0.003, כלומר הכיול הכליל ולא התאים את עצמו ל-train.

## הממצאים המרכזיים

### 1. כל הרווח בא מ-`fit_prior=False`

ב-grid search, **כל השילובים המובילים הם `fit_prior=False`**, והגרועים
ביותר הם `fit_prior=True` עם alpha גבוה (macro-F1 של 0.378 בלבד).

ההסבר: `fit_prior=True` לומד את ההסתברות האפריורית מהנתונים, וכיוון שהם
לא מאוזנים, המודל מוטה מראש לטובת POLITICS ומוחק את המחלקות הקטנות —
בדיוק מה ש-macro-F1 מעניש עליו. `fit_prior=False` מניח התפלגות אחידה
ולכן מתייחס לכל המחלקות באופן שווה. **זהו כבר טיפול בחוסר איזון, דרך
ה-hyperparameter ולא דרך הנתונים.**

### 2. ה-oversampling הזיק, וזה מסביר את עצמו

Macro-F1 ירד מ-0.6105 ל-0.5910. הסיבה: `fit_prior=False` כבר מנטרל את
חוסר האיזון, ולכן ה-oversampling לא מוסיף אינפורמציה — הוא רק משכפל
שורות קיימות (132,845 → 569,620, פי 4.3) ומנפח את זמן האימון.

המסקנה: כשמטפלים באותה בעיה פעמיים בשתי דרכים, השנייה לא מוסיפה. שווה
לבדוק *למה* שיטה אמורה לעזור לפני שמפעילים אותה.

### 3. תקרת הביצועים כאן היא איכות התיוג, לא המודל

שתי המחלקות הגרועות בפער גדול היו `PARENTS` (F1≈0.30) ו-`HEALTHY LIVING`
(F1≈0.29) — בדיוק הכפילויות הסמנטיות של `PARENTING` ו-`WELLNESS`.
HuffPost שינו שמות מדורים לאורך השנים, ולכן אותו נושא מופיע תחת שני שמות.

המספרים חד-משמעיים:

```
אמת = PARENTS          → נובא נכון 24.5%  |  נובא כתאום 'PARENTING' 48.8%
אמת = HEALTHY LIVING   → נובא נכון 23.7%  |  נובא כתאום 'WELLNESS'  47.9%
```

**המודל מנבא את התאום פי שניים יותר מהתווית ה"נכונה".** זו אינה שגיאה של
המודל — הוא מזהה את הנושא נכון, והתיוג הוא זה ששרירותי. מיזוג הזוגות
בהערכה בלבד, **בלי אימון מחדש**, מעלה את ה-macro-F1 ל-0.6580. מיזוג לפני
האימון נותן 0.6418 ובנוסף שומר 85.1% מהנתונים במקום 79.3%.

אחרי המיזוג, המחלקות הגרועות הן `IMPACT` ו-`WOMEN` — קטגוריות מערכתיות
מעורפלות שיכולות להכיל כמעט כל נושא. גם הן מגבלת תיוג ולא מגבלת מודל.

### 4. דירוג ה-explainability חייב להיות דיסקרימינטיבי

דירוג לפי `feature_log_prob_` גולמי מחזיר את המילים ה*נפוצות* בכל מחלקה,
ולכן אותן מילים גנריות עולות לראש הרשימה בכל מחלקה ולא מסבירות כלום.
המימוש כאן משתמש ב-log-odds — `log P(w|c)` פחות הממוצע על שאר המחלקות —
שמבודד את מה שמייחד את המחלקה:

| קטגוריה | המילים המייחדות |
|---|---|
| POLITICS | `gop`, `trump`, `republican`, `donald`, `clinton` |
| SPORTS | `nba`, `nfl`, `player`, `football`, `brady` |
| FOOD & DRINK | `recipes`, `recipe`, `butter`, `kitchen`, `cheese` |
| TRAVEL | `travel`, `hotel`, `destinations`, `vacation`, `hotels` |

זה אימות איכותי שהמודל לומד סיגנל אמיתי ולא ארטיפקטים של הקורפוס.

## תוצרים ב-`outputs/`

| קובץ | תוכן |
|---|---|
| `class_distribution.png` | התפלגות 20 המחלקות (חוסר האיזון) |
| `grid_search.png` | macro-F1 כפונקציה של alpha, קו לכל `fit_prior` |
| `confusion_matrix.png` | confusion matrix מנורמל לפי שורה |
| `per_class_f1.png` | F1 לכל מחלקה, ממוין |
| `model_comparison.png` | accuracy מול macro-F1 לשלושת המודלים |
| `top_features.png` | המילים המייחדות לכל קטגוריה (small multiples) |
| `grid_search_results.csv` | טבלת ה-grid search המלאה |
| `model_comparison.csv` | טבלת ההשוואה |
| `per_class_f1.csv` | F1 ותמיכה לכל מחלקה |
| `merge_experiment.csv` | תוצאות ניסוי מיזוג התיוג |
| `full_run.log` | הפלט המלא של `main.py` |

## מה היה הצעד הבא

- `max_features=20000` הוא **חסם פעיל** — המילון הגיע בדיוק לתקרה, כלומר
  התקרה חוסמת features שהיו יכולים לעזור. שווה לסרוק גם אותו ב-grid.
- `ComplementNB` — וריאנט של Naive Bayes שתוכנן במיוחד ל-data לא מאוזן.
- השוואה מול Linear SVM או Logistic Regression כ-baseline חזק יותר.
