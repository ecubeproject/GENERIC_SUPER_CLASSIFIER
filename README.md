# GENERIC_SUPER_CLASSIFIER – Interactive Multi-Model Auto-Classification Tool

**Organization:** Ecube Analytics
**Interface:** Streamlit web app
**Live demo:** https://appapppy-fyndt7lkblfki6pakz7i6x.streamlit.app/

---

## Project Overview

GENERIC_SUPER_CLASSIFIER is a machine-learning application designed for users who want to train and evaluate classification models **without writing any code**. With intuitive UI controls, it enables end-to-end workflows from data upload to model selection, training, evaluation, and visualization — all in one place.

The modelling and plotting engine is a set of **UI-free, independently importable modules** (`data_io`, `profiling`, `pipeline`, `plots`); the Streamlit layer is only widgets and layout, so the same engine runs from a notebook, a script, or the test suite.

**Key Features**:
- Accepts `.csv` and `.xlsx` datasets of variable size (hardware limitations apply).
- Automatically performs **Exploratory Data Analysis (EDA)**, including data types, missing values, and summary statistics.
- User-driven selection of:
  - Target variable (binary or multi-class classification)
  - One of **13 classification algorithms**, with dynamic, algorithm-specific hyperparameter controls
- Model evaluation metrics and classification reports available with a single button click.
- **Feature-column selection** — exclude ID / leakage columns from training.
- **Class-imbalance correction** — one toggle applies `class_weight='balanced'` (or a balanced `sample_weight` where the estimator needs it).
- **K-fold cross-validation** — stratified CV with a mean ± std summary of accuracy, macro precision / recall / F1 and ROC-AUC.
- **Model leaderboard** — every train / CV run is added to a ranked table so runs can be compared at a glance.
- **Save / load models and score new data** — persist a trained pipeline to a `.joblib` file and batch-predict on a fresh `.csv` / `.xlsx`.
- Includes **8 visualization options**; each plot includes interpretive guidance to aid decision-making.
- Built-in preprocessing pipeline covering:
  - Missing-value imputation
  - Categorical encoding
  - Feature scaling
  - Datetime-column expansion (year / month / day / day-of-week)

Supported classifiers include:
`Random Forest`, `SVC`, `KNN`, `XGBoost`, `AdaBoost`, `HistGradientBoosting`, `Logistic Regression`, `Decision Tree`, `Gradient Boosting`, `LightGBM`, `Gaussian Naive Bayes`, `Bernoulli Naive Bayes`, and a simple **Neural Network (MLP)**.

Available visualizations: ROC Curve, Precision-Recall Curve, PCA Explained Variance, PCA 2D Projection, Cumulative Gains Curve, Calibration Plot, Lift Curve, and Silhouette Analysis.

---

## The eight diagnostic plots

Each plot answers a different question about the trained model. The app shows a
short version of this text next to every chart.

**ROC Curve** — Plots true-positive rate against false-positive rate at every
threshold. The area under it (AUC) is a single, threshold-independent score for
how well the model *ranks* positives above negatives (1.0 perfect, 0.5 random).
Best for comparing models when the classes are reasonably balanced.

**Precision–Recall Curve** — Traces precision against recall as the threshold
sweeps; the area is the average precision (AP). This is the more honest view when
the positive class is rare, because — unlike ROC — it is not flattered by a
large, easy pool of true negatives.

**PCA Explained Variance** — Shows how much of the total variance in the fully
preprocessed feature matrix is captured as principal components are added one by
one. A curve that saturates quickly means the features are highly redundant and
the problem is effectively low-dimensional.

**PCA 2D Projection** — Projects every row onto the first two principal
components and colours it by class. A quick eyeball test of separability: cleanly
grouped colours suggest an easy problem; heavy overlap warns that no classifier
will do well without better features.

**Cumulative Gains Curve** — For each fraction of the sample worked through in
descending score order, shows what share of all actual positives has been
captured. It answers the operational question "if we can only act on the top
X%, how many real cases do we catch?"

**Calibration Plot** — Bins predictions by their stated probability and plots
predicted vs. observed frequency. Points on the diagonal mean the probabilities
are trustworthy (a "0.7" really happens ~70% of the time); bowing above or below
exposes systematic over- or under-confidence.

**Lift Curve** — Expresses how many times better than random the model is at
surfacing a class as you widen the fraction of the sample you target. Same
information as the gains curve, framed the way campaign and risk teams usually
think about it.

**Silhouette Analysis** — Runs k-means for several cluster counts on the
preprocessed features and plots the silhouette score for each. It hints at
whether the data has natural group structure and roughly how many groups —
useful context alongside the supervised model.

---

## Outcome & Impact

- Provides a **fully functional web UI** for ML experimentation, model comparison, and visualization.
- Ideal for:
  - In-house model benchmarking
  - Teaching ML concepts in workshops
  - Rapid prototyping of classification use-cases
- Saves time and technical overhead by handling preprocessing, training, and plotting in a clean, user-centric interface.

---

## Tech Stack

| Layer               | Technologies Used                             |
|---------------------|-----------------------------------------------|
| **Language**         | Python 3.x with Pandas, NumPy                |
| **ML Framework**     | Scikit-learn pipelines, XGBoost, LightGBM    |
| **Algorithms**       | 13 classifiers (see Features section)         |
| **Preprocessing**    | ColumnTransformer, Imputers, Scalers         |
| **Interface**        | Streamlit                                     |
| **Visualization**    | Matplotlib                                    |
| **Data Input**       | `.csv` and `.xlsx` file handling              |

---

## Folder Structure

```plaintext
.
├── code/
│   ├── streamlit_app.py        # The Streamlit UI (widgets, layout, session state)
│   ├── Generic_classifier.py   # Legacy launcher — runs `streamlit run streamlit_app.py`
│   ├── data_io.py              # Load CSV / XLSX
│   ├── profiling.py            # Text EDA summary of a DataFrame
│   ├── pipeline.py             # Spec → fitted sklearn pipeline + evaluation (no UI)
│   ├── plots.py                # Diagnostic plots as pure (ax, …) functions (no UI)
│   └── tests/                  # pytest suite (run: python -m pytest)
├── datafiles/                  # Sample datasets to try the app with
│   ├── IRIS.csv
│   ├── IRIS.xlsx
│   └── diabetes.csv
├── .streamlit/config.toml      # Theme
├── pytest.ini
├── requirements.txt            # Project dependencies
├── LICENSE
└── README.md                   # This documentation
```

The modelling and plotting code (`data_io`, `profiling`, `pipeline`, `plots`) is
UI-free and independently importable — usable from a notebook or a script with no
Streamlit dependency.

---

## How to Use?
1. Clone the repository: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Launch the application: `streamlit run code/streamlit_app.py` (or `python code/Generic_classifier.py`)
4. Using the app:
    - In the sidebar, pick a bundled sample dataset or upload your own (`.csv` / `.xlsx`)
    - Choose the target column; optionally narrow the feature columns (leaving all selected uses every column)
    - Optionally tick **Balance classes** for imbalanced targets and set the hold-out test size
    - Choose a classifier and adjust its hyperparameters
    - **Train & Evaluate** tab: run a hold-out evaluation or a stratified **Cross-validate** (set *k*) — both land on the leaderboard
    - **Plots** tab: pick one of the eight diagnostics and read its interpretation
    - **Leaderboard** tab: compare every run, sorted by any metric
    - **Predict & Model** tab: download the trained model as `.joblib`, re-upload one later, and batch-score a new file
5. Run the test suite (optional): `python -m pytest`

### Deploying to Streamlit Community Cloud

The [live demo](https://appapppy-fyndt7lkblfki6pakz7i6x.streamlit.app/) runs on
Streamlit Community Cloud. To deploy your own copy, point a new app at this repo
with **main file path** `code/streamlit_app.py` and Python **3.11**; the
`.streamlit/config.toml` theme, `requirements.txt` and `packages.txt` (`libgomp1`,
needed by LightGBM) are picked up automatically.

### Using the engine without the GUI

```python
from pipeline import train_and_evaluate, cross_validate_model, save_model, load_model, predict_dataframe

result = train_and_evaluate(df, target="churn", classifier_name="XGBoost",
                            params={"n_estimators": 200}, balance_classes=True)
cv = cross_validate_model(df, "churn", "XGBoost", {"n_estimators": 200}, k=5)
save_model(result, "model.joblib")
preds = predict_dataframe(load_model("model.joblib"), new_df)
```

---

## Roadmap / Planned Improvements

- Optional model explainability (permutation importance / SHAP) in the Plots tab.
- Regression mode for continuous targets (currently classification only).

---

## Contributors & Acknowledgments
1. Developer: [Tejas Desai]
2. LinkedIn: [https://www.linkedin.com/in/tejasddesaiindia/]

---

## License

This project is released under the MIT License.
