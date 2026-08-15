# GENERIC_SUPER_CLASSIFIER – Interactive Multi-Model Auto-Classification Tool

**Organization:** Ecube Analytics
**Interface:** Desktop GUI (Python + Tkinter)

---

## Project Overview

GENERIC_SUPER_CLASSIFIER is a machine-learning application designed for users who want to train and evaluate classification models **without writing any code**. With intuitive UI controls, it enables end-to-end workflows from data upload to model selection, training, evaluation, and visualization — all in one place.

**Key Features**:
- Accepts `.csv` and `.xlsx` datasets of variable size (hardware limitations apply).
- Automatically performs **Exploratory Data Analysis (EDA)**, including data types, missing values, and summary statistics.
- User-driven selection of:
  - Target variable (binary or multi-class classification)
  - One of **13 classification algorithms**, with dynamic, algorithm-specific hyperparameter controls
- Model evaluation metrics and classification reports available with a single button click.
- Includes **8 visualization options**; each plot includes interpretive guidance to aid decision-making.
- Built-in preprocessing pipeline covering:
  - Missing-value imputation
  - Categorical encoding
  - Feature scaling

Supported classifiers include:
`Random Forest`, `SVC`, `KNN`, `XGBoost`, `AdaBoost`, `HistGradientBoosting`, `Logistic Regression`, `Decision Tree`, `Gradient Boosting`, `LightGBM`, `Gaussian Naive Bayes`, `Bernoulli Naive Bayes`, and a simple **Neural Network (MLP)**.

Available visualizations: ROC Curve, Precision-Recall Curve, PCA Explained Variance, PCA 2D Projection, Cumulative Gains Curve, Calibration Plot, Lift Curve, and Silhouette Analysis.

---

## Outcome & Impact

- Provides a **fully functional GUI** for ML experimentation, model comparison, and visualization.
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
| **Interface GUI**    | Tkinter (native Python UI)                    |
| **Visualization**    | Matplotlib                                    |
| **Data Input**       | `.csv` and `.xlsx` file handling              |

---

## Folder Structure

```plaintext
.
├── code/
│   └── Generic_classifier.py   # Main app — launch this to start the GUI
├── datafiles/                  # Sample datasets to try the app with
│   ├── IRIS.csv
│   ├── IRIS.xlsx
│   └── diabetes.csv
├── requirements.txt            # Project dependencies
├── LICENSE
└── README.md                   # This documentation
```

---

## How to Use?
1. Clone the repository: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Launch the application: `python code/Generic_classifier.py`
4. Using the app:
    - Upload a dataset (`.csv` or `.xlsx`) — try one of the sample files in `datafiles/` to get started quickly
    - Select target and features
    - Choose a classifier and adjust hyperparameters
    - Run model training and view evaluation results
    - Generate one of the eight plots and read the provided interpretation text

---

## Roadmap / Planned Improvements

- Polish the Tkinter UI to a more professional look and feel.
- General code review and cleanup pass.
- Deploy a web version on Streamlit Community Cloud.

---

## Contributors & Acknowledgments
1. Developer: [Tejas Desai]
2. LinkedIn: [https://www.linkedin.com/in/tejasddesaiindia/]

---

## License

This project is released under the MIT License.
