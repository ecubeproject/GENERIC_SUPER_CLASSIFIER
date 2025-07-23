# GENERIC_SUPER_CLASSIFIER – Interactive Multi-Model Auto-Classification Tool

**Organization:** Ecube Analytics  
**Interface:** Standalone desktop app (standalone executable); cloud-ready (Streamlit/Flask)

---

## Project Overview

GENERIC_SUPER_CLASSIFIER is a versatile machine-learning application designed for users who want to train and evaluate classification models **without writing any code**. With intuitive UI controls, it enables end-to-end workflows from data upload to model selection, training, evaluation, and visualization—all in one place.

**Key Features**:
- Accepts `.csv` and `.xlsx` datasets of variable size (hardware limitations apply).
- Automatically performs **Exploratory Data Analysis (EDA)**, including data types, missing values, and summary statistics.
- User-driven selection of:
  - Target variable (binary or multi-class classification)
  - One of **13 classification algorithms**, with dynamic, algorithm-specific hyperparameter controls
- Model evaluation metrics and classification reports available with a single button click.
- Includes **7 visualization options**; each plot includes interpretive guidance to aid decision-making.
- Built-in preprocessing pipeline covering:
  - Missing-value imputation
  - Categorical encoding
  - Feature scaling

Supported classifiers include:
`Random Forest`, `SVC`, `KNN`, `XGBoost`, `AdaBoost`, `HistGradientBoosting`, `Logistic Regression`, `Decision Tree`, `Gradient Boosting`, `LightGBM`, `Gaussian Naive Bayes`, `Bernoulli Naive Bayes`, and a simple **Neural Network**.

---

##  Outcome & Impact

- Provides a **fully functional GUI** for ML experimentation, model comparison, and visualization.
- Ideal for:
  - In-house model benchmarking
  - Teaching ML concepts in workshops
  - Rapid prototyping of classification use-cases
- Portable as:
  - A standalone executable on any OS
  - A web app via cloud deployment (e.g., Streamlit, Flask)
- Saves time and technical overhead by handling preprocessing, training, and plotting in a clean, user-centric interface.

---

## Tech Stack

| Layer               | Technologies Used                             |
|---------------------|-----------------------------------------------|
| **Language**         | Python 3.x with Pandas, NumPy                |
| **ML Framework**     | Scikit-learn pipelines, XGBoost, LightGBM    |
| **Algorithms**       | 13 classifiers (see Features section)         |
| **Preprocessing**    | ColumnTransformer, Imputers, Scalers         |
| **Interface GUI**    | Native Python UI (Tkinter / PySimpleGUI) <br>or Flask/Streamlit for web |
| **Visualization**    | Matplotlib, Seaborn, SHAP/feature importance |
| **Data Input**       | `.csv` and `.xlsx` file handling              |

---

## Folder Structure

```plaintext
.
├── app.py                    # Main app launcher
├── gui/                      # GUI-related code and resources
├── preprocessing/           # Modules for pipelines and data cleaning
├── models/                   # Saved model objects for reuse
├── visualizations/           # Plotting modules with guidance text
├── requirements.txt         # Project dependencies
└── README.md                 # This documentation
```
---

## How to Use?
1. Clone the repository:    git clone <repo-url>
2. Install dependencies:    pip install -r requirements.txt
3. Launch the application:  python app.py
4. Using the app:
    - Upload your dataset (.csv or .xlsx)
    - Select target and features
    - Choose a classifier and adjust hyperparameters
    - Run model training and view evaluation results
    - Generate one of seven plots and read the provided interpretation text

---

## Contributors & Acknowledgments
 1. Developer: [Tejas Desai]
 2. LinkedIn: [https://www.linkedin.com/in/tejasddesaiindia/]

---

License

This project is released under the MIT License.

