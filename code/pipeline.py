"""Turn a (DataFrame, target, classifier name, params) spec into a fitted
scikit-learn pipeline plus its held-out evaluation.

No Tkinter here - everything is a plain function so it can be driven from a
notebook, the Streamlit port, or pytest. The Tk app only supplies the params
dict and renders the returned ``TrainResult``.
"""

import ast
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier,
                              HistGradientBoostingClassifier)
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb


# ---------------------------------------------------------------------------
# Classifier registry
# ---------------------------------------------------------------------------

CLASSIFIER_CLASSES = {
    'Random Forest': RandomForestClassifier,
    'SVM/SVC': SVC,
    'KNN': KNeighborsClassifier,
    'XGBoost': xgb.XGBClassifier,
    'AdaBoost': AdaBoostClassifier,
    'HistGradientBoostingClassifier': HistGradientBoostingClassifier,
    'Logistic Regression': LogisticRegression,
    'Decision Tree': DecisionTreeClassifier,
    'Gradient Boosting': GradientBoostingClassifier,
    'LightGBM': lgb.LGBMClassifier,
    'Gaussian Naive Bayes': GaussianNB,
    'Bernoulli Naive Bayes': BernoulliNB,
    'Neural Network (MLP)': MLPClassifier,
}

# Plain-Python defaults (the Tk layer wraps these in tk.Var for its widgets).
DEFAULT_PARAMS = {
    'Random Forest': {'n_estimators': 100, 'max_depth': 10,
                      'min_samples_split': 2, 'min_samples_leaf': 1},
    'SVM/SVC': {'C': 1.0, 'kernel': 'rbf', 'probability': True},
    'KNN': {'n_neighbors': 5},
    'XGBoost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6},
    'AdaBoost': {'n_estimators': 50, 'learning_rate': 1.0},
    'HistGradientBoostingClassifier': {'learning_rate': 0.1, 'max_iter': 100},
    'Logistic Regression': {'solver': 'lbfgs', 'C': 1.0, 'max_iter': 100,
                            'penalty': 'l2', 'l1_ratio': 0.5},
    'Decision Tree': {'criterion': 'gini', 'splitter': 'best', 'max_depth': 10},
    'Gradient Boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
    'LightGBM': {'n_estimators': 100, 'learning_rate': 0.1, 'num_leaves': 31},
    'Gaussian Naive Bayes': {'var_smoothing': 1e-9},
    'Bernoulli Naive Bayes': {'alpha': 1.0, 'binarize': 0.0},
    'Neural Network (MLP)': {'hidden_layer_sizes': '(100,)', 'activation': 'relu',
                             'solver': 'adam', 'max_iter': 200},
}

# valid penalties per Logistic Regression solver
VALID_PENALTIES = {
    'newton-cg': ['l2', 'none'],
    'lbfgs': ['l2', 'none'],
    'liblinear': ['l1', 'l2'],
    'sag': ['l2', 'none'],
    'saga': ['l1', 'l2', 'elasticnet', 'none'],
}


# ---------------------------------------------------------------------------
# Data-preparation helpers (also used directly by the Tk layer)
# ---------------------------------------------------------------------------

def parse_hidden_layer_sizes(text):
    """Safely parse the MLP 'hidden_layer_sizes' field.

    Accepts '100', '100,50', '(100,)', '[100, 50]'. Returns a tuple of
    positive ints. Raises ValueError on anything else (no eval()).
    """
    if isinstance(text, (tuple, list)):
        value = tuple(text)
    else:
        value = ast.literal_eval(str(text).strip())
    if isinstance(value, int):
        value = (value,)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("hidden_layer_sizes must be an int or a non-empty tuple/list of ints")
    sizes = tuple(int(v) for v in value)
    if any(s <= 0 for s in sizes):
        raise ValueError("hidden_layer_sizes values must be positive integers")
    return sizes


def looks_continuous(y):
    """Heuristic: does this target look like a regression target rather than a
    set of class labels? Numeric, high-cardinality, and mostly-unique."""
    y = pd.Series(y).dropna()
    if len(y) == 0 or not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return False
    n_unique = y.nunique()
    return n_unique > 20 and (n_unique / len(y)) > 0.05


def expand_datetime_columns(df):
    """Replace datetime columns with numeric year/month/day/dayofweek parts.

    Only touches columns that are already datetime64, or object columns that
    parse as dates for >=90% of non-null rows with a plausible string length.
    """
    df = df.copy()
    for col in list(df.columns):
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            dt = s
        elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            sample = s.dropna().astype(str)
            if sample.empty or sample.str.len().median() < 8:
                continue
            parsed = pd.to_datetime(s, errors='coerce')
            if parsed.notna().mean() < 0.9 or parsed.nunique() < 3:
                continue
            dt = parsed
        else:
            continue
        df[f'{col}_year'] = dt.dt.year
        df[f'{col}_month'] = dt.dt.month
        df[f'{col}_day'] = dt.dt.day
        df[f'{col}_dayofweek'] = dt.dt.dayofweek
        df = df.drop(columns=[col])
    return df


def split_feature_types(X):
    """Return (X, numeric_cols, categorical_cols, dropped) after casting
    bool -> int and dropping constant / all-NaN columns. Anything not in either
    list is intentionally left out of the model."""
    X = X.copy()
    dropped = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if dropped:
        X = X.drop(columns=dropped)
    for c in X.columns:
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(int)
    numeric_cols, categorical_cols = [], []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            numeric_cols.append(c)
        elif (pd.api.types.is_object_dtype(X[c]) or pd.api.types.is_string_dtype(X[c])
              or isinstance(X[c].dtype, pd.CategoricalDtype)):
            categorical_cols.append(c)
    return X, numeric_cols, categorical_cols, dropped


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------

def build_preprocessor(numeric_cols, categorical_cols):
    """ColumnTransformer: median-impute + scale numerics, most-frequent-impute
    + one-hot (dense, capped cardinality) categoricals."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # sparse_output=False: GaussianNB and a few plot paths cannot accept a
        # sparse matrix. min_frequency/max_categories stop a high-cardinality
        # column (IDs, free text) exploding into thousands of one-hot columns.
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False,
                                 min_frequency=0.01, max_categories=25)),
    ])
    transformers = []
    if numeric_cols:
        transformers.append(('num', numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(('cat', categorical_transformer, categorical_cols))
    return ColumnTransformer(transformers=transformers)


def build_classifier(name, params):
    """Instantiate ``name`` from a plain params dict, applying the same
    per-classifier fix-ups the Tk app used to do inline.

    Raises KeyError for an unknown name, ValueError/TypeError for bad params.
    """
    if name not in CLASSIFIER_CLASSES:
        raise KeyError(f"Unknown classifier: {name!r}")
    params = dict(params)

    if name == 'Logistic Regression':
        penalty = params.get('penalty', 'l2')
        params['penalty'] = None if penalty in ('none', 'None', '', None) else penalty
        if params['penalty'] != 'elasticnet':
            params.pop('l1_ratio', None)

    if name == 'Neural Network (MLP)' and 'hidden_layer_sizes' in params:
        params['hidden_layer_sizes'] = parse_hidden_layer_sizes(params['hidden_layer_sizes'])

    if name == 'SVM/SVC':
        params['probability'] = True  # required for predict_proba / the plots

    return CLASSIFIER_CLASSES[name](**params)


# ---------------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    classifier_name: str
    params: dict
    pipeline: Pipeline
    X_test: pd.DataFrame
    y_test: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    class_names: list
    is_binary: bool
    numeric_cols: list
    categorical_cols: list
    dropped_cols: list
    accuracy: float
    confusion_matrix: np.ndarray
    classification_report: str
    model_classes: list = field(default_factory=list)

    def transformed_X_test(self):
        """X_test pushed through the fitted preprocessor -> dense ndarray.
        Used by the PCA / silhouette plots so they work on non-numeric data."""
        mat = self.pipeline.named_steps['preprocessor'].transform(self.X_test)
        return np.asarray(mat.todense()) if hasattr(mat, 'todense') else np.asarray(mat)


def train_and_evaluate(df, target, classifier_name, params,
                       test_size=0.3, random_state=42):
    """Fit ``classifier_name`` on ``df`` predicting ``target`` and evaluate on a
    stratified hold-out. Returns a :class:`TrainResult`.

    Raises ValueError for data problems the caller should surface to the user
    (missing target, single-class target, no usable features).
    """
    if target not in df.columns:
        raise ValueError(f"Target variable {target!r} not found in the dataset.")

    data = df.dropna(subset=[target])
    X = data.drop(columns=[target])
    y = data[target]

    if y.nunique() < 2:
        raise ValueError(
            f"Target {target!r} has only one class after dropping missing values.")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = [str(c) for c in label_encoder.classes_]
    is_binary = len(class_names) == 2

    X = expand_datetime_columns(X)
    X, numeric_cols, categorical_cols, dropped_cols = split_feature_types(X)
    if not numeric_cols and not categorical_cols:
        raise ValueError("No usable feature columns after preprocessing.")

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    classifier = build_classifier(classifier_name, params)
    clf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', classifier)])

    # Stratify when every class has >=2 rows, else fall back to a plain split.
    min_class_count = int(np.min(np.bincount(y_encoded)))
    stratify = y_encoded if min_class_count >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=stratify)

    clf_pipeline.fit(X_train, y_train)
    y_pred = clf_pipeline.predict(X_test)
    y_proba = clf_pipeline.predict_proba(X_test)

    target_names = class_names if len(class_names) == len(np.unique(y_test)) else None
    return TrainResult(
        classifier_name=classifier_name,
        params=dict(params),
        pipeline=clf_pipeline,
        X_test=X_test,
        y_test=np.asarray(y_test),
        y_pred=np.asarray(y_pred),
        y_proba=np.asarray(y_proba),
        class_names=class_names,
        is_binary=is_binary,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        dropped_cols=dropped_cols,
        accuracy=accuracy_score(y_test, y_pred),
        confusion_matrix=confusion_matrix(y_test, y_pred),
        classification_report=classification_report(
            y_test, y_pred, target_names=target_names, zero_division=0),
        model_classes=list(clf_pipeline.classes_),
    )
