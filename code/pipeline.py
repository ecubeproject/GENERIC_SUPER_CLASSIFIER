"""Turn a (DataFrame, target, classifier name, params) spec into a fitted
scikit-learn pipeline plus its held-out evaluation.

No Tkinter here - everything is a plain function so it can be driven from a
notebook, the Streamlit port, or pytest. The Tk app only supplies the params
dict and renders the returned ``TrainResult``.
"""

import ast
import inspect
from dataclasses import dataclass, field

import joblib

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

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
    input_columns: list        # raw feature columns fed in (before datetime expansion)
    numeric_cols: list
    categorical_cols: list
    dropped_cols: list
    accuracy: float
    confusion_matrix: np.ndarray
    classification_report: str
    model_classes: list = field(default_factory=list)
    balanced: bool = False           # was class-imbalance correction requested?
    balance_applied: str = "not requested"  # how it was applied (or why it wasn't)
    label_encoder: object = None     # fitted LabelEncoder (target labels <-> ints)

    def transformed_X_test(self):
        """X_test pushed through the fitted preprocessor -> dense ndarray.
        Used by the PCA / silhouette plots so they work on non-numeric data."""
        mat = self.pipeline.named_steps['preprocessor'].transform(self.X_test)
        return np.asarray(mat.todense()) if hasattr(mat, 'todense') else np.asarray(mat)


@dataclass
class _PreparedData:
    """Feature matrix + label-encoded target ready for a pipeline, shared by
    :func:`train_and_evaluate` and :func:`cross_validate_model`."""
    X: pd.DataFrame
    y: np.ndarray
    label_encoder: LabelEncoder
    class_names: list
    is_binary: bool
    input_columns: list
    numeric_cols: list
    categorical_cols: list
    dropped_cols: list


def _prepare_xy(df, target, feature_columns):
    """Validate inputs, drop rows with a missing target, optionally restrict to
    ``feature_columns``, label-encode the target, expand datetime columns and
    classify the remaining columns as numeric / categorical.

    Raises ValueError for every data problem the caller should surface.
    """
    if target not in df.columns:
        raise ValueError(f"Target variable {target!r} not found in the dataset.")

    data = df.dropna(subset=[target])
    X = data.drop(columns=[target])
    y = data[target]

    if feature_columns is not None:
        feature_columns = list(feature_columns)
        if not feature_columns:
            raise ValueError("No feature columns selected.")
        if target in feature_columns:
            raise ValueError(f"Target {target!r} cannot also be a feature column.")
        missing = [c for c in feature_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Feature column(s) not in the dataset: {missing}")
        X = X[feature_columns]

    input_columns = list(X.columns)

    if y.nunique() < 2:
        raise ValueError(
            f"Target {target!r} has only one class after dropping missing values.")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = [str(c) for c in label_encoder.classes_]

    X = expand_datetime_columns(X)
    X, numeric_cols, categorical_cols, dropped_cols = split_feature_types(X)
    if not numeric_cols and not categorical_cols:
        raise ValueError("No usable feature columns after preprocessing.")

    return _PreparedData(X, y_encoded, label_encoder, class_names,
                         len(class_names) == 2, input_columns,
                         numeric_cols, categorical_cols, dropped_cols)


def _apply_class_balance(clf_pipeline, classifier_name, y_train):
    """Wire class-imbalance correction into the classifier step in-place.

    Prefers the estimator's own ``class_weight='balanced'``; falls back to a
    balanced ``sample_weight`` for estimators whose ``fit`` accepts one
    (XGBoost, LightGBM, the boosting / MLP / NB families). Returns
    ``(fit_kwargs, description)`` - ``fit_kwargs`` is merged into
    ``clf_pipeline.fit``.
    """
    clf_step = clf_pipeline.named_steps['classifier']
    if 'class_weight' in clf_step.get_params():
        clf_step.set_params(class_weight='balanced')
        return {}, "class_weight='balanced'"
    if 'sample_weight' in inspect.signature(clf_step.fit).parameters:
        sw = compute_sample_weight('balanced', y_train)
        return {'classifier__sample_weight': sw}, "balanced sample_weight"
    return {}, f"not supported by {classifier_name}"


def train_and_evaluate(df, target, classifier_name, params,
                       feature_columns=None, balance_classes=False,
                       test_size=0.3, random_state=42):
    """Fit ``classifier_name`` on ``df`` predicting ``target`` and evaluate on a
    stratified hold-out. Returns a :class:`TrainResult`.

    ``feature_columns`` optionally restricts training to a subset of columns
    (e.g. to drop ID / leakage columns); ``None`` uses every column except the
    target.

    ``balance_classes`` corrects for class imbalance (``class_weight='balanced'``
    where the estimator supports it, otherwise a balanced ``sample_weight``;
    KNN cannot use either and is trained unweighted).

    Raises ValueError for data problems the caller should surface to the user
    (missing target, single-class target, bad column selection, no usable
    features).
    """
    prep = _prepare_xy(df, target, feature_columns)
    X, y_encoded = prep.X, prep.y
    class_names, is_binary, input_columns = prep.class_names, prep.is_binary, prep.input_columns
    numeric_cols, categorical_cols, dropped_cols = (
        prep.numeric_cols, prep.categorical_cols, prep.dropped_cols)

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    classifier = build_classifier(classifier_name, params)
    clf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', classifier)])

    # Stratify when every class has >=2 rows, else fall back to a plain split.
    min_class_count = int(np.min(np.bincount(y_encoded)))
    stratify = y_encoded if min_class_count >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=stratify)

    fit_kwargs, balance_applied = ({}, "not requested")
    if balance_classes:
        fit_kwargs, balance_applied = _apply_class_balance(
            clf_pipeline, classifier_name, y_train)

    clf_pipeline.fit(X_train, y_train, **fit_kwargs)
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
        input_columns=input_columns,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        dropped_cols=dropped_cols,
        accuracy=accuracy_score(y_test, y_pred),
        confusion_matrix=confusion_matrix(y_test, y_pred),
        classification_report=classification_report(
            y_test, y_pred, target_names=target_names, zero_division=0),
        model_classes=list(clf_pipeline.classes_),
        balanced=balance_classes,
        balance_applied=balance_applied,
        label_encoder=prep.label_encoder,
    )


# ---------------------------------------------------------------------------
# K-fold cross-validation
# ---------------------------------------------------------------------------

@dataclass
class CVResult:
    """Cross-validation metrics for one (classifier, params) spec.

    ``fold_metrics`` maps a metric name to its list of per-fold values;
    ``mean`` / ``std`` are the aggregates. ``roc_auc`` is present only when it
    could be computed on every fold.
    """
    classifier_name: str
    params: dict
    k: int
    class_names: list
    is_binary: bool
    balanced: bool
    balance_applied: str
    fold_metrics: dict
    mean: dict
    std: dict

    def summary(self):
        head = (f"{self.k}-fold cross-validation - {self.classifier_name} "
                f"({'binary' if self.is_binary else f'{len(self.class_names)}-class'})")
        rows = [f"  {m:<16s} {self.mean[m]:.4f} +/- {self.std[m]:.4f}"
                for m in self.fold_metrics]
        return head + "\n" + "\n".join(rows)


def cross_validate_model(df, target, classifier_name, params,
                         feature_columns=None, balance_classes=False,
                         k=5, random_state=42):
    """Stratified k-fold cross-validation for one classifier spec.

    ``k`` is capped at the smallest class count (need >=1 test row per class per
    fold) and floored at 2. Returns a :class:`CVResult` with accuracy,
    macro precision / recall / F1 and - when computable on every fold -
    macro one-vs-rest ROC-AUC.
    """
    if k < 2:
        raise ValueError("k must be at least 2.")
    prep = _prepare_xy(df, target, feature_columns)
    X, y = prep.X, prep.y

    min_class_count = int(np.min(np.bincount(y)))
    if min_class_count < 2:
        raise ValueError("Every class needs at least 2 rows for cross-validation.")
    n_splits = max(2, min(k, min_class_count))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics = {m: [] for m in
                    ('accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc')}
    can_auc = True
    balance_applied = "not requested"

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        pipe = Pipeline(steps=[
            ('preprocessor', build_preprocessor(prep.numeric_cols, prep.categorical_cols)),
            ('classifier', build_classifier(classifier_name, params)),
        ])
        fit_kwargs = {}
        if balance_classes:
            fit_kwargs, balance_applied = _apply_class_balance(pipe, classifier_name, y_tr)
        pipe.fit(X_tr, y_tr, **fit_kwargs)

        y_hat = pipe.predict(X_te)
        fold_metrics['accuracy'].append(accuracy_score(y_te, y_hat))
        fold_metrics['precision_macro'].append(
            precision_score(y_te, y_hat, average='macro', zero_division=0))
        fold_metrics['recall_macro'].append(
            recall_score(y_te, y_hat, average='macro', zero_division=0))
        fold_metrics['f1_macro'].append(
            f1_score(y_te, y_hat, average='macro', zero_division=0))

        if can_auc:
            try:
                proba = pipe.predict_proba(X_te)
                if prep.is_binary:
                    fold_metrics['roc_auc'].append(roc_auc_score(y_te, proba[:, 1]))
                else:
                    fold_metrics['roc_auc'].append(
                        roc_auc_score(y_te, proba, multi_class='ovr', average='macro'))
            except (ValueError, AttributeError):
                can_auc = False

    if not can_auc or len(fold_metrics['roc_auc']) != n_splits:
        fold_metrics.pop('roc_auc')

    mean = {m: float(np.mean(v)) for m, v in fold_metrics.items()}
    std = {m: float(np.std(v)) for m, v in fold_metrics.items()}
    return CVResult(classifier_name, dict(params), n_splits, prep.class_names,
                    prep.is_binary, balance_classes, balance_applied,
                    fold_metrics, mean, std)


# ---------------------------------------------------------------------------
# Model persistence + scoring new data
# ---------------------------------------------------------------------------

SAVED_MODEL_FORMAT = 1


@dataclass
class SavedModel:
    """Everything needed to score new rows with a previously trained model -
    no training data, no Tk. Serialised with joblib by :func:`save_model`."""
    pipeline: Pipeline
    label_encoder: object
    class_names: list
    input_columns: list          # raw columns the caller must supply
    numeric_cols: list           # columns the fitted preprocessor expects...
    categorical_cols: list       # ...after datetime expansion / bool cast
    classifier_name: str
    format_version: int = SAVED_MODEL_FORMAT

    def describe(self):
        kind = 'binary' if len(self.class_names) == 2 else f'{len(self.class_names)}-class'
        return (f"{self.classifier_name} ({kind}) - classes {self.class_names}\n"
                f"Required input columns: {self.input_columns}")


def save_model(result, path):
    """Persist a trained :class:`TrainResult` to ``path`` (joblib)."""
    if result.label_encoder is None:
        raise ValueError("This result has no label encoder and cannot be saved.")
    bundle = SavedModel(
        pipeline=result.pipeline,
        label_encoder=result.label_encoder,
        class_names=list(result.class_names),
        input_columns=list(result.input_columns),
        numeric_cols=list(result.numeric_cols),
        categorical_cols=list(result.categorical_cols),
        classifier_name=result.classifier_name,
    )
    joblib.dump(bundle, path)
    return path


def load_model(path):
    """Load a :class:`SavedModel` written by :func:`save_model`."""
    bundle = joblib.load(path)
    if not isinstance(bundle, SavedModel):
        raise ValueError(f"{path!r} is not a GENERIC_SUPER_CLASSIFIER model file.")
    if bundle.format_version != SAVED_MODEL_FORMAT:
        raise ValueError(
            f"Model file format v{bundle.format_version} is unsupported "
            f"(this build expects v{SAVED_MODEL_FORMAT}).")
    return bundle


def _replay_feature_prep(df, bundle):
    """Reproduce the datetime-expansion + bool-cast that ``train_and_evaluate``
    applied before fitting, then select exactly the columns the fitted
    preprocessor expects."""
    X = df[bundle.input_columns].copy()
    X = expand_datetime_columns(X)
    for c in X.columns:
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(int)
    needed = bundle.numeric_cols + bundle.categorical_cols
    missing = [c for c in needed if c not in X.columns]
    if missing:
        raise ValueError(
            f"Could not rebuild the model's feature columns from the input: {missing}")
    return X[needed]


def predict_dataframe(bundle, df):
    """Score ``df`` with a loaded :class:`SavedModel`.

    Returns a new DataFrame (same index as ``df``) with a ``prediction`` column
    in the target's original labels plus one ``proba_<class>`` column per class.
    Extra columns in ``df`` are ignored; a missing required column raises
    ValueError.
    """
    missing = [c for c in bundle.input_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required column(s): {missing}")

    X = _replay_feature_prep(df, bundle)
    encoded = bundle.pipeline.predict(X)
    out = pd.DataFrame(
        {'prediction': bundle.label_encoder.inverse_transform(encoded)}, index=df.index)

    proba = bundle.pipeline.predict_proba(X)
    for j, cls in enumerate(bundle.pipeline.classes_):
        cls = int(cls)
        name = bundle.class_names[cls] if cls < len(bundle.class_names) else str(cls)
        out[f'proba_{name}'] = proba[:, j]
    return out


# ---------------------------------------------------------------------------
# Model leaderboard
# ---------------------------------------------------------------------------

def _roc_auc_or_none(y_true, y_proba, is_binary):
    try:
        if is_binary:
            return float(roc_auc_score(y_true, y_proba[:, 1]))
        return float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro'))
    except (ValueError, IndexError):
        return None


def _trainresult_metrics(result):
    yt, yp = result.y_test, result.y_pred
    return {
        'accuracy': float(result.accuracy),
        'precision_macro': float(precision_score(yt, yp, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(yt, yp, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(yt, yp, average='macro', zero_division=0)),
        'roc_auc': _roc_auc_or_none(yt, result.y_proba, result.is_binary),
    }


@dataclass
class LeaderboardRow:
    index: int
    label: str
    classifier_name: str
    source: str            # 'holdout' or 'cv-<k>'
    balanced: bool
    metrics: dict          # metric name -> value or None
    params: dict


class Leaderboard:
    """Accumulate :class:`TrainResult` / :class:`CVResult` runs and rank them.

    UI-free: ``table()`` returns plain text the caller drops into a widget, a
    notebook cell or a log.
    """

    METRICS = ('accuracy', 'f1_macro', 'roc_auc', 'precision_macro', 'recall_macro')

    def __init__(self):
        self.rows = []

    def __len__(self):
        return len(self.rows)

    def add(self, result, label=None):
        """Add a run. ``result`` is a TrainResult or a CVResult."""
        if isinstance(result, TrainResult):
            metrics, source = _trainresult_metrics(result), 'holdout'
        elif isinstance(result, CVResult):
            metrics = {m: float(result.mean[m]) for m in result.fold_metrics}
            source = f'cv-{result.k}'
        else:
            raise TypeError("Leaderboard.add() expects a TrainResult or CVResult")

        row = LeaderboardRow(
            index=len(self.rows) + 1,
            label=label or result.classifier_name,
            classifier_name=result.classifier_name,
            source=source,
            balanced=bool(result.balanced),
            metrics=metrics,
            params=dict(result.params),
        )
        self.rows.append(row)
        return row

    def best(self, metric='accuracy'):
        """The row with the highest ``metric`` (None if no row has it)."""
        scored = [r for r in self.rows if r.metrics.get(metric) is not None]
        return max(scored, key=lambda r: r.metrics[metric], default=None)

    def table(self, sort_by='accuracy'):
        if not self.rows:
            return "(no models on the leaderboard yet)"
        cols = [m for m in self.METRICS
                if any(r.metrics.get(m) is not None for r in self.rows)]
        rows = self.rows
        if sort_by in cols:
            rows = sorted(
                self.rows,
                key=lambda r: (r.metrics.get(sort_by) is not None,
                               r.metrics.get(sort_by) or 0.0),
                reverse=True)

        header = (f"{'#':>2}  {'Model':<24} {'Eval':<8} {'Bal':<4} "
                  + " ".join(f"{c:>15}" for c in cols))
        lines = [header, "-" * len(header)]
        for r in rows:
            cells = " ".join(
                (f"{r.metrics[c]:>15.4f}" if r.metrics.get(c) is not None else f"{'-':>15}")
                for c in cols)
            lines.append(f"{r.index:>2}  {r.label[:24]:<24} {r.source:<8} "
                         f"{('yes' if r.balanced else 'no'):<4} {cells}")
        return "\n".join(lines)
