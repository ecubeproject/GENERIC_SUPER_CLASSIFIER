import ast
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Toplevel
from io import StringIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, roc_curve, auc, precision_recall_curve,
                             average_precision_score, confusion_matrix)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.calibration import calibration_curve

global_df = None
X_test, y_test, clf_pipeline = None, None, None
plot_dropdown = None
label_encoder = LabelEncoder()
# Set by on_train_clicked so the plot functions never have to guess:
class_names = None          # list[str] - display labels, index i == encoded class i
is_binary_task = False      # True when exactly two classes

def upload_file():
    global global_df
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
    if file_path:
        try:
            if file_path.endswith('.xlsx'):
                global_df = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                global_df = pd.read_csv(file_path)
            else:
                messagebox.showerror("Error", "Unsupported file type.")
                return

            results_text.insert(tk.END, "\n\n")  # Add space before new data
            # Capture DataFrame info
            buf = StringIO()
            global_df.info(buf=buf)
            data_info = buf.getvalue()
            
            data_desc = global_df.describe().to_string()
            missing_values = global_df.isna().sum().to_frame('Missing Values').to_string()
            unique_values = global_df.nunique().to_frame('Unique Values').to_string()

            results_text.insert(tk.END, f"Data Info:\n{data_info}\n\n")
            results_text.insert(tk.END, f"Data Description:\n{data_desc}\n\n")
            results_text.insert(tk.END, f"Missing Values:\n{missing_values}\n\n")
            results_text.insert(tk.END, f"Unique Values:\n{unique_values}\n\n")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error processing file: {e}")
    # If no file was chosen the user simply cancelled the dialog - nothing to do.


def print_to_output(message):
    """Append a line to the results pane (used for non-fatal diagnostics)."""
    try:
        results_text_middle.insert(tk.END, f"\n{message}\n")
        results_text_middle.see(tk.END)
    except Exception:
        print(message)


def parse_hidden_layer_sizes(text):
    """Safely parse the MLP 'hidden_layer_sizes' text field.

    Accepts '100', '100,50', '(100,)', '[100, 50]'. Returns a tuple of
    positive ints. Raises ValueError on anything else (no eval()).
    """
    value = ast.literal_eval(text.strip())
    if isinstance(value, int):
        value = (value,)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("hidden_layer_sizes must be an int or a non-empty tuple/list of ints")
    sizes = tuple(int(v) for v in value)
    if any(s <= 0 for s in sizes):
        raise ValueError("hidden_layer_sizes values must be positive integers")
    return sizes


# ----------------------------------------------------------------------------
# Data-preparation helpers (kept UI-free so they are easy to test / reuse)
# ----------------------------------------------------------------------------

def looks_continuous(y):
    """Heuristic: does this target look like a regression target rather than
    a set of class labels? Numeric, high-cardinality, and mostly-unique."""
    y = y.dropna()
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
    """Return (numeric_cols, categorical_cols) after casting bool -> int and
    dropping constant / all-NaN columns. Anything not in either list is
    intentionally left out of the model."""
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


def transformed_test_matrix():
    """X_test pushed through the fitted preprocessor -> dense numeric ndarray.
    Used by the PCA / silhouette plots so they work on non-numeric datasets."""
    mat = clf_pipeline.named_steps['preprocessor'].transform(X_test)
    return np.asarray(mat.todense()) if hasattr(mat, 'todense') else np.asarray(mat)


def class_color(i, n):
    """Stable per-class color from a colormap (handles any number of classes)."""
    if n <= 10:
        return plt.get_cmap('tab10')(i % 10)
    if n <= 20:
        return plt.get_cmap('tab20')(i % 20)
    return plt.get_cmap('hsv')(i / max(n, 1))

def setup_ui(root):
    global plot_dropdown
    # Create upper frame
    upper_frame = tk.Frame(root)
    upper_frame.pack(fill=tk.BOTH, expand=True)

    # Create middle frame
    middle_frame = tk.Frame(root)
    middle_frame.pack(fill=tk.BOTH, expand=True)

    # Create lower frame
    lower_frame = tk.Frame(root)
    lower_frame.pack(fill=tk.BOTH, expand=True)

    # Upper Frame UI Elements

    # File upload button
    upload_button = tk.Button(upper_frame, text="Upload Datafile", command=upload_file)
    upload_button.grid(row=0, column=0, columnspan=2, padx=170, pady=10)
    
    # Adding a frame to hold the text widget and the scrollbar in the upper frame
    text_frame = tk.Frame(upper_frame, width=1200, height=320)
    text_frame.grid(row=1, column=0, columnspan=2, padx=170, pady=10)
   
    global results_text
    results_text = tk.Text(text_frame, width=120, height=16, wrap="none")  # Increase height to 16

    # Horizontal Scrollbar
    h_scrollbar = tk.Scrollbar(text_frame, orient=tk.HORIZONTAL, command=results_text.xview)
    h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
    results_text.config(xscrollcommand=h_scrollbar.set)

    # Vertical Scrollbar
    v_scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=results_text.yview)
    v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    results_text.config(yscrollcommand=v_scrollbar.set)
    
    results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # Middle Frame UI Elements
    tk.Label(middle_frame, text="Target Variable:").grid(row=0, column=0, padx=170, pady=10)
    target_entry = tk.Entry(middle_frame, textvariable=target_var)
    target_entry.grid(row=0, column=1, padx=170, pady=10)

    tk.Label(middle_frame, text="Choose Classifier:").grid(row=1, column=0, padx=170, pady=10)
    global classifier_dropdown
    classifier_dropdown = ttk.Combobox(middle_frame, values=list(classifier_params.keys()), state="readonly")
    classifier_dropdown.grid(row=1, column=1, padx=170, pady=10)
    classifier_dropdown.bind("<<ComboboxSelected>>", on_classifier_change)

    global classifier_params_frame
    classifier_params_frame = tk.Frame(middle_frame)
    classifier_params_frame.grid(row=2, column=0, columnspan=2, padx=170, pady=10)

    # Train and Evaluate Button
    train_button = tk.Button(middle_frame, text="Train and Evaluate", command=on_train_clicked)
    train_button.grid(row=3, column=0, columnspan=2, padx=170, pady=10)

    # Results Display in middle frame
    results_label = tk.Label(middle_frame, text="Results:", anchor="w", justify="left")
    results_label.grid(row=4, column=0, padx=170, pady=10)

    text_frame_middle = tk.Frame(middle_frame, width=1200, height=360)
    text_frame_middle.grid(row=5, column=0, columnspan=2, padx=170, pady=10)

    global results_text_middle
    results_text_middle = tk.Text(text_frame_middle, width=120, height=18, wrap="none")  # Increase height to 18

    # Horizontal Scrollbar
    h_scrollbar_middle = tk.Scrollbar(text_frame_middle, orient=tk.HORIZONTAL, command=results_text_middle.xview)
    h_scrollbar_middle.pack(side=tk.BOTTOM, fill=tk.X)
    results_text_middle.config(xscrollcommand=h_scrollbar_middle.set)

    # Vertical Scrollbar
    v_scrollbar_middle = tk.Scrollbar(text_frame_middle, orient=tk.VERTICAL, command=results_text_middle.yview)
    v_scrollbar_middle.pack(side=tk.RIGHT, fill=tk.Y)
    results_text_middle.config(yscrollcommand=v_scrollbar_middle.set)

    results_text_middle.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    return classifier_dropdown, train_button, results_text, plot_dropdown

classifiers = [
    'Random Forest', 'SVM/SVC', 'KNN', 'XGBoost', 'AdaBoost',
    'HistGradientBoostingClassifier', 'Decision Tree',
    'Gradient Boosting', 'LightGBM', 'Gaussian Naive Bayes',
    'Bernoulli Naive Bayes', 'Neural Network (MLP)', 'Logistic Regression'
]

# valid penalties are applicable for logistic regression only
valid_penalties = {
    'newton-cg': ['l2', 'none'],
    'lbfgs': ['l2', 'none'],
    'liblinear': ['l1', 'l2'],
    'sag': ['l2', 'none'],
    'saga': ['l1', 'l2', 'elasticnet', 'none']
}

def initialize_classifier_params():
    return {
        'Random Forest': {
            'n_estimators': tk.IntVar(value=100),
            'max_depth': tk.IntVar(value=10),
            'min_samples_split': tk.IntVar(value=2),
            'min_samples_leaf': tk.IntVar(value=1)
        },
        'SVM/SVC': {
            'C': tk.DoubleVar(value=1.0),
            'kernel': tk.StringVar(value='rbf'),
            'probability': tk.BooleanVar(value=True)  # Ensure probability is always True
        },
        'KNN': {
            'n_neighbors': tk.IntVar(value=5)
        },
        'XGBoost': {
            'n_estimators': tk.IntVar(value=100),
            'learning_rate': tk.DoubleVar(value=0.1),
            'max_depth': tk.IntVar(value=6)
        },
        'AdaBoost': {
            'n_estimators': tk.IntVar(value=50),
            'learning_rate': tk.DoubleVar(value=1.0)
            # 'algorithm' intentionally omitted: deprecated in scikit-learn 1.6
            # and removed in 1.8. 'SAMME' is the only supported value and is the
            # default, so passing it explicitly only risks a TypeError.
        },
        'HistGradientBoostingClassifier': {
            'learning_rate': tk.DoubleVar(value=0.1),
            'max_iter': tk.IntVar(value=100)
        },
        'Logistic Regression': {
            'solver': tk.StringVar(value='lbfgs'),
            'C': tk.DoubleVar(value=1.0),
            'max_iter': tk.IntVar(value=100),
            'l1_ratio': tk.DoubleVar(value=0.5)  # Add l1_ratio parameter with default value
        },
        'Decision Tree': {
            'criterion': tk.StringVar(value='gini'),
            'splitter': tk.StringVar(value='best'),
            'max_depth': tk.IntVar(value=10)
        },
        'Gradient Boosting': {
            'n_estimators': tk.IntVar(value=100),
            'learning_rate': tk.DoubleVar(value=0.1),
            'max_depth': tk.IntVar(value=3)
        },
        'LightGBM': {
            'n_estimators': tk.IntVar(value=100),
            'learning_rate': tk.DoubleVar(value=0.1),
            'num_leaves': tk.IntVar(value=31)
        },
        'Gaussian Naive Bayes': {
            'var_smoothing': tk.DoubleVar(value=1e-9)
        },
        'Bernoulli Naive Bayes': {
            'alpha': tk.DoubleVar(value=1.0),
            'binarize': tk.DoubleVar(value=0.0)
        },
        'Neural Network (MLP)': {
            'hidden_layer_sizes': tk.StringVar(value='(100,)'),
            'activation': tk.StringVar(value='relu'),
            'solver': tk.StringVar(value='adam'),
            'max_iter': tk.IntVar(value=200)
        }
    }

classifier_classes = {
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
    'Neural Network (MLP)': MLPClassifier
}

def update_penalty_options(event):
    global penalty_dropdown, valid_penalties
    solver = solver_logit_dropdown.get()
    penalties = valid_penalties.get(solver, ['l2', 'none'])
    penalty_dropdown['values'] = penalties
    if penalty_dropdown.get() not in penalties:
        penalty_dropdown.set(penalties[0])

def on_classifier_change(event):
    global penalty_label, penalty_dropdown, solver_logit_dropdown, solver_label, label_c, entry_c, label_n_iter, entry_n_iter, label_l1_ratio, entry_l1_ratio
    selected_classifier = classifier_dropdown.get()
    for widget in classifier_params_frame.winfo_children():
        widget.destroy()

    params_ui = classifier_params[selected_classifier]
    row = 0
    for param, var in params_ui.items():
        label = tk.Label(classifier_params_frame, text=param.replace('_', ' ').title() + ":")
        label.grid(row=row, column=0, padx=170, pady=5)
        
        if isinstance(var, tk.StringVar):
            if param == 'kernel':
                entry = ttk.Combobox(classifier_params_frame, textvariable=var, values=['linear', 'poly', 'rbf', 'sigmoid'])
            elif param == 'algorithm':
                entry = ttk.Combobox(classifier_params_frame, textvariable=var, values=['SAMME', 'SAMME.R'])
            elif param == 'activation':
                entry = ttk.Combobox(classifier_params_frame, textvariable=var, values=['relu', 'tanh', 'logistic'])
            elif param == 'solver' and selected_classifier == 'Logistic Regression':
                solver_label = tk.Label(classifier_params_frame, text="Solver:")
                solver_label.grid(row=row, column=0, padx=170, pady=5)
                solver_logit_dropdown = ttk.Combobox(classifier_params_frame, textvariable=var, values=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
                solver_logit_dropdown.bind("<<ComboboxSelected>>", update_penalty_options)
                solver_logit_dropdown.grid(row=row, column=1, padx=170, pady=5)
                penalty_label = tk.Label(classifier_params_frame, text="Penalty:")  # This is the only penalty label we want
                penalty_label.grid(row=row+1, column=0, padx=170, pady=5)
                penalty_dropdown = ttk.Combobox(classifier_params_frame, state="readonly")
                penalty_dropdown.grid(row=row+1, column=1, padx=170, pady=5)
                penalty_dropdown.set('l2')  # default value
                penalty_dropdown['values'] = valid_penalties['lbfgs']  # default solver is 'lbfgs'
                row += 2  # Increment row to avoid overlap with other parameters
                continue
            elif param == 'solver' and selected_classifier == 'Neural Network (MLP)':
                entry = ttk.Combobox(classifier_params_frame, textvariable=var, values=['adam', 'sgd', 'lbfgs'])
            else:
                entry = ttk.Combobox(classifier_params_frame, textvariable=var)
        elif isinstance(var, tk.IntVar):
            if param == 'n_neighbors':
                entry = tk.Scale(classifier_params_frame, from_=1, to=20, orient='horizontal', variable=var)
            elif param == 'max_iter':
                entry = tk.Scale(classifier_params_frame, from_=1, to=5000, orient='horizontal', variable=var)
                label_n_iter = tk.Label(classifier_params_frame, text="n_iter:")
                label_n_iter.grid(row=row, column=0, padx=170, pady=5)
                entry.grid(row=row, column=1, padx=170, pady=5)
                row += 1  # Increment row for the next widget
                continue  # Skip the final grid call as it's already placed
            elif param == 'num_leaves':
                entry = tk.Scale(classifier_params_frame, from_=1, to=20, orient='horizontal', variable=var)
            elif param == 'n_estimators':
                entry = tk.Scale(classifier_params_frame, from_=1, to=100, orient='horizontal', variable=var)
            else:
                entry = tk.Scale(classifier_params_frame, from_=1, to=50, orient='horizontal', variable=var)
        elif isinstance(var, tk.DoubleVar):
            if param == 'learning_rate': 
                entry = tk.Scale(classifier_params_frame, from_=0.01, to=1, resolution=0.01, orient='horizontal', variable=var)
            elif param == 'C':
                label_c = tk.Label(classifier_params_frame, text="C:")
                label_c.grid(row=row, column=0, padx=170, pady=5)
                entry_c = tk.Scale(classifier_params_frame, from_=0.001, to=1, resolution=0.001, orient='horizontal', variable=var)
                entry_c.grid(row=row, column=1, padx=170, pady=5)
                row += 1  # Increment row for the next widget
                continue  # Skip the final grid call as it's already placed
            elif param == 'l1_ratio':
                label_l1_ratio = tk.Label(classifier_params_frame, text="l1_ratio:")
                label_l1_ratio.grid(row=row, column=0, padx=170, pady=5)
                entry_l1_ratio = tk.Scale(classifier_params_frame, from_=0.0, to=1.0, resolution=0.01, orient='horizontal', variable=var)
                entry_l1_ratio.grid(row=row, column=1, padx=170, pady=5)
                row += 1  # Increment row for the next widget
                continue
            else:
                entry = tk.Scale(classifier_params_frame, from_=0.01, to=1, resolution=0.01, orient='horizontal', variable=var)
        elif isinstance(var, tk.BooleanVar):
            entry = ttk.Combobox(classifier_params_frame, textvariable=var, values=[True, False], state="disabled")  # Ensure always True
        entry.grid(row=row, column=1, padx=170, pady=5)
        row += 1
    if selected_classifier == 'Logistic Regression':
        penalty_label.grid(row=row, column=0, padx=170, pady=5)
        penalty_dropdown.grid(row=row, column=1, padx=170, pady=5)

def on_train_clicked():
    global global_df, label_encoder, X_test, y_test, clf_pipeline, class_names, is_binary_task
    if global_df is None:
        messagebox.showerror("Error", "No dataset loaded. Please upload a dataset.")
        return

    target_variable = target_var.get()
    if target_variable not in global_df.columns:
        messagebox.showerror("Error", f"Target variable '{target_variable}' not found in the dataset.")
        return

    # Separating features and target; drop rows with a missing target
    data = global_df.dropna(subset=[target_variable])
    X = data.drop(target_variable, axis=1)
    y = data[target_variable]

    if y.nunique() < 2:
        messagebox.showerror("Error", f"Target '{target_variable}' has only one class after dropping missing values.")
        return

    if looks_continuous(y):
        proceed = messagebox.askyesno(
            "Target looks continuous",
            f"'{target_variable}' has {y.nunique()} distinct numeric values and looks like a "
            "regression target. This tool does classification.\n\nTreat it as "
            f"{y.nunique()} separate classes anyway?")
        if not proceed:
            return

    # Always label-encode the target: gives contiguous 0..k-1 labels (required by
    # XGBoost/LightGBM) and a single, reliable source of class names for the plots.
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = [str(c) for c in label_encoder.classes_]
    is_binary_task = len(class_names) == 2

    # Feature typing: expand datetimes, cast bools, drop constant/all-NaN columns
    X = expand_datetime_columns(X)
    X, numeric_cols, categorical_cols, dropped_cols = split_feature_types(X)
    if not numeric_cols and not categorical_cols:
        messagebox.showerror("Error", "No usable feature columns after preprocessing.")
        return

    # Preprocessors for numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # sparse_output=False: GaussianNB (and a few plot paths) cannot accept a
        # sparse matrix, and ColumnTransformer would otherwise return one.
        # max_categories/min_frequency stop a high-cardinality column (IDs, free
        # text) from exploding into thousands of one-hot columns.
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False,
                                 min_frequency=0.01, max_categories=25))
    ])

    transformers = []
    if numeric_cols:
        transformers.append(('num', numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(('cat', categorical_transformer, categorical_cols))
    preprocessor = ColumnTransformer(transformers=transformers)

    # Classifier setup
    selected_classifier = classifier_dropdown.get()
    classifier_parameters = {param: var.get() for param, var in classifier_params[selected_classifier].items()}

    # Use the dynamically updated penalty dropdown
    if selected_classifier == 'Logistic Regression':
        penalty = penalty_dropdown.get()
        # scikit-learn >= 1.2 expects the Python object None, not the string 'none'.
        classifier_parameters['penalty'] = None if penalty in ('none', 'None', '') else penalty
        classifier_parameters['solver'] = solver_logit_dropdown.get()
        if classifier_parameters['penalty'] == 'elasticnet':
            classifier_parameters['l1_ratio'] = classifier_params[selected_classifier]['l1_ratio'].get()
        else:
            # l1_ratio is only valid with penalty='elasticnet'.
            classifier_parameters.pop('l1_ratio', None)

    # Convert hidden_layer_sizes text to a tuple for MLPClassifier (no eval()).
    if selected_classifier == 'Neural Network (MLP)':
        try:
            classifier_parameters['hidden_layer_sizes'] = parse_hidden_layer_sizes(
                classifier_parameters['hidden_layer_sizes'])
        except (ValueError, SyntaxError) as e:
            messagebox.showerror(
                "Invalid parameter",
                f"'Hidden Layer Sizes' must look like  100  or  100,50  or  (100,)\n\n{e}")
            return


    # Ensure SVM classifier has probability=True
    if selected_classifier == 'SVM/SVC':
        classifier_parameters['probability'] = True

    # Build classifier instance
    try:
        classifier = classifier_classes[selected_classifier](**classifier_parameters)
    except TypeError as e:
        messagebox.showerror("Invalid parameter", f"Could not build {selected_classifier}:\n\n{e}")
        return

    # Full pipeline: preprocessing + classifier
    clf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    # Stratified split so every class is represented in both train and test.
    # Falls back to a plain split if any class is too small to stratify.
    min_class_count = np.min(np.bincount(y_encoded))
    stratify = y_encoded if min_class_count >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=stratify)

    # Training the model
    try:
        clf_pipeline.fit(X_train, y_train)
        y_pred = clf_pipeline.predict(X_test)
    except Exception as e:
        messagebox.showerror("Training failed", f"{type(e).__name__}: {e}")
        clf_pipeline = None
        return

    # Evaluating the model
    target_names = class_names if len(class_names) == len(np.unique(y_test)) else None
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)

    # Display results
    results_text_middle.insert(tk.END, "\n" + "=" * 93)
    results_text_middle.insert(tk.END, f"\nClassifier Used: {selected_classifier}  ({'binary' if is_binary_task else str(len(class_names)) + '-class'})\n")
    results_text_middle.insert(tk.END, f"Parameters Used: {classifier_parameters}\n")
    results_text_middle.insert(tk.END, f"Numeric features: {numeric_cols or 'none'}\n")
    results_text_middle.insert(tk.END, f"Categorical features: {categorical_cols or 'none'}\n")
    if dropped_cols:
        results_text_middle.insert(tk.END, f"Dropped (constant/empty): {dropped_cols}\n")
    results_text_middle.insert(tk.END, f"\nAccuracy: {accuracy:.4f}\n\n")
    results_text_middle.insert(tk.END, f"Confusion Matrix:\n{conf_matrix}\n\n")
    results_text_middle.insert(tk.END, f"Classification Report:\n{class_report}\n")
    results_text_middle.see(tk.END)

    # Call to save results to file
    save_results_to_file(selected_classifier, clf_pipeline.named_steps['classifier'].get_params(), accuracy, conf_matrix, class_report)

def save_results_to_file(classifier, params, accuracy, conf_matrix, class_report):
    with open("model_evaluation_results.txt", "a") as file:
        file.write(f"==========================================================================\n")
        file.write(f"Classifier: {classifier}\n")
        file.write(f"Parameters: {params}\n")
        for param, value in params.items():
            file.write(f"  {param}: {value}\n")
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Confusion Matrix:\n{conf_matrix}\n")
        file.write(f"Classification Report:\n{class_report}\n")
        file.write("=" * 60 + "\n")

# PLOT DROP DOWN SECTION : SECTION 3
plot_options = {
    'ROC Curve': 'roc_curve',
    'Precision-Recall Curve': 'precision_recall_curve',
    'PCA Explained Variance': 'pca_explained_variance',
    'PCA 2D Projection': 'pca_2d_projection',
    'Cumulative Gains Curve': 'cumulative_gains_curve',
    'Calibration Plot': 'calibration_plot',
    'Lift Curve': 'lift_curve',    
    'Silhouette Analysis': 'silhouette_analysis'
}


def setup_plot_ui(root):
    # Create a frame for plot dropdown
    plot_frame = tk.Frame(root)
    plot_frame.pack(fill=tk.BOTH, expand=True)
    
    # Plot Selection Dropdown
    tk.Label(plot_frame, text="Select Plot:").grid(row=0, column=0, padx=170, pady=10)
    global plot_dropdown
    plot_dropdown = ttk.Combobox(plot_frame, values=list(plot_options.keys()), state="readonly")
    plot_dropdown.grid(row=0, column=1, padx=170, pady=10)
    plot_dropdown.bind("<<ComboboxSelected>>", on_plot_selected)
    
    return plot_dropdown


def on_plot_selected(event):
    global X_test, y_test, clf_pipeline, label_encoder
    plot_type = plot_dropdown.get()

    if clf_pipeline is None or X_test is None or y_test is None:
        messagebox.showinfo("Train first", "Train and evaluate a model before generating plots.")
        return

    try:
        if plot_type == 'ROC Curve':
            plot_roc_curve()
        elif plot_type == 'Precision-Recall Curve':
            plot_precision_recall_curve()
        elif plot_type == 'PCA Explained Variance':
            plot_pca_explained_variance()
        elif plot_type == 'PCA 2D Projection':
            plot_pca_2d_projection(transformed_test_matrix(), y_test)
        elif plot_type == 'Cumulative Gains Curve':
            plot_cumulative_gains_curve(np.asarray(y_test), clf_pipeline.predict_proba(X_test))
        elif plot_type == 'Calibration Plot':
            plot_calibration_curve(np.asarray(y_test), clf_pipeline.predict_proba(X_test), class_names)
        elif plot_type == 'Lift Curve':
            plot_adaptive_lift_curve(np.asarray(y_test), clf_pipeline.predict_proba(X_test),
                                     list(clf_pipeline.classes_))
        elif plot_type == 'Silhouette Analysis':
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score

            mat = transformed_test_matrix()
            range_n_clusters = [n for n in (2, 3, 4, 5) if n < len(mat)]
            silhouette_avg = []
            for n_clusters in range_n_clusters:
                labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=10).fit_predict(mat)
                silhouette_avg.append(silhouette_score(mat, labels))
            show_plot_and_text(
                "Silhouette Analysis",
                "This plot helps determine the optimal number of clusters by providing a "
                "silhouette score for each cluster count. The silhouette score measures how "
                "similar an object is to its own cluster compared to other clusters.",
                plot_silhouette_analysis_fig, range_n_clusters, silhouette_avg)
    except Exception as e:
        messagebox.showerror("Plot error", f"Could not draw '{plot_type}':\n\n{type(e).__name__}: {e}")

def cls_label(encoded_value):
    """Map an encoded class (0, 1, 2, ...) back to its display name."""
    try:
        return class_names[int(encoded_value)]
    except (TypeError, ValueError, IndexError):
        return str(encoded_value)


def show_plot_and_text(title, text, plot_function, *plot_args):
    popup = Toplevel()
    popup.title(title)

    fig = plt.figure(figsize=(7, 5))
    try:
        plot_function(fig, *plot_args)
    except Exception:
        plt.close(fig)
        popup.destroy()
        raise
    fig.tight_layout()

    fig_canvas = FigureCanvasTkAgg(fig, master=popup)
    fig_canvas.draw()
    fig_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    text_widget = tk.Text(popup, height=5, wrap="word")
    text_widget.insert(tk.END, text)
    text_widget.pack(fill=tk.X)
    text_widget.config(state=tk.DISABLED)

    # Release the matplotlib figure when the window closes (avoids a slow leak).
    popup.protocol("WM_DELETE_WINDOW", lambda: (plt.close(fig), popup.destroy()))


def plot_silhouette_analysis_fig(fig, range_n_clusters, silhouette_avg):
    ax = fig.add_subplot(111)
    ax.plot(range_n_clusters, silhouette_avg, marker='o', label='Silhouette Score')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Analysis for Optimal Cluster Count')
    ax.legend()

def _ovr_indices(model_classes):
    """Which probability columns to plot: just the positive column for a binary
    task, every column for multiclass one-vs-rest."""
    return [1] if len(model_classes) == 2 else list(range(len(model_classes)))


def plot_roc_curve():
    proba = clf_pipeline.predict_proba(X_test)
    model_classes = list(clf_pipeline.classes_)
    y_true = np.asarray(y_test)
    idxs = _ovr_indices(model_classes)

    curves = []
    for i in idxs:
        pos = model_classes[i]
        fpr, tpr, _ = roc_curve((y_true == pos).astype(int), proba[:, i])
        curves.append((cls_label(pos), fpr, tpr, auc(fpr, tpr)))

    def fig_func(fig):
        ax = fig.add_subplot(111)
        for j, (lbl, fpr, tpr, a) in enumerate(curves):
            ax.plot(fpr, tpr, lw=2, color=class_color(j, len(curves)),
                    label=f'{lbl} (AUC = {a:0.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set(xlim=(0, 1), ylim=(0, 1.05), xlabel='False Positive Rate',
               ylabel='True Positive Rate',
               title='ROC Curve' if len(curves) == 1 else 'ROC Curve (one-vs-rest)')
        ax.legend(loc='lower right')

    show_plot_and_text(
        "ROC Curve",
        "Performance across all classification thresholds. Area under the curve (AUC) "
        "summarises it: 1.0 is perfect, 0.5 is random.", fig_func)


def plot_precision_recall_curve():
    proba = clf_pipeline.predict_proba(X_test)
    model_classes = list(clf_pipeline.classes_)
    y_true = np.asarray(y_test)
    idxs = _ovr_indices(model_classes)

    curves = []
    for i in idxs:
        pos = model_classes[i]
        binary_true = (y_true == pos).astype(int)
        precision, recall, _ = precision_recall_curve(binary_true, proba[:, i])
        ap = average_precision_score(binary_true, proba[:, i])
        curves.append((cls_label(pos), recall, precision, ap))

    def fig_func(fig):
        ax = fig.add_subplot(111)
        for j, (lbl, recall, precision, ap) in enumerate(curves):
            ax.plot(recall, precision, lw=2, color=class_color(j, len(curves)),
                    label=f'{lbl} (AP = {ap:0.3f})')
        ax.set(xlabel='Recall', ylabel='Precision',
               title='Precision-Recall' if len(curves) == 1 else 'Precision-Recall (one-vs-rest)')
        ax.legend(loc='lower left')

    show_plot_and_text(
        "Precision-Recall Curve",
        "The trade-off between precision and recall. Average precision (AP) is the area "
        "under this curve - useful when classes are imbalanced.", fig_func)


def plot_calibration_curve(y_true, y_proba, classes):
    model_classes = list(clf_pipeline.classes_)
    y_true = np.asarray(y_true)
    idxs = _ovr_indices(model_classes)

    def fig_func(fig):
        ax = fig.add_subplot(111)
        for j, i in enumerate(idxs):
            pos = model_classes[i]
            prob_true, prob_pred = calibration_curve((y_true == pos).astype(int),
                                                     y_proba[:, i], n_bins=10)
            ax.plot(prob_pred, prob_true, marker='o', color=class_color(j, len(idxs)),
                    label=cls_label(pos))
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
        ax.set(xlabel='Mean predicted probability', ylabel='Fraction of positives',
               title='Calibration')
        ax.legend()
        ax.grid(True)

    show_plot_and_text(
        "Calibration Plot",
        "Compares predicted probabilities with observed frequencies. A well-calibrated "
        "model sits on the diagonal: of the cases it gives ~0.7, about 70% are positive.",
        fig_func)


def plot_pca_explained_variance():
    X = transformed_test_matrix()
    n_comp = min(X.shape)
    pca = PCA(n_components=n_comp).fit(X)

    def fig_func(fig):
        ax = fig.add_subplot(111)
        ax.plot(np.arange(1, n_comp + 1), np.cumsum(pca.explained_variance_ratio_), marker='o')
        ax.set(xlabel='Number of components', ylabel='Cumulative explained variance',
               title='PCA Explained Variance')
        ax.grid(True)

    show_plot_and_text(
        "PCA Explained Variance",
        "How much of the total variance is captured as principal components are added. "
        "Computed on the fully preprocessed feature matrix.", fig_func)


def plot_pca_2d_projection(X, y):
    X = np.asarray(X)
    y = np.asarray(y)
    coords = PCA(n_components=2).fit_transform(X)

    def fig_func(fig):
        ax = fig.add_subplot(111)
        classes = np.unique(y)
        for i, c in enumerate(classes):
            m = y == c
            ax.scatter(coords[m, 0], coords[m, 1], s=40, alpha=0.7,
                       color=class_color(i, len(classes)), label=cls_label(c))
        ax.set(xlabel='Principal Component 1', ylabel='Principal Component 2',
               title='PCA 2D Projection')
        ax.legend(loc='best')
        ax.grid(True)

    show_plot_and_text(
        "PCA 2D Projection",
        "The preprocessed features projected onto their first two principal components - "
        "a rough view of how separable the classes are.", fig_func)


def plot_cumulative_gains_curve(y_true, y_proba):
    model_classes = list(clf_pipeline.classes_)
    y_true = np.asarray(y_true)
    trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)  # renamed in NumPy 2.0
    idxs = _ovr_indices(model_classes)

    series = []
    for i in idxs:
        pos = model_classes[i]
        order = np.argsort(y_proba[:, i])[::-1]
        hits = (y_true[order] == pos).astype(int)
        gains = np.cumsum(hits) / max(hits.sum(), 1)
        frac = np.arange(1, len(gains) + 1) / len(gains)
        series.append((cls_label(pos), np.r_[0, frac], np.r_[0, gains]))

    def fig_func(fig):
        ax = fig.add_subplot(111)
        for j, (lbl, x, yv) in enumerate(series):
            ax.plot(x, yv, lw=2, color=class_color(j, len(series)),
                    label=f'{lbl} (area = {trapz(yv, x):.3f})')
        ax.plot([0, 1], [0, 1], '--', color='gray', label='Baseline (random)')
        ax.set(xlabel='Fraction of sample (ranked by score)',
               ylabel='Fraction of positives captured', title='Cumulative Gains')
        ax.legend(loc='lower right')
        ax.grid(True)

    show_plot_and_text(
        "Cumulative Gains Curve",
        "If you act on the highest-scoring cases first, what share of the positives do you "
        "capture? The further above the diagonal, the better the ranking.", fig_func)

def plot_adaptive_lift_curve(y_true, y_scores, class_labels):
    y_true = np.asarray(y_true)
    idxs = _ovr_indices(class_labels)
    baseline = {i: max(np.mean(y_true == class_labels[i]), 1e-12) for i in idxs}

    def fig_func(fig):
        ax = fig.add_subplot(111)
        for j, i in enumerate(idxs):
            binary_true = (y_true == class_labels[i]).astype(int)
            precision, recall, _ = precision_recall_curve(binary_true, y_scores[:, i])
            ax.plot(recall * 100, precision / baseline[i], marker='.', markersize=4,
                    color=class_color(j, len(idxs)), label=cls_label(class_labels[i]))
        ax.plot([0, 100], [1, 1], 'k--', label='Baseline (random)')
        ax.set(xlabel='% of samples covered (recall)', ylabel='Lift',
               title='Lift Curve' if len(idxs) == 1 else 'Lift Curves (one-vs-rest)')
        ax.legend()
        ax.grid(True)

    show_plot_and_text(
        "Lift Curve",
        "How many times better than random the model is at finding a class, as you widen "
        "the fraction of the sample you act on.", fig_func)

def main():
    root = tk.Tk()
    root.title("Classifier GUI Application")
    root.geometry("1200x900")
    
    global target_var, info_text, classifier_dropdown, results_text, plot_dropdown, classifier_params, plot_text, solver_logit_dropdown, penalty_label, penalty_dropdown
    classifier_params = initialize_classifier_params()

    target_var = tk.StringVar()
    info_text = tk.StringVar()
    
    classifier_dropdown, train_button, results_text, _ = setup_ui(root)  # Remove plot_dropdown from setup_ui return values
    setup_plot_ui(root)
    
    root.mainloop()

if __name__ == "__main__":
    main()
