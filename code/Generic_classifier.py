import tkinter as tk
from tkinter import ttk, filedialog, Toplevel
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
    else:
        messagebox.showinfo("Info", "No file selected.")

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
            'learning_rate': tk.DoubleVar(value=1.0),
            'algorithm': tk.StringVar(value='SAMME')
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
    global global_df, label_encoder, X_test, y_test, clf_pipeline
    if global_df is None:
        messagebox.showerror("Error", "No dataset loaded. Please upload a dataset.")
        return

    target_variable = target_var.get()
    if target_variable not in global_df.columns:
        messagebox.showerror("Error", f"Target variable '{target_variable}' not found in the dataset.")
        return

    # Separating features and target
    X = global_df.drop(target_variable, axis=1)
    y = global_df[target_variable]

    # Check if the target data needs encoding
    if not np.issubdtype(y.dtype, np.number):
        label_encoder.fit(y)
        y_encoded = label_encoder.transform(y)
    else:
        y_encoded = y

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['int', 'float']).columns.tolist()

    # Preprocessors for numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combining preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Classifier setup
    selected_classifier = classifier_dropdown.get()
    classifier_parameters = {param: var.get() for param, var in classifier_params[selected_classifier].items()}

    # Use the dynamically updated penalty dropdown
    if selected_classifier == 'Logistic Regression':
        classifier_parameters['penalty'] = penalty_dropdown.get()
        classifier_parameters['solver'] = solver_logit_dropdown.get()
        if classifier_parameters['penalty'] == 'elasticnet':
            classifier_parameters['l1_ratio'] = classifier_params[selected_classifier]['l1_ratio'].get()

    # Convert hidden_layer_sizes to tuple for MLPClassifier
    if selected_classifier == 'Neural Network (MLP)':
        classifier_parameters['hidden_layer_sizes'] = eval(classifier_parameters['hidden_layer_sizes'])
        classifier_parameters['solver'] = classifier_parameters.pop('solver')
        
    # Ensure SVM classifier has probability=True
    if selected_classifier == 'SVM/SVC':
        classifier_parameters['probability'] = True

    # Build classifier instance
    classifier = classifier_classes[selected_classifier](**classifier_parameters)

    # Full pipeline: preprocessing + classifier
    clf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

    # Training the model
    clf_pipeline.fit(X_train, y_train)
    y_pred = clf_pipeline.predict(X_test)

    # Evaluating the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Display results
    results_text_middle.insert(tk.END, f"\n=============================================================================================")
    results_text_middle.insert(tk.END, f"\nClassifier Used: {selected_classifier}\n\n")
    results_text_middle.insert(tk.END, f"\nParameters Used: {classifier_parameters}\n\n")
    results_text_middle.insert(tk.END, f"\nAccuracy: {accuracy}\n\n")
    results_text_middle.insert(tk.END, f"Confusion Matrix:\n{conf_matrix}\n\n")
    results_text_middle.insert(tk.END, f"Classification Report:\n{class_report}")

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
    
    if plot_type == 'ROC Curve':
        plot_roc_curve()
    elif plot_type == 'Precision-Recall Curve':
        plot_precision_recall_curve()
    elif plot_type == 'PCA Explained Variance':
        plot_pca_explained_variance()
    elif plot_type == 'PCA 2D Projection':
        plot_pca_2d_projection(X_test, y_test)
    elif plot_type == 'Cumulative Gains Curve':
        y_scores = clf_pipeline.predict_proba(X_test)[:, 1]
        plot_cumulative_gains_curve(y_test, y_scores)
    elif plot_type == 'Calibration Plot':
        y_proba = clf_pipeline.predict_proba(X_test)
        try:
            # Check if the label encoder has been used and has classes_
            if hasattr(label_encoder, 'classes_'):
                plot_calibration_curve(y_test, y_proba, label_encoder.classes_)
            else:
                # If no encoding was done, use the unique values from y_test as class labels
                unique_classes = np.unique(y_test)
                plot_calibration_curve(y_test, y_proba, [str(cls) for cls in unique_classes])
        except AttributeError as e:
            print_to_output(f"Error: {e}")
            plot_calibration_curve(y_test, y_proba, [str(cls) for cls in np.unique(y_test)])
    elif plot_type == 'Lift Curve':
        y_scores = clf_pipeline.predict_proba(X_test)
        class_labels = clf_pipeline.classes_ 
        plot_adaptive_lift_curve(y_test, y_scores, class_labels)
    elif plot_type == 'Silhouette Analysis':
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_samples, silhouette_score

        range_n_clusters = [2, 3, 4, 5]
        silhouette_avg = []
        for n_clusters in range_n_clusters:
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(X_test)
            silhouette_avg.append(silhouette_score(X_test, cluster_labels))

        show_plot_and_text("Silhouette Analysis", "This plot helps determine the optimal number of clusters by providing a silhouette score for each cluster count. The silhouette score measures how similar an object is to its own cluster compared to other clusters.", 
                           plot_silhouette_analysis_fig, range_n_clusters, silhouette_avg)

def show_plot_and_text(title, text, plot_function, *plot_args):
    popup = Toplevel()
    popup.title(title)
    
    fig = plt.figure()
    plot_function(fig, *plot_args)

    # Convert the plot to a Tkinter PhotoImage
    fig_canvas = FigureCanvasTkAgg(fig, master=popup)
    fig_canvas.draw()
    fig_canvas.get_tk_widget().pack()

    text_widget = tk.Text(popup, height=10, wrap="word", font=("Helvetica", 16))
    text_widget.insert(tk.END, text)
    text_widget.pack(fill=tk.BOTH, expand=True)
    text_widget.config(state=tk.DISABLED)


def plot_silhouette_analysis_fig(fig, range_n_clusters, silhouette_avg):
    plt.figure(fig.number)
    plt.plot(range_n_clusters, silhouette_avg, marker='o', label='Silhouette Score')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis for Optimal Cluster Count')
    plt.legend()

def plot_roc_curve():
    from sklearn.preprocessing import label_binarize
    # Binarize the labels for multi-class ROC
    y_bin = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_bin.shape[1]

    # Get the classifier probabilities
    probabilities = clf_pipeline.predict_proba(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    def plot_roc_curve_fig(fig):
        plt.figure(fig.number)
        colors = ['blue', 'green', 'red']
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Curve')
        plt.legend(loc="lower right")

    show_plot_and_text("ROC Curve", "This plot shows the performance of the classification model at all classification thresholds. The area under the curve (AUC) can be used as a summary.", plot_roc_curve_fig)

def plot_precision_recall_curve():
    from sklearn.preprocessing import label_binarize
    # Binarize the output labels for multi-class precision-recall
    y_bin = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_bin.shape[1]

    # Compute Precision-Recall curve and AUC for each class
    precision = dict()
    recall = dict()
    average_precision = dict()

    # Get probabilities for each class
    probabilities = clf_pipeline.predict_proba(X_test)

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], probabilities[:, i])
        average_precision[i] = average_precision_score(y_bin[:, i], probabilities[:, i])

    def plot_precision_recall_curve_fig(fig):
        plt.figure(fig.number)
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
        for i, color in zip(range(n_classes), colors[:n_classes]):
            plt.plot(recall[i], precision[i], color=color, lw=2,
                     label='Precision-Recall curve of class {0} (area = {1:0.2f})'.format(i, average_precision[i]))

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve per Class')
        plt.legend(loc="lower right")

    show_plot_and_text("Precision-Recall Curve", "This plot shows the trade-off between precision and high recall. The area under the curve represents the expected precision at a given recall rate. High area suggests high precision and high recall.", plot_precision_recall_curve_fig)

def plot_calibration_curve(y_true, y_proba, classes):
    # y_proba should be the output from predict_proba and have shape (n_samples, n_classes)
    def plot_calibration_curve_fig(fig):
        plt.figure(fig.number)
        # Plot calibration curve for each class
        for i, class_name in enumerate(classes):
            # One-vs-rest for the current class
            prob_true, prob_pred = calibration_curve(y_true == i, y_proba[:, i], n_bins=10)

            # Plotting
            plt.plot(prob_pred, prob_true, marker='o', label=f'Class {class_name}')

        # Plot perfectly calibrated
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curves')
        plt.legend()
        plt.grid(True)

    show_plot_and_text("Calibration Plot", "This plot compares predicted probabilities with actual outcomes. A well-calibrated model will have probabilities that match the actual likelihood of an event. For example, if a model assigns a 0.7 probability to an event, that event should occur approximately 70% of the time. Calibration plots are useful for understanding if your model's predicted probabilities are accurate and reliable.", plot_calibration_curve_fig)

def plot_pca_explained_variance():
    pca = PCA().fit(X_test)
    
    def plot_pca_explained_variance_fig(fig):
        plt.figure(fig.number)
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')

    show_plot_and_text("PCA Explained Variance", "This plot shows the amount of variance each PCA component captures from the data. It helps in understanding the contribution of each dimension to the variance in the dataset.", plot_pca_explained_variance_fig)

def plot_pca_2d_projection(X, y):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import pandas as pd

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)

    # Prepare the data for plotting
    pca_df = pd.DataFrame(data = principal_components,
                          columns = ['Principal Component 1', 'Principal Component 2'])
    pca_df['Target'] = pd.Series(y).astype(str)  # Ensure y is a string for consistent handling

    def plot_pca_2d_projection_fig(fig):
        plt.figure(fig.number)
        colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']

        # Extract unique classes
        classes = pca_df['Target'].unique()

        # Plot each class with different color
        for i, label in enumerate(classes):
            idx = pca_df['Target'] == label
            plt.scatter(pca_df.loc[idx, 'Principal Component 1'],
                        pca_df.loc[idx, 'Principal Component 2'],
                        color=colors[i % len(colors)], label=label, s=50)

        plt.title('PCA 2D Projection')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(loc='best')
        plt.grid(True)

    show_plot_and_text("PCA 2D Projection", "This plot visualizes the data in two principal component axes. It helps to understand the variance in data and how well different classes are separable in a reduced dimension.", plot_pca_2d_projection_fig)

def plot_cumulative_gains_curve(y_true, y_scores):
    # Convert y_true to a numpy array to use positional indexing
    y_true = np.array(y_true)    
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    cum_gains = np.cumsum(y_true_sorted) / np.sum(y_true_sorted)
    cum_perc = np.arange(start=1, stop=len(cum_gains) + 1) / len(cum_gains)

    def plot_cumulative_gains_curve_fig(fig):
        plt.figure(fig.number)
        plt.plot(cum_perc, cum_gains, color='blue', lw=2, label='Cumulative Gains Curve')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Baseline')
        plt.title('Cumulative Gains Curve')
        plt.xlabel('Percentage of Sample')
        plt.ylabel('Gain')
        plt.legend()
        plt.grid(True)

    show_plot_and_text("Cumulative Gains Curve", "AUC for Cumulative Gains: {:.2f}".format(np.trapezoid(cum_gains, cum_perc)), plot_cumulative_gains_curve_fig)

def plot_adaptive_lift_curve(y_true, y_scores, class_labels):
    # Determine if binary or multiclass
    is_binary = len(class_labels) == 2

    baseline = np.array([np.mean(y_true == cl) for cl in class_labels])
    
    def plot_adaptive_lift_curve_fig(fig):
        plt.figure(fig.number)

        if is_binary:
            # Treat the problem as binary classification
            binary_y_true = (y_true == class_labels[1]).astype(int)
            precision, recall, _ = precision_recall_curve(binary_y_true, y_scores[:, 1])
            lift = precision / baseline[1]
            sample_coverage = recall * 100
            plt.plot(sample_coverage, lift, marker='o', label='Lift Curve for Positive Class')
        else:
            # Handle each class separately for multiclass
            for i, class_label in enumerate(class_labels):
                binary_y_true = (y_true == class_label).astype(int)
                precision, recall, _ = precision_recall_curve(binary_y_true, y_scores[:, i])
                lift = precision / baseline[i]
                sample_coverage = recall * 100
                plt.plot(sample_coverage, lift, marker='o', label=f'Lift for {class_label}')

        plt.plot([0, 100], [1, 1], 'k--', label='Baseline (Random)')
        plt.xlabel('Percentage of Samples Covered')
        plt.ylabel('Lift')
        plt.title('Lift Curve(s)')
        plt.legend()
        plt.grid(True)

    if is_binary:
        show_plot_and_text("Lift Curve", "Lift curve for binary classification shows how much better the classifier performs for the positive class compared to random guessing.", plot_adaptive_lift_curve_fig)
    else:
        show_plot_and_text("Lift Curve", "Lift curves for each class show how much better the classifier performs at various thresholds compared to random guessing for that class.", plot_adaptive_lift_curve_fig)

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
