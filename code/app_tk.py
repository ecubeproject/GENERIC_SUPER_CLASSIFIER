"""Tkinter front-end for the generic tabular classifier.

All the modelling / plotting logic lives in the sibling modules
(``data_io``, ``profiling``, ``pipeline``, ``plots``); this file is only
widgets, layout and event wiring.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Toplevel

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import data_io
import profiling
import pipeline
import plots

# --- shared state -----------------------------------------------------------
global_df = None            # the uploaded DataFrame
last_result = None          # the most recent pipeline.TrainResult
loaded_model = None         # the most recent pipeline.SavedModel (Load Model / after training)
leaderboard = pipeline.Leaderboard()


# --- classifier parameter widgets -----------------------------------------

def _make_var(value):
    if isinstance(value, bool):
        return tk.BooleanVar(value=value)
    if isinstance(value, int):
        return tk.IntVar(value=value)
    if isinstance(value, float):
        return tk.DoubleVar(value=value)
    return tk.StringVar(value=str(value))


def initialize_classifier_params():
    """Wrap pipeline.DEFAULT_PARAMS in tk.Var objects for the widgets.

    'penalty' for Logistic Regression is skipped here - it is driven by its own
    solver-dependent dropdown.
    """
    out = {}
    for name, params in pipeline.DEFAULT_PARAMS.items():
        d = {}
        for key, value in params.items():
            if name == 'Logistic Regression' and key == 'penalty':
                continue
            d[key] = _make_var(value)
        out[name] = d
    return out


def update_penalty_options(event):
    solver = solver_logit_dropdown.get()
    penalties = pipeline.VALID_PENALTIES.get(solver, ['l2', 'none'])
    penalty_dropdown['values'] = penalties
    if penalty_dropdown.get() not in penalties:
        penalty_dropdown.set(penalties[0])


def on_classifier_change(event):
    global penalty_label, penalty_dropdown, solver_logit_dropdown, solver_label
    global label_c, entry_c, label_n_iter, label_l1_ratio, entry_l1_ratio
    selected_classifier = classifier_dropdown.get()
    for widget in classifier_params_frame.winfo_children():
        widget.destroy()

    params_ui = classifier_params[selected_classifier]
    row = 0
    for param, var in params_ui.items():
        label = tk.Label(classifier_params_frame, text=param.replace('_', ' ').title() + ":")
        label.grid(row=row, column=0, padx=170, pady=5)

        if isinstance(var, tk.BooleanVar):
            entry = ttk.Combobox(classifier_params_frame, textvariable=var,
                                 values=[True, False], state="disabled")
        elif isinstance(var, tk.StringVar):
            if param == 'kernel':
                entry = ttk.Combobox(classifier_params_frame, textvariable=var,
                                     values=['linear', 'poly', 'rbf', 'sigmoid'])
            elif param == 'activation':
                entry = ttk.Combobox(classifier_params_frame, textvariable=var,
                                     values=['relu', 'tanh', 'logistic'])
            elif param == 'solver' and selected_classifier == 'Logistic Regression':
                solver_label = tk.Label(classifier_params_frame, text="Solver:")
                solver_label.grid(row=row, column=0, padx=170, pady=5)
                solver_logit_dropdown = ttk.Combobox(
                    classifier_params_frame, textvariable=var,
                    values=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
                solver_logit_dropdown.bind("<<ComboboxSelected>>", update_penalty_options)
                solver_logit_dropdown.grid(row=row, column=1, padx=170, pady=5)
                penalty_label = tk.Label(classifier_params_frame, text="Penalty:")
                penalty_label.grid(row=row + 1, column=0, padx=170, pady=5)
                penalty_dropdown = ttk.Combobox(classifier_params_frame, state="readonly")
                penalty_dropdown.grid(row=row + 1, column=1, padx=170, pady=5)
                penalty_dropdown.set('l2')
                penalty_dropdown['values'] = pipeline.VALID_PENALTIES['lbfgs']
                row += 2
                continue
            elif param == 'solver' and selected_classifier == 'Neural Network (MLP)':
                entry = ttk.Combobox(classifier_params_frame, textvariable=var,
                                     values=['adam', 'sgd', 'lbfgs'])
            else:
                entry = ttk.Combobox(classifier_params_frame, textvariable=var)
        elif isinstance(var, tk.IntVar):
            if param == 'n_neighbors':
                entry = tk.Scale(classifier_params_frame, from_=1, to=20,
                                 orient='horizontal', variable=var)
            elif param == 'max_iter':
                entry = tk.Scale(classifier_params_frame, from_=1, to=5000,
                                 orient='horizontal', variable=var)
                label_n_iter = tk.Label(classifier_params_frame, text="n_iter:")
                label_n_iter.grid(row=row, column=0, padx=170, pady=5)
                entry.grid(row=row, column=1, padx=170, pady=5)
                row += 1
                continue
            elif param == 'num_leaves':
                entry = tk.Scale(classifier_params_frame, from_=1, to=20,
                                 orient='horizontal', variable=var)
            elif param == 'n_estimators':
                entry = tk.Scale(classifier_params_frame, from_=1, to=100,
                                 orient='horizontal', variable=var)
            else:
                entry = tk.Scale(classifier_params_frame, from_=1, to=50,
                                 orient='horizontal', variable=var)
        elif isinstance(var, tk.DoubleVar):
            if param == 'learning_rate':
                entry = tk.Scale(classifier_params_frame, from_=0.01, to=1,
                                 resolution=0.01, orient='horizontal', variable=var)
            elif param == 'C':
                label_c = tk.Label(classifier_params_frame, text="C:")
                label_c.grid(row=row, column=0, padx=170, pady=5)
                entry_c = tk.Scale(classifier_params_frame, from_=0.001, to=1,
                                   resolution=0.001, orient='horizontal', variable=var)
                entry_c.grid(row=row, column=1, padx=170, pady=5)
                row += 1
                continue
            elif param == 'l1_ratio':
                label_l1_ratio = tk.Label(classifier_params_frame, text="l1_ratio:")
                label_l1_ratio.grid(row=row, column=0, padx=170, pady=5)
                entry_l1_ratio = tk.Scale(classifier_params_frame, from_=0.0, to=1.0,
                                          resolution=0.01, orient='horizontal', variable=var)
                entry_l1_ratio.grid(row=row, column=1, padx=170, pady=5)
                row += 1
                continue
            else:
                entry = tk.Scale(classifier_params_frame, from_=0.01, to=1,
                                 resolution=0.01, orient='horizontal', variable=var)
        else:
            entry = ttk.Combobox(classifier_params_frame, textvariable=var)
        entry.grid(row=row, column=1, padx=170, pady=5)
        row += 1

    if selected_classifier == 'Logistic Regression':
        penalty_label.grid(row=row, column=0, padx=170, pady=5)
        penalty_dropdown.grid(row=row, column=1, padx=170, pady=5)


# --- events ---------------------------------------------------------------

def upload_file():
    global global_df
    file_path = filedialog.askopenfilename(
        filetypes=[("Data files", "*.csv *.xlsx"), ("CSV files", "*.csv"),
                   ("Excel files", "*.xlsx")])
    if not file_path:
        return  # user cancelled the dialog
    try:
        global_df = data_io.load_dataframe(file_path)
        _populate_feature_listbox(list(global_df.columns))
        results_text.insert(tk.END, "\n\n" + profiling.profile_report(global_df))
        results_text.see(tk.END)
    except Exception as e:
        messagebox.showerror("Error", f"Error processing file: {e}")


def _populate_feature_listbox(columns):
    feature_listbox.delete(0, tk.END)
    for col in columns:
        feature_listbox.insert(tk.END, col)
    feature_listbox.selection_set(0, tk.END)  # everything selected by default


def _selected_feature_columns(target):
    """The listbox selection minus the target, or None when the user has left
    everything (or nothing) selected - which means 'use all columns'."""
    picked = [feature_listbox.get(i) for i in feature_listbox.curselection()]
    picked = [c for c in picked if c != target]
    all_but_target = [c for c in feature_listbox.get(0, tk.END) if c != target]
    if not picked or set(picked) == set(all_but_target):
        return None
    return picked


def _collect_spec():
    """Validate the shared inputs and return
    ``(target, classifier_name, params, feature_columns, balance)`` or None
    (after showing an error dialog)."""
    if global_df is None:
        messagebox.showerror("Error", "No dataset loaded. Please upload a dataset.")
        return None

    target = target_var.get()
    if target not in global_df.columns:
        messagebox.showerror("Error", f"Target variable '{target}' not found in the dataset.")
        return None

    y_raw = global_df.dropna(subset=[target])[target]
    if y_raw.nunique() < 2:
        messagebox.showerror("Error", f"Target '{target}' has only one class after "
                                      "dropping missing values.")
        return None
    if pipeline.looks_continuous(y_raw):
        proceed = messagebox.askyesno(
            "Target looks continuous",
            f"'{target}' has {y_raw.nunique()} distinct numeric values and looks like a "
            "regression target. This tool does classification.\n\nTreat it as "
            f"{y_raw.nunique()} separate classes anyway?")
        if not proceed:
            return None

    selected_classifier = classifier_dropdown.get()
    if not selected_classifier:
        messagebox.showerror("Error", "Please choose a classifier.")
        return None
    params = {p: v.get() for p, v in classifier_params[selected_classifier].items()}
    if selected_classifier == 'Logistic Regression':
        params['penalty'] = penalty_dropdown.get()
        params['solver'] = solver_logit_dropdown.get()

    return (target, selected_classifier, params,
            _selected_feature_columns(target), bool(balance_var.get()))


def _refresh_leaderboard():
    leaderboard_text.config(state=tk.NORMAL)
    leaderboard_text.delete("1.0", tk.END)
    leaderboard_text.insert(tk.END, leaderboard.table(sort_by="f1_macro"))
    leaderboard_text.config(state=tk.DISABLED)


def on_train_clicked():
    global last_result, loaded_model
    spec = _collect_spec()
    if spec is None:
        return
    target, selected_classifier, params, feature_columns, balance = spec

    try:
        result = pipeline.train_and_evaluate(
            global_df, target, selected_classifier, params,
            feature_columns=feature_columns, balance_classes=balance)
    except ValueError as e:
        messagebox.showerror("Cannot train", str(e))
        return
    except (TypeError, SyntaxError) as e:
        messagebox.showerror("Invalid parameter",
                             f"Could not build {selected_classifier}:\n\n{e}")
        return
    except Exception as e:
        messagebox.showerror("Training failed", f"{type(e).__name__}: {e}")
        return

    last_result = result
    loaded_model = None  # a fresh train supersedes any loaded model for Save
    kind = 'binary' if result.is_binary else f'{len(result.class_names)}-class'
    results_text_middle.insert(tk.END, "\n" + "=" * 93)
    results_text_middle.insert(tk.END, f"\nClassifier Used: {selected_classifier}  ({kind})\n")
    results_text_middle.insert(tk.END, f"Parameters Used: {result.params}\n")
    results_text_middle.insert(tk.END, f"Features used: {result.input_columns}\n")
    results_text_middle.insert(tk.END, f"Numeric features: {result.numeric_cols or 'none'}\n")
    results_text_middle.insert(tk.END, f"Categorical features: {result.categorical_cols or 'none'}\n")
    if result.dropped_cols:
        results_text_middle.insert(tk.END, f"Dropped (constant/empty): {result.dropped_cols}\n")
    if result.balanced:
        results_text_middle.insert(tk.END, f"Class balancing: {result.balance_applied}\n")
    results_text_middle.insert(tk.END, f"\nAccuracy: {result.accuracy:.4f}\n\n")
    results_text_middle.insert(tk.END, f"Confusion Matrix:\n{result.confusion_matrix}\n\n")
    results_text_middle.insert(tk.END, f"Classification Report:\n{result.classification_report}\n")
    results_text_middle.see(tk.END)

    leaderboard.add(result)
    _refresh_leaderboard()
    _save_results_to_file(selected_classifier,
                          result.pipeline.named_steps['classifier'].get_params(),
                          result.accuracy, result.confusion_matrix,
                          result.classification_report)


def on_cross_validate_clicked():
    spec = _collect_spec()
    if spec is None:
        return
    target, selected_classifier, params, feature_columns, balance = spec
    try:
        k = int(cv_k_var.get())
    except (tk.TclError, ValueError):
        messagebox.showerror("Error", "Folds (k) must be a whole number >= 2.")
        return

    try:
        cv = pipeline.cross_validate_model(
            global_df, target, selected_classifier, params,
            feature_columns=feature_columns, balance_classes=balance, k=k)
    except ValueError as e:
        messagebox.showerror("Cannot cross-validate", str(e))
        return
    except Exception as e:
        messagebox.showerror("Cross-validation failed", f"{type(e).__name__}: {e}")
        return

    results_text_middle.insert(tk.END, "\n" + "=" * 93 + "\n")
    results_text_middle.insert(tk.END, cv.summary() + "\n")
    results_text_middle.see(tk.END)
    leaderboard.add(cv)
    _refresh_leaderboard()


def on_save_model():
    if loaded_model is None and last_result is None:
        messagebox.showinfo("Nothing to save", "Train a model first.")
        return
    path = filedialog.asksaveasfilename(
        defaultextension=".joblib", filetypes=[("Model files", "*.joblib")])
    if not path:
        return
    try:
        if last_result is not None:
            pipeline.save_model(last_result, path)
        else:
            import joblib
            joblib.dump(loaded_model, path)
        messagebox.showinfo("Saved", f"Model written to:\n{path}")
    except Exception as e:
        messagebox.showerror("Save failed", f"{type(e).__name__}: {e}")


def on_load_model():
    global loaded_model, last_result
    path = filedialog.askopenfilename(filetypes=[("Model files", "*.joblib")])
    if not path:
        return
    try:
        loaded_model = pipeline.load_model(path)
        last_result = None  # the loaded model has no held-out set / plots
    except Exception as e:
        messagebox.showerror("Load failed", f"{type(e).__name__}: {e}")
        return
    results_text_middle.insert(tk.END, "\n" + "=" * 93 + "\n")
    results_text_middle.insert(tk.END, "Loaded model:\n" + loaded_model.describe() + "\n")
    results_text_middle.see(tk.END)


def on_predict_file():
    model = loaded_model
    if model is None and last_result is not None:
        model = pipeline.SavedModel(
            pipeline=last_result.pipeline, label_encoder=last_result.label_encoder,
            class_names=last_result.class_names, input_columns=last_result.input_columns,
            numeric_cols=last_result.numeric_cols,
            categorical_cols=last_result.categorical_cols,
            classifier_name=last_result.classifier_name)
    if model is None:
        messagebox.showinfo("No model", "Train or load a model first.")
        return

    in_path = filedialog.askopenfilename(
        title="Data to score",
        filetypes=[("Data files", "*.csv *.xlsx")])
    if not in_path:
        return
    try:
        df = data_io.load_dataframe(in_path)
        preds = pipeline.predict_dataframe(model, df)
    except Exception as e:
        messagebox.showerror("Prediction failed", f"{type(e).__name__}: {e}")
        return

    out_path = filedialog.asksaveasfilename(
        title="Save predictions", defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")])
    if out_path:
        df.join(preds).to_csv(out_path, index=False)
    results_text_middle.insert(tk.END, "\n" + "=" * 93 + "\n")
    results_text_middle.insert(tk.END, f"Predictions for {in_path} (first rows):\n")
    results_text_middle.insert(tk.END, preds.head(10).to_string() + "\n")
    results_text_middle.see(tk.END)


def _save_results_to_file(classifier, params, accuracy, conf_matrix, class_report):
    with open("model_evaluation_results.txt", "a") as file:
        file.write("=" * 74 + "\n")
        file.write(f"Classifier: {classifier}\n")
        file.write(f"Parameters: {params}\n")
        for param, value in params.items():
            file.write(f"  {param}: {value}\n")
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Confusion Matrix:\n{conf_matrix}\n")
        file.write(f"Classification Report:\n{class_report}\n")
        file.write("=" * 60 + "\n")


def on_plot_selected(event):
    plot_type = plot_dropdown.get()
    if last_result is None:
        messagebox.showinfo("Train first", "Train and evaluate a model before generating plots.")
        return
    try:
        _show_plot(plot_type)
    except Exception as e:
        messagebox.showerror("Plot error",
                             f"Could not draw '{plot_type}':\n\n{type(e).__name__}: {e}")


def _show_plot(label):
    spec = plots.PLOTS[label]
    popup = Toplevel()
    popup.title(label)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    try:
        plots.draw(label, ax, last_result)
    except Exception:
        plt.close(fig)
        popup.destroy()
        raise
    fig.tight_layout()

    fig_canvas = FigureCanvasTkAgg(fig, master=popup)
    fig_canvas.draw()
    fig_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    text_widget = tk.Text(popup, height=5, wrap="word")
    text_widget.insert(tk.END, spec['desc'])
    text_widget.pack(fill=tk.X)
    text_widget.config(state=tk.DISABLED)

    popup.protocol("WM_DELETE_WINDOW", lambda: (plt.close(fig), popup.destroy()))


# --- layout --------------------------------------------------------------

def setup_ui(root):
    upper_frame = tk.Frame(root)
    upper_frame.pack(fill=tk.BOTH, expand=True)
    middle_frame = tk.Frame(root)
    middle_frame.pack(fill=tk.BOTH, expand=True)

    upload_button = tk.Button(upper_frame, text="Upload Datafile", command=upload_file)
    upload_button.grid(row=0, column=0, columnspan=2, padx=170, pady=10)

    text_frame = tk.Frame(upper_frame, width=1200, height=320)
    text_frame.grid(row=1, column=0, columnspan=2, padx=170, pady=10)

    global results_text
    results_text = tk.Text(text_frame, width=120, height=16, wrap="none")
    h_scrollbar = tk.Scrollbar(text_frame, orient=tk.HORIZONTAL, command=results_text.xview)
    h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
    results_text.config(xscrollcommand=h_scrollbar.set)
    v_scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=results_text.yview)
    v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    results_text.config(yscrollcommand=v_scrollbar.set)
    results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    tk.Label(middle_frame, text="Target Variable:").grid(row=0, column=0, padx=170, pady=10)
    target_entry = tk.Entry(middle_frame, textvariable=target_var)
    target_entry.grid(row=0, column=1, padx=170, pady=10)

    # --- feature selection + Stage-3 options -----------------------------
    options_frame = tk.Frame(middle_frame)
    options_frame.grid(row=3, column=0, columnspan=2, padx=170, pady=10)

    tk.Label(options_frame, text="Feature columns (all = use everything):").grid(
        row=0, column=0, sticky="w")
    global feature_listbox
    lb_frame = tk.Frame(options_frame)
    lb_frame.grid(row=1, column=0, sticky="w")
    lb_scroll = tk.Scrollbar(lb_frame, orient=tk.VERTICAL)
    feature_listbox = tk.Listbox(lb_frame, selectmode=tk.EXTENDED, height=6, width=32,
                                 exportselection=False, yscrollcommand=lb_scroll.set)
    lb_scroll.config(command=feature_listbox.yview)
    feature_listbox.pack(side=tk.LEFT)
    lb_scroll.pack(side=tk.LEFT, fill=tk.Y)

    right = tk.Frame(options_frame)
    right.grid(row=1, column=1, sticky="nw", padx=20)
    tk.Checkbutton(right, text="Balance classes", variable=balance_var).grid(
        row=0, column=0, columnspan=2, sticky="w")
    tk.Label(right, text="CV folds (k):").grid(row=1, column=0, sticky="w")
    tk.Spinbox(right, from_=2, to=20, width=5, textvariable=cv_k_var).grid(
        row=1, column=1, sticky="w")
    tk.Button(right, text="Cross-validate", command=on_cross_validate_clicked).grid(
        row=2, column=0, columnspan=2, sticky="we", pady=(6, 0))
    tk.Button(right, text="Save Model", command=on_save_model).grid(
        row=3, column=0, sticky="we", pady=(6, 0))
    tk.Button(right, text="Load Model", command=on_load_model).grid(
        row=3, column=1, sticky="we", pady=(6, 0))
    tk.Button(right, text="Predict on File", command=on_predict_file).grid(
        row=4, column=0, columnspan=2, sticky="we", pady=(6, 0))

    tk.Label(middle_frame, text="Choose Classifier:").grid(row=1, column=0, padx=170, pady=10)
    global classifier_dropdown
    classifier_dropdown = ttk.Combobox(middle_frame, values=list(classifier_params.keys()),
                                       state="readonly")
    classifier_dropdown.grid(row=1, column=1, padx=170, pady=10)
    classifier_dropdown.bind("<<ComboboxSelected>>", on_classifier_change)

    global classifier_params_frame
    classifier_params_frame = tk.Frame(middle_frame)
    classifier_params_frame.grid(row=2, column=0, columnspan=2, padx=170, pady=10)

    train_button = tk.Button(middle_frame, text="Train and Evaluate", command=on_train_clicked)
    train_button.grid(row=4, column=0, columnspan=2, padx=170, pady=10)

    tk.Label(middle_frame, text="Results:", anchor="w", justify="left").grid(
        row=5, column=0, padx=170, pady=10)

    text_frame_middle = tk.Frame(middle_frame, width=1200, height=360)
    text_frame_middle.grid(row=6, column=0, columnspan=2, padx=170, pady=10)

    global results_text_middle
    results_text_middle = tk.Text(text_frame_middle, width=120, height=18, wrap="none")
    h_scrollbar_middle = tk.Scrollbar(text_frame_middle, orient=tk.HORIZONTAL,
                                      command=results_text_middle.xview)
    h_scrollbar_middle.pack(side=tk.BOTTOM, fill=tk.X)
    results_text_middle.config(xscrollcommand=h_scrollbar_middle.set)
    v_scrollbar_middle = tk.Scrollbar(text_frame_middle, orient=tk.VERTICAL,
                                      command=results_text_middle.yview)
    v_scrollbar_middle.pack(side=tk.RIGHT, fill=tk.Y)
    results_text_middle.config(yscrollcommand=v_scrollbar_middle.set)
    results_text_middle.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    tk.Label(middle_frame, text="Model leaderboard (sorted by macro F1):").grid(
        row=7, column=0, padx=170, pady=(10, 0), sticky="w")
    global leaderboard_text
    leaderboard_text = tk.Text(middle_frame, width=120, height=8, wrap="none")
    leaderboard_text.grid(row=8, column=0, columnspan=2, padx=170, pady=10)
    leaderboard_text.insert(tk.END, leaderboard.table())
    leaderboard_text.config(state=tk.DISABLED)


def setup_plot_ui(root):
    plot_frame = tk.Frame(root)
    plot_frame.pack(fill=tk.BOTH, expand=True)
    tk.Label(plot_frame, text="Select Plot:").grid(row=0, column=0, padx=170, pady=10)
    global plot_dropdown
    plot_dropdown = ttk.Combobox(plot_frame, values=list(plots.PLOTS.keys()), state="readonly")
    plot_dropdown.grid(row=0, column=1, padx=170, pady=10)
    plot_dropdown.bind("<<ComboboxSelected>>", on_plot_selected)


def main():
    root = tk.Tk()
    root.title("Classifier GUI Application")
    root.geometry("1200x900")

    global target_var, classifier_dropdown, results_text, plot_dropdown
    global classifier_params, solver_logit_dropdown, penalty_label, penalty_dropdown
    global balance_var, cv_k_var
    classifier_params = initialize_classifier_params()

    target_var = tk.StringVar()
    balance_var = tk.BooleanVar(value=False)
    cv_k_var = tk.IntVar(value=5)

    setup_ui(root)
    setup_plot_ui(root)

    root.mainloop()


if __name__ == "__main__":
    main()
