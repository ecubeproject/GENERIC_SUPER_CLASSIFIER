"""Streamlit front-end for the generic tabular classifier.

All modelling / plotting logic lives in the sibling UI-free modules
(``data_io``, ``profiling``, ``pipeline``, ``plots``); this file is only the
web UI - widgets, layout and session state.

Run locally:   streamlit run code/streamlit_app.py
"""

import io
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import data_io
import profiling
import pipeline
import plots

DATAFILES = Path(__file__).resolve().parent.parent / "datafiles"

st.set_page_config(page_title="Generic Super Classifier", page_icon="🧮",
                   layout="wide", initial_sidebar_state="expanded")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state():
    st.session_state.setdefault("df", None)
    st.session_state.setdefault("df_name", None)
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_metrics", {})
    st.session_state.setdefault("loaded_model", None)
    st.session_state.setdefault("leaderboard", pipeline.Leaderboard())


_init_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_upload(uploaded):
    """Read a Streamlit UploadedFile the same way ``data_io`` reads a path."""
    name = uploaded.name.lower()
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded)
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    raise ValueError(f"Unsupported file type: {uploaded.name!r}. Supported: .csv, .xlsx")


def _set_df(df, name):
    st.session_state.df = df
    st.session_state.df_name = name
    st.session_state.last_result = None


CHOICES = {
    ("SVM/SVC", "kernel"): ["linear", "poly", "rbf", "sigmoid"],
    ("Neural Network (MLP)", "activation"): ["relu", "tanh", "logistic"],
    ("Neural Network (MLP)", "solver"): ["adam", "sgd", "lbfgs"],
    ("Decision Tree", "criterion"): ["gini", "entropy", "log_loss"],
    ("Decision Tree", "splitter"): ["best", "random"],
    ("Logistic Regression", "solver"):
        ["lbfgs", "newton-cg", "liblinear", "sag", "saga"],
}


def _param_widget(clf_name, param, default):
    """One hyperparameter input, typed from its default value."""
    key = f"param::{clf_name}::{param}"
    label = param.replace("_", " ")

    if param == "hidden_layer_sizes":
        return st.text_input(label, value=str(default), key=key,
                             help="e.g. 100  or  100,50  or  (100,)")
    if (clf_name, param) in CHOICES:
        opts = CHOICES[(clf_name, param)]
        return st.selectbox(label, opts, index=opts.index(default) if default in opts else 0,
                            key=key)
    if isinstance(default, bool):
        return st.checkbox(label, value=default, key=key)
    if isinstance(default, int):
        return int(st.number_input(label, min_value=1, value=int(default), step=1, key=key))
    if isinstance(default, float):
        if abs(default) < 1e-3:  # var_smoothing and friends - free-text scientific
            raw = st.text_input(label, value=repr(default), key=key)
            try:
                return float(raw)
            except ValueError:
                st.warning(f"{label}: {raw!r} is not a number - using {default}")
                return default
        return float(st.number_input(label, min_value=0.0, value=float(default),
                                     step=0.01, format="%.3f", key=key))
    return st.text_input(label, value=str(default), key=key)


def _collect_params(clf_name):
    """Render the hyperparameter block for ``clf_name`` and return a plain dict."""
    defaults = pipeline.DEFAULT_PARAMS.get(clf_name, {})
    params = {}
    skip = {"penalty", "l1_ratio"} if clf_name == "Logistic Regression" else set()
    for param, default in defaults.items():
        if param in skip:
            continue
        params[param] = _param_widget(clf_name, param, default)

    if clf_name == "Logistic Regression":
        solver = params.get("solver", "lbfgs")
        pen_opts = pipeline.VALID_PENALTIES.get(solver, ["l2", "none"])
        params["penalty"] = st.selectbox("penalty", pen_opts, key="param::LR::penalty")
        if params["penalty"] == "elasticnet":
            params["l1_ratio"] = st.slider("l1 ratio", 0.0, 1.0, 0.5, key="param::LR::l1")
    return params


def _labelled_cm(result):
    cm = result.confusion_matrix
    names = result.class_names if len(result.class_names) == cm.shape[0] else \
        [str(i) for i in range(cm.shape[0])]
    return pd.DataFrame(cm, index=[f"actual · {n}" for n in names],
                        columns=[f"predicted · {n}" for n in names])


def _model_for_scoring():
    """The loaded model, or one built from the last training run (like the old
    Tk app's Predict-on-File)."""
    if st.session_state.loaded_model is not None:
        return st.session_state.loaded_model
    r = st.session_state.last_result
    if r is None:
        return None
    return pipeline.SavedModel(
        pipeline=r.pipeline, label_encoder=r.label_encoder, class_names=r.class_names,
        input_columns=r.input_columns, numeric_cols=r.numeric_cols,
        categorical_cols=r.categorical_cols, classifier_name=r.classifier_name)


# ---------------------------------------------------------------------------
# Sidebar - data source + training spec
# ---------------------------------------------------------------------------

st.sidebar.title("🧮 Generic Super Classifier")
st.sidebar.caption("Train & evaluate a classifier on any tabular dataset — no code.")

source = st.sidebar.radio("Dataset", ["Sample dataset", "Upload a file"], horizontal=True)

if source == "Upload a file":
    up = st.sidebar.file_uploader("CSV or XLSX", type=["csv", "xlsx"])
    if up is not None and st.session_state.df_name != up.name:
        try:
            _set_df(_read_upload(up), up.name)
        except Exception as e:  # noqa: BLE001 - surfaced to the user
            st.sidebar.error(f"Could not read the file: {e}")
else:
    samples = sorted(p.name for p in DATAFILES.glob("*.csv")) if DATAFILES.is_dir() else []
    if samples:
        pick = st.sidebar.selectbox("Choose a sample", samples)
        if st.session_state.df_name != pick:
            _set_df(data_io.load_dataframe(DATAFILES / pick), pick)
    else:
        st.sidebar.info("No bundled sample datasets found.")

df = st.session_state.df

spec = None
if df is not None:
    st.sidebar.divider()
    st.sidebar.subheader("Training spec")
    target = st.sidebar.selectbox("Target column", list(df.columns),
                                  index=len(df.columns) - 1)
    feat_opts = [c for c in df.columns if c != target]
    features = st.sidebar.multiselect("Feature columns", feat_opts, default=feat_opts)
    balance = st.sidebar.checkbox("Balance classes (for imbalanced targets)")
    test_size = st.sidebar.slider("Hold-out test size", 0.1, 0.5, 0.3, 0.05)
    clf_name = st.sidebar.selectbox("Classifier", list(pipeline.CLASSIFIER_CLASSES))
    with st.sidebar.expander("Hyperparameters", expanded=False):
        params = _collect_params(clf_name)
    cv_k = st.sidebar.slider("Cross-validation folds (k)", 2, 10, 5)

    feature_columns = None if set(features) == set(feat_opts) or not features else features
    spec = dict(target=target, features=feature_columns, balance=balance,
                test_size=test_size, clf_name=clf_name, params=params, cv_k=cv_k)

    y_raw = df.dropna(subset=[target])[target]
    if y_raw.nunique() < 2:
        st.sidebar.error(f"'{target}' has only one class — pick another target.")
        spec = None
    elif pipeline.looks_continuous(y_raw):
        st.sidebar.warning(
            f"'{target}' looks continuous ({y_raw.nunique()} distinct numeric values). "
            "It will be treated as that many classes.")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Generic Super Classifier")

if df is None:
    st.info("⬅️  Pick a sample dataset or upload a `.csv` / `.xlsx` file to begin.")
    st.stop()

tab_data, tab_train, tab_plots, tab_board, tab_predict = st.tabs(
    ["📄 Data", "🎯 Train & Evaluate", "📈 Plots", "🏆 Leaderboard", "🔮 Predict & Model"])


with tab_data:
    st.subheader(st.session_state.df_name)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    if spec:
        vc = df.dropna(subset=[spec["target"]])[spec["target"]].value_counts()
        c3.metric("Classes", f"{vc.size}")
        c4.metric("Imbalance ratio", f"{vc.max() / max(vc.min(), 1):.1f}×")
    st.dataframe(df.head(200), width="stretch")
    with st.expander("Full text profile (EDA)"):
        st.code(profiling.profile_report(df))


with tab_train:
    if spec is None:
        st.warning("Fix the training spec in the sidebar first.")
    else:
        col_a, col_b = st.columns(2)
        run_train = col_a.button("Train & evaluate", type="primary",
                                 width="stretch")
        run_cv = col_b.button(f"Cross-validate ({spec['cv_k']}-fold)",
                              width="stretch")

        if run_train:
            try:
                with st.spinner("Training…"):
                    result = pipeline.train_and_evaluate(
                        df, spec["target"], spec["clf_name"], spec["params"],
                        feature_columns=spec["features"],
                        balance_classes=spec["balance"], test_size=spec["test_size"])
            except ValueError as e:
                st.error(f"Cannot train: {e}")
            except (TypeError, SyntaxError) as e:
                st.error(f"Invalid hyperparameter for {spec['clf_name']}: {e}")
            except Exception as e:  # noqa: BLE001
                st.error(f"Training failed — {type(e).__name__}: {e}")
            else:
                st.session_state.last_result = result
                st.session_state.loaded_model = None
                row = st.session_state.leaderboard.add(result)
                st.session_state.last_metrics = dict(row.metrics)
                st.toast("Model trained and added to the leaderboard.")

        result = st.session_state.last_result
        if result is not None:
            kind = "binary" if result.is_binary else f"{len(result.class_names)}-class"
            st.caption(f"Last run · **{result.classifier_name}** · {kind} · "
                       f"features: {', '.join(result.input_columns)}")
            m = st.session_state.last_metrics
            k1, k2, k3, k4 = st.columns(4)
            f1, auc_ = m.get("f1_macro"), m.get("roc_auc")
            k1.metric("Accuracy", f"{result.accuracy:.3f}")
            k2.metric("Macro F1", "—" if f1 is None else f"{f1:.3f}")
            k3.metric("ROC-AUC", "—" if auc_ is None else f"{auc_:.3f}")
            k4.metric("Balancing", result.balance_applied)

            st.markdown("**Confusion matrix**")
            st.dataframe(_labelled_cm(result), width="stretch")
            st.markdown("**Classification report**")
            st.code(result.classification_report)
            if result.dropped_cols:
                st.caption(f"Dropped constant/empty columns: {result.dropped_cols}")

        if run_cv:
            try:
                with st.spinner("Cross-validating…"):
                    cv = pipeline.cross_validate_model(
                        df, spec["target"], spec["clf_name"], spec["params"],
                        feature_columns=spec["features"],
                        balance_classes=spec["balance"], k=spec["cv_k"])
            except ValueError as e:
                st.error(f"Cannot cross-validate: {e}")
            except Exception as e:  # noqa: BLE001
                st.error(f"Cross-validation failed — {type(e).__name__}: {e}")
            else:
                st.session_state.leaderboard.add(cv)
                st.markdown("**Cross-validation**")
                st.code(cv.summary())
                st.dataframe(pd.DataFrame(cv.fold_metrics), width="stretch")


with tab_plots:
    result = st.session_state.last_result
    if result is None:
        st.info("Train a model on the **Train & Evaluate** tab to unlock the plots.")
    else:
        label = st.selectbox("Diagnostic plot", list(plots.PLOTS))
        try:
            fig, ax = plt.subplots(figsize=(7, 5))
            plots.draw(label, ax, result)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not draw '{label}' — {type(e).__name__}: {e}")
        st.caption(plots.PLOTS[label]["desc"])


with tab_board:
    board = st.session_state.leaderboard
    if len(board) == 0:
        st.info("No runs yet — train or cross-validate a model.")
    else:
        sort_by = st.selectbox("Sort by", list(pipeline.Leaderboard.METRICS), index=1)
        rows = [
            {"#": r.index, "Model": r.label, "Eval": r.source,
             "Balanced": "yes" if r.balanced else "no",
             **{k: r.metrics.get(k) for k in pipeline.Leaderboard.METRICS}}
            for r in board.rows
        ]
        table = pd.DataFrame(rows)
        if sort_by in table:
            table = table.sort_values(sort_by, ascending=False, na_position="last")
        st.dataframe(table, width="stretch", hide_index=True)
        if st.button("Clear leaderboard"):
            st.session_state.leaderboard = pipeline.Leaderboard()
            st.rerun()


with tab_predict:
    st.subheader("Save the trained model")
    result = st.session_state.last_result
    if result is None:
        st.caption("Train a model to enable download.")
    else:
        buf = io.BytesIO()
        pipeline.save_model(result, buf)
        st.download_button("⬇️  Download model (.joblib)", buf.getvalue(),
                           file_name=f"{result.classifier_name.replace('/', '-')}.joblib")

    st.divider()
    st.subheader("Load a model")
    mup = st.file_uploader("A .joblib model file", type=["joblib"])
    if mup is not None:
        try:
            st.session_state.loaded_model = pipeline.load_model(mup)
            st.session_state.last_result = None
            st.success(st.session_state.loaded_model.describe())
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not load the model — {type(e).__name__}: {e}")

    st.divider()
    st.subheader("Score a new file")
    model = _model_for_scoring()
    if model is None:
        st.caption("Train or load a model first.")
    else:
        sup = st.file_uploader("Data to score (.csv / .xlsx)", type=["csv", "xlsx"],
                               key="score_upload")
        if sup is not None:
            try:
                score_df = _read_upload(sup)
                preds = pipeline.predict_dataframe(model, score_df)
            except Exception as e:  # noqa: BLE001
                st.error(f"Prediction failed — {type(e).__name__}: {e}")
            else:
                out = score_df.join(preds)
                st.dataframe(out.head(50), width="stretch")
                st.download_button("⬇️  Download predictions (.csv)",
                                   out.to_csv(index=False).encode(),
                                   file_name="predictions.csv")
