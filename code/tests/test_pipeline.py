import numpy as np
import pandas as pd
import pytest

import pipeline


# --- helpers ---------------------------------------------------------------

def test_looks_continuous_true_for_wide_numeric_range():
    s = pd.Series(np.arange(500) + 0.5)
    assert pipeline.looks_continuous(s)


def test_looks_continuous_false_for_class_labels():
    assert not pipeline.looks_continuous(pd.Series([0, 1, 2] * 50))
    assert not pipeline.looks_continuous(pd.Series(["a", "b", "c"] * 50))
    assert not pipeline.looks_continuous(pd.Series([True, False] * 50))


def test_parse_hidden_layer_sizes_forms():
    assert pipeline.parse_hidden_layer_sizes("100") == (100,)
    assert pipeline.parse_hidden_layer_sizes("100,50") == (100, 50)
    assert pipeline.parse_hidden_layer_sizes("(100,)") == (100,)
    assert pipeline.parse_hidden_layer_sizes([64, 32]) == (64, 32)
    for bad in ("100,-5", "abc", "()", "1.5"):
        with pytest.raises((ValueError, SyntaxError)):
            pipeline.parse_hidden_layer_sizes(bad)


def test_split_feature_types_drops_constant_and_casts_bool():
    df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0],
        "cat": ["x", "y", "x", "y"],
        "flag": [True, False, True, False],
        "const": [7, 7, 7, 7],
    })
    X, num, cat, dropped = pipeline.split_feature_types(df)
    assert "const" in dropped
    assert set(num) == {"num", "flag"}
    assert cat == ["cat"]
    assert X["flag"].dtype != bool


def test_expand_datetime_columns_creates_parts():
    df = pd.DataFrame({"when": pd.date_range("2020-01-01", periods=10, freq="D"),
                       "v": range(10)})
    out = pipeline.expand_datetime_columns(df)
    assert "when" not in out.columns
    assert {"when_year", "when_month", "when_day", "when_dayofweek"} <= set(out.columns)


# --- build_classifier ----------------------------------------------------

def test_build_classifier_logreg_penalty_none_and_l1_ratio_dropped():
    clf = pipeline.build_classifier(
        "Logistic Regression",
        {"solver": "lbfgs", "C": 1.0, "max_iter": 100, "penalty": "none", "l1_ratio": 0.5})
    assert clf.penalty is None


def test_build_classifier_svc_forces_probability():
    clf = pipeline.build_classifier("SVM/SVC", {"C": 1.0, "kernel": "rbf", "probability": False})
    assert clf.probability is True


def test_build_classifier_unknown_name():
    with pytest.raises(KeyError):
        pipeline.build_classifier("Nope", {})


# --- train_and_evaluate -------------------------------------------------

def test_train_multiclass_string_target(iris_df):
    r = pipeline.train_and_evaluate(
        iris_df, "species", "Random Forest", pipeline.DEFAULT_PARAMS["Random Forest"])
    assert r.class_names == ["setosa", "versicolor", "virginica"]
    assert not r.is_binary
    assert r.y_proba.shape == (len(r.y_test), 3)
    assert np.allclose(r.y_proba.sum(axis=1), 1.0, atol=1e-6)
    assert r.accuracy > 0.8


def test_train_binary_int_target(diabetes_df):
    r = pipeline.train_and_evaluate(
        diabetes_df, "Outcome", "XGBoost", pipeline.DEFAULT_PARAMS["XGBoost"])
    assert r.is_binary
    assert r.class_names == ["0", "1"]
    assert r.y_proba.shape == (len(r.y_test), 2)
    assert r.accuracy > 0.6
    assert r.confusion_matrix.shape == (2, 2)


@pytest.mark.parametrize("name", list(pipeline.CLASSIFIER_CLASSES))
def test_every_classifier_trains_on_iris(iris_df, name):
    r = pipeline.train_and_evaluate(iris_df, "species", name, pipeline.DEFAULT_PARAMS[name])
    assert r.y_proba.shape == (len(r.y_test), 3)
    assert 0.0 <= r.accuracy <= 1.0


def test_missing_target_raises(iris_df):
    with pytest.raises(ValueError):
        pipeline.train_and_evaluate(iris_df, "not_a_column", "KNN",
                                    pipeline.DEFAULT_PARAMS["KNN"])


def test_single_class_target_raises(iris_df):
    df = iris_df.copy()
    df["species"] = "only_one"
    with pytest.raises(ValueError):
        pipeline.train_and_evaluate(df, "species", "KNN", pipeline.DEFAULT_PARAMS["KNN"])


def test_no_usable_features_raises():
    df = pd.DataFrame({"target": [0, 1, 0, 1, 0, 1], "constant": [1, 1, 1, 1, 1, 1]})
    with pytest.raises(ValueError):
        pipeline.train_and_evaluate(df, "target", "KNN", pipeline.DEFAULT_PARAMS["KNN"])


# --- feature_columns selection ----------------------------------------

def test_feature_columns_restricts_training(iris_df):
    cols = ["petal_length", "petal_width"]
    r = pipeline.train_and_evaluate(
        iris_df, "species", "Random Forest",
        pipeline.DEFAULT_PARAMS["Random Forest"], feature_columns=cols)
    assert r.input_columns == cols
    assert set(r.numeric_cols) == set(cols)
    assert r.accuracy > 0.8


def test_feature_columns_none_uses_all(iris_df):
    r = pipeline.train_and_evaluate(
        iris_df, "species", "KNN", pipeline.DEFAULT_PARAMS["KNN"])
    assert "species" not in r.input_columns
    assert len(r.input_columns) == iris_df.shape[1] - 1


def test_feature_columns_unknown_raises(iris_df):
    with pytest.raises(ValueError):
        pipeline.train_and_evaluate(
            iris_df, "species", "KNN", pipeline.DEFAULT_PARAMS["KNN"],
            feature_columns=["petal_width", "nope"])


def test_feature_columns_empty_raises(iris_df):
    with pytest.raises(ValueError):
        pipeline.train_and_evaluate(
            iris_df, "species", "KNN", pipeline.DEFAULT_PARAMS["KNN"],
            feature_columns=[])


def test_feature_columns_cannot_include_target(iris_df):
    with pytest.raises(ValueError):
        pipeline.train_and_evaluate(
            iris_df, "species", "KNN", pipeline.DEFAULT_PARAMS["KNN"],
            feature_columns=["species", "petal_width"])


# --- balance_classes --------------------------------------------------

def test_balance_default_is_off(iris_result):
    assert iris_result.balanced is False
    assert iris_result.balance_applied == "not requested"


def test_balance_uses_class_weight_when_supported(iris_df):
    r = pipeline.train_and_evaluate(
        iris_df, "species", "Logistic Regression",
        pipeline.DEFAULT_PARAMS["Logistic Regression"], balance_classes=True)
    assert r.balanced is True
    assert r.balance_applied == "class_weight='balanced'"
    assert r.pipeline.named_steps["classifier"].get_params()["class_weight"] == "balanced"


def test_balance_falls_back_to_sample_weight_for_xgboost(diabetes_df):
    r = pipeline.train_and_evaluate(
        diabetes_df, "Outcome", "XGBoost",
        pipeline.DEFAULT_PARAMS["XGBoost"], balance_classes=True)
    assert r.balance_applied == "balanced sample_weight"
    assert r.accuracy > 0.6


def test_balance_not_supported_for_knn(iris_df):
    r = pipeline.train_and_evaluate(
        iris_df, "species", "KNN", pipeline.DEFAULT_PARAMS["KNN"],
        balance_classes=True)
    assert r.balance_applied == "not supported by KNN"


def test_balance_improves_minority_recall_on_imbalanced_data():
    rng = np.random.default_rng(0)
    n_major, n_minor = 470, 30
    X_major = rng.normal(0, 1, size=(n_major, 4))
    X_minor = rng.normal(1.1, 1, size=(n_minor, 4))
    df = pd.DataFrame(np.vstack([X_major, X_minor]), columns=list("abcd"))
    df["y"] = [0] * n_major + [1] * n_minor
    common = dict(df=df, target="y", classifier_name="Logistic Regression",
                  params=pipeline.DEFAULT_PARAMS["Logistic Regression"])
    base = pipeline.train_and_evaluate(**common)
    bal = pipeline.train_and_evaluate(**common, balance_classes=True)
    minor = 1
    base_recall = (base.y_pred[base.y_test == minor] == minor).mean()
    bal_recall = (bal.y_pred[bal.y_test == minor] == minor).mean()
    assert bal_recall >= base_recall


# --- cross_validate_model --------------------------------------------

def test_cv_result_shape_and_metrics(cv_result):
    assert cv_result.k == 5
    for m in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc"):
        assert len(cv_result.fold_metrics[m]) == 5
        assert 0.0 <= cv_result.mean[m] <= 1.0
        assert cv_result.std[m] >= 0.0
    assert cv_result.mean["accuracy"] > 0.8
    assert "Random Forest" in cv_result.summary()


def test_cv_binary_target(diabetes_df):
    r = pipeline.cross_validate_model(
        diabetes_df, "Outcome", "Logistic Regression",
        pipeline.DEFAULT_PARAMS["Logistic Regression"], k=4)
    assert r.k == 4
    assert r.is_binary
    assert r.mean["roc_auc"] > 0.6


def test_cv_k_capped_to_min_class_count(iris_df):
    df = iris_df.copy()
    # leave only 3 rows of one class -> k must drop to 3
    drop = df.index[df["species"] == "setosa"][3:]
    df = df.drop(index=drop)
    r = pipeline.cross_validate_model(
        df, "species", "KNN", pipeline.DEFAULT_PARAMS["KNN"], k=10)
    assert r.k == 3


def test_cv_rejects_k_below_2(iris_df):
    with pytest.raises(ValueError):
        pipeline.cross_validate_model(
            iris_df, "species", "KNN", pipeline.DEFAULT_PARAMS["KNN"], k=1)


def test_cv_respects_feature_columns_and_balance(diabetes_df):
    r = pipeline.cross_validate_model(
        diabetes_df, "Outcome", "Logistic Regression",
        pipeline.DEFAULT_PARAMS["Logistic Regression"],
        feature_columns=["Glucose", "BMI", "Age"], balance_classes=True, k=3)
    assert r.balanced is True
    assert r.balance_applied == "class_weight='balanced'"


def test_transformed_X_test_is_dense_2d(iris_result):
    mat = iris_result.transformed_X_test()
    assert isinstance(mat, np.ndarray)
    assert mat.ndim == 2
    assert mat.shape[0] == len(iris_result.y_test)
