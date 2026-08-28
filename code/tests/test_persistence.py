import numpy as np
import pandas as pd
import pytest

import pipeline


def _round_trip(result, tmp_path):
    path = tmp_path / "model.joblib"
    pipeline.save_model(result, path)
    return pipeline.load_model(path)


def test_save_load_predict_multiclass(iris_df, iris_result, tmp_path):
    bundle = _round_trip(iris_result, tmp_path)
    assert bundle.classifier_name == "Random Forest"
    assert bundle.class_names == ["setosa", "versicolor", "virginica"]

    preds = pipeline.predict_dataframe(bundle, iris_df.head(10))
    assert list(preds.index) == list(iris_df.head(10).index)
    assert set(preds["prediction"]).issubset(set(iris_df["species"]))
    proba_cols = [f"proba_{c}" for c in bundle.class_names]
    assert proba_cols == [c for c in preds.columns if c.startswith("proba_")]
    assert np.allclose(preds[proba_cols].sum(axis=1), 1.0, atol=1e-6)


def test_predictions_match_original_pipeline(diabetes_df, tmp_path):
    result = pipeline.train_and_evaluate(
        diabetes_df, "Outcome", "Logistic Regression",
        pipeline.DEFAULT_PARAMS["Logistic Regression"])
    bundle = _round_trip(result, tmp_path)

    sample = result.X_test.copy()
    reloaded = pipeline.predict_dataframe(bundle, sample)
    assert np.array_equal(
        bundle.label_encoder.transform(reloaded["prediction"]), result.y_pred)


def test_predict_ignores_extra_columns(iris_df, iris_result, tmp_path):
    bundle = _round_trip(iris_result, tmp_path)
    df = iris_df.head(5).copy()
    df["some_note"] = "ignore me"
    preds = pipeline.predict_dataframe(bundle, df)
    assert len(preds) == 5


def test_predict_missing_column_raises(iris_df, iris_result, tmp_path):
    bundle = _round_trip(iris_result, tmp_path)
    df = iris_df.head(5).drop(columns=["petal_width"])
    with pytest.raises(ValueError, match="missing required column"):
        pipeline.predict_dataframe(bundle, df)


def test_feature_columns_model_only_needs_those_columns(diabetes_df, tmp_path):
    result = pipeline.train_and_evaluate(
        diabetes_df, "Outcome", "Random Forest",
        pipeline.DEFAULT_PARAMS["Random Forest"],
        feature_columns=["Glucose", "BMI", "Age"])
    bundle = _round_trip(result, tmp_path)
    assert bundle.input_columns == ["Glucose", "BMI", "Age"]

    preds = pipeline.predict_dataframe(
        bundle, diabetes_df[["Glucose", "BMI", "Age"]].head(8))
    assert len(preds) == 8


def test_load_model_rejects_non_model_file(tmp_path):
    import joblib
    path = tmp_path / "not_a_model.joblib"
    joblib.dump({"hello": "world"}, path)
    with pytest.raises(ValueError):
        pipeline.load_model(path)
