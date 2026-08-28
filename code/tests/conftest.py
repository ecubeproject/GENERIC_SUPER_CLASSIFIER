"""Shared fixtures for the classifier test suite.

Run from the repo root or from ``code/``:  ``python -m pytest``
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")  # no display needed / wanted in tests
import pytest

# Make the sibling modules importable without installing the package.
CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

DATA_DIR = os.path.abspath(os.path.join(CODE_DIR, os.pardir, "datafiles"))

import data_io  # noqa: E402
import pipeline  # noqa: E402


@pytest.fixture(autouse=True)
def _quiet_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(scope="session")
def iris_df():
    return data_io.load_dataframe(os.path.join(DATA_DIR, "IRIS.csv"))


@pytest.fixture(scope="session")
def diabetes_df():
    return data_io.load_dataframe(os.path.join(DATA_DIR, "diabetes.csv"))


@pytest.fixture(scope="session")
def iris_result(iris_df):
    """Multiclass task, string target."""
    return pipeline.train_and_evaluate(
        iris_df, "species", "Random Forest", pipeline.DEFAULT_PARAMS["Random Forest"])


@pytest.fixture(scope="session")
def diabetes_result(diabetes_df):
    """Binary task, integer target."""
    return pipeline.train_and_evaluate(
        diabetes_df, "Outcome", "Logistic Regression",
        pipeline.DEFAULT_PARAMS["Logistic Regression"])


@pytest.fixture(scope="session")
def cv_result(iris_df):
    """5-fold CV on the multiclass task."""
    return pipeline.cross_validate_model(
        iris_df, "species", "Random Forest",
        pipeline.DEFAULT_PARAMS["Random Forest"], k=5)
