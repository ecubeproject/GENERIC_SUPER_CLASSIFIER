"""Smoke tests for the Streamlit front-end.

These drive the real app through ``streamlit.testing.v1.AppTest`` - no browser,
no server - exercising the load -> train -> cross-validate -> plot -> leaderboard
path on the bundled sample datasets. The modelling itself is covered in depth by
the other test modules; here we only guard the UI wiring.
"""

import io
import os

import pandas as pd
import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

import plots  # noqa: E402  (conftest put code/ on sys.path)
import pipeline  # noqa: E402
import streamlit_app  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(__file__)), "streamlit_app.py")


def _sidebar_selectbox(at, label):
    return next(sb for sb in at.sidebar.selectbox if sb.label == label)


def _trained_app(target, classifier="Random Forest", sample="IRIS.csv"):
    """An AppTest that has loaded ``sample`` and trained one model."""
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception

    _sidebar_selectbox(at, "Choose a sample").select(sample)
    at.run()
    _sidebar_selectbox(at, "Target column").select(target)
    _sidebar_selectbox(at, "Classifier").select(classifier)
    at.run()

    next(b for b in at.button if b.label.startswith("Train")).click()
    at.run()
    assert not at.exception, at.exception
    return at


def test_app_loads_clean():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception


def test_train_multiclass_iris():
    at = _trained_app("species")
    labels = {m.label for m in at.metric}
    assert {"Accuracy", "Macro F1", "ROC-AUC"} <= labels
    assert at.session_state.last_result is not None
    assert len(at.session_state.leaderboard) == 1


def test_train_binary_diabetes():
    at = _trained_app("Outcome", classifier="Logistic Regression", sample="diabetes.csv")
    assert at.session_state.last_result.is_binary


def test_cross_validate_button():
    at = _trained_app("species")
    next(b for b in at.button if b.label.startswith("Cross-validate")).click()
    at.run()
    assert not at.exception
    assert len(at.session_state.leaderboard) == 2  # train + CV


def test_every_plot_renders():
    at = _trained_app("species")
    for label in list(plots.PLOTS):
        sb = next(s for s in at.selectbox if set(s.options) == set(plots.PLOTS))
        sb.select(label)
        at.run()
        assert not at.exception, (label, at.exception)


def test_continuous_target_warns():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    _sidebar_selectbox(at, "Choose a sample").select("IRIS.csv")
    at.run()
    _sidebar_selectbox(at, "Target column").select("petal_length")
    at.run()
    assert any("continuous" in w.value for w in at.sidebar.warning)


# --- upload helper (AppTest can't drive file_uploader in this version) -------

def test_read_upload_csv_and_xlsx():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    csv_buf = io.BytesIO(df.to_csv(index=False).encode())
    csv_buf.name = "sample.csv"
    pd.testing.assert_frame_equal(streamlit_app._read_upload(csv_buf), df)

    xlsx_buf = io.BytesIO()
    df.to_excel(xlsx_buf, index=False)
    xlsx_buf.seek(0)
    xlsx_buf.name = "sample.xlsx"
    pd.testing.assert_frame_equal(streamlit_app._read_upload(xlsx_buf), df)


def test_read_upload_rejects_unknown_extension():
    buf = io.BytesIO(b"nope")
    buf.name = "data.txt"
    with pytest.raises(ValueError):
        streamlit_app._read_upload(buf)


def test_param_collection_matches_defaults(monkeypatch):
    """_collect_params must return a dict that build_classifier accepts for every
    registered classifier (guards the widget-typing logic)."""
    import streamlit as st

    for name in pipeline.CLASSIFIER_CLASSES:
        # emulate the widgets returning their default values
        monkeypatch.setattr(st, "text_input", lambda label, **k: k.get("value", ""))
        monkeypatch.setattr(st, "selectbox", lambda label, opts, **k: opts[k.get("index", 0)])
        monkeypatch.setattr(st, "checkbox", lambda label, **k: k.get("value", False))
        monkeypatch.setattr(st, "number_input", lambda label, **k: k["value"])
        monkeypatch.setattr(st, "slider", lambda *a, **k: a[3] if len(a) > 3 else 0.5)
        params = streamlit_app._collect_params(name)
        pipeline.build_classifier(name, params)  # must not raise
