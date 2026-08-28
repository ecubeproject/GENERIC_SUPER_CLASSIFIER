import os

import pytest

import data_io
from conftest import DATA_DIR


def test_load_csv():
    df = data_io.load_dataframe(os.path.join(DATA_DIR, "IRIS.csv"))
    assert df.shape == (150, 5)
    assert "species" in df.columns


def test_load_xlsx():
    df = data_io.load_dataframe(os.path.join(DATA_DIR, "IRIS.xlsx"))
    assert df.shape[0] == 150
    assert "species" in df.columns


def test_csv_and_xlsx_agree():
    a = data_io.load_dataframe(os.path.join(DATA_DIR, "IRIS.csv"))
    b = data_io.load_dataframe(os.path.join(DATA_DIR, "IRIS.xlsx"))
    assert list(a.columns) == list(b.columns)
    assert len(a) == len(b)


def test_unsupported_extension_raises():
    with pytest.raises(ValueError):
        data_io.load_dataframe("some_file.txt")


def test_extension_check_is_case_insensitive(tmp_path):
    src = os.path.join(DATA_DIR, "diabetes.csv")
    dst = tmp_path / "DIABETES.CSV"
    dst.write_bytes(open(src, "rb").read())
    df = data_io.load_dataframe(str(dst))
    assert len(df) > 0
