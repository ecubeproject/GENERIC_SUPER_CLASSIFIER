"""Loading tabular datasets from disk.

Kept free of any Tkinter / UI code so it can be reused from a notebook, a
Streamlit port, or the test suite.
"""

import pandas as pd

SUPPORTED_EXTS = ('.csv', '.xlsx')


def load_dataframe(path):
    """Read a CSV or XLSX file into a DataFrame.

    Raises ValueError for an unsupported extension so the caller can show a
    friendly message instead of a pandas stack trace.
    """
    lower = str(path).lower()
    if lower.endswith('.xlsx'):
        return pd.read_excel(path)
    if lower.endswith('.csv'):
        return pd.read_csv(path)
    raise ValueError(
        f"Unsupported file type: {path!r}. Supported: {', '.join(SUPPORTED_EXTS)}")
