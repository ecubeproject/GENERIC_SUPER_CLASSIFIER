"""Quick text profile of a DataFrame (shape, dtypes, describe, missing, unique).

UI-free: returns a plain string the caller drops into a text widget, a
notebook cell, or a log.
"""

from io import StringIO


def profile_report(df):
    """Return a multi-section text summary of ``df``."""
    buf = StringIO()
    df.info(buf=buf)
    data_info = buf.getvalue()

    data_desc = df.describe(include='all').to_string()
    missing_values = df.isna().sum().to_frame('Missing Values').to_string()
    unique_values = df.nunique().to_frame('Unique Values').to_string()

    return (
        f"Data Info:\n{data_info}\n\n"
        f"Data Description:\n{data_desc}\n\n"
        f"Missing Values:\n{missing_values}\n\n"
        f"Unique Values:\n{unique_values}\n\n"
    )
