"""Legacy entry point.

The app is now a Streamlit web UI. The modelling / plotting engine lives in
focused, UI-free modules:

  data_io.py    - load CSV / XLSX
  profiling.py  - text summary of a DataFrame
  pipeline.py   - spec -> fitted sklearn pipeline + evaluation
  plots.py      - diagnostic plots as pure (ax, ...) functions
  streamlit_app.py - the Streamlit UI

Running this file just launches Streamlit for you (equivalent to
``streamlit run code/streamlit_app.py``).
"""

import os
import sys

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "streamlit_app.py")


def main():
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        sys.exit("Streamlit is not installed. Run:  pip install -r requirements.txt\n"
                 f"Then:  streamlit run {APP}")
    sys.argv = ["streamlit", "run", APP]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
