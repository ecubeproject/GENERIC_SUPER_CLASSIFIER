"""Entry point for the Classifier GUI.

The implementation now lives in focused modules:
  data_io.py    - load CSV / XLSX
  profiling.py  - text summary of a DataFrame
  pipeline.py   - spec -> fitted sklearn pipeline + evaluation (no Tk)
  plots.py      - diagnostic plots as pure (ax, ...) functions (no Tk)
  app_tk.py     - the Tkinter UI

Run this file (or `python -m app_tk`) to launch the app.
"""

from app_tk import main

if __name__ == "__main__":
    main()
