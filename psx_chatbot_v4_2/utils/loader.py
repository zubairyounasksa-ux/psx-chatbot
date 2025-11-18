# utils/loader.py

import os
import re
import pandas as pd


def find_csvs_recursive(folder):
    csvs = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".csv") and "__macosx" not in root.lower():
                csvs.append(os.path.join(root, f))
    return csvs


def detect_ticker(files):
    """
    Detect ticker from file names.

    Works with names like:
      GAL - Annual Accounts - Balance Sheet.csv
      HCAR - Income Statement.csv
      AGTL_PL.csv

    It simply takes the first 2–5 letters from the beginning of the file name.
    """
    for fp in files:
        name = os.path.basename(fp)
        # Match first 2–5 alphabetic characters at the start
        m = re.match(r"([A-Za-z]{2,5})", name)
        if m:
            return m.group(1).upper()
    return "UNKNOWN"


def load_company_zip(folder):
    files = find_csvs_recursive(folder)
    if not files:
        return None

    ticker = detect_ticker(files)
    datasets = []

    for fp in files:
        try:
            df = pd.read_csv(fp)
            datasets.append(df)
        except Exception:
            # Ignore unreadable CSVs
            pass

    if not datasets:
        return None

    metrics = set()
    periods = set()

    for df in datasets:
        df_cols = list(df.columns)
        if len(df_cols) > 1:
            # First column = Heading / Metric
            metrics.update(df.iloc[:, 0].astype(str))

            # Remaining columns = periods (years etc.), but skip 'Trend'
            for col in df_cols[1:]:
                col_str = str(col).strip()
                if col_str.lower() == "trend":
                    continue
                periods.add(col_str)

    return {
        "ticker": ticker,
        "files": datasets,
        "metrics": list(metrics),
        "periods": list(periods),
    }
