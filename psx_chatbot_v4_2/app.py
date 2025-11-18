import streamlit as st
import zipfile
import tempfile
import os
from pathlib import Path

import pandas as pd  # (not used directly, but fine to keep)
from utils.loader import load_company_zip
from utils.engine import answer_query


# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="PSX Chatbot V4.2",
    layout="wide"
)

# -------------------------------------------------
# Load dark theme CSS (path relative to this file)
# -------------------------------------------------
BASE_DIR = Path(__file__).parent
theme_path = BASE_DIR / "themes" / "dark.css"

if theme_path.exists():
    with open(theme_path, "r", encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
# If file does not exist, app will still run without crashing.


# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("📊 PSX Chatbot V4.2 – Multi-Company Engine (macOS ZIP Fix + Recursive Support)")

uploaded = st.file_uploader(
    "Upload company ZIP files",
    type=["zip"],
    accept_multiple_files=True
)
query = st.text_input("Ask a question (e.g., 'Net profit of AGTL 2023')")

companies = {}


# -------------------------------------------------
# Handle uploads
# -------------------------------------------------
if uploaded:
    st.write("### Upload Status")
    for up in uploaded:
        try:
            # Reset pointer and read file
            up.seek(0)
            data = up.read()

            # Write to a temporary file
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.write(data)
            tmp.flush()
            tmp.seek(0)

            # Extract ZIP to a temporary directory
            extract_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(tmp.name, "r") as z:
                z.extractall(extract_dir)

            # Build company structure from extracted CSVs
            info = load_company_zip(extract_dir)

            if info:
                companies[info["ticker"]] = info
                st.success(f"Loaded {info['ticker']}: {len(info['files'])} CSV files detected.")
            else:
                st.error(f"No CSV files found in {up.name}")

        except Exception as e:
            st.error(f"Error reading {up.name}: {e}")

    if companies:
        st.write("### Companies Loaded:")
        st.write(", ".join(companies.keys()))


# -------------------------------------------------
# Q&A
# -------------------------------------------------
if query and companies:
    st.write("### Answer")
    st.write(answer_query(query, companies))
