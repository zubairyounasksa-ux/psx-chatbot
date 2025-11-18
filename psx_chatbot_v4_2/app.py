import streamlit as st
import zipfile
import tempfile
from pathlib import Path

import pandas as pd  # optional, but fine to keep
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

# Folder where backend ZIPs live (inside the repo)
DATA_DIR = BASE_DIR / "data"


# -------------------------------------------------
# Load ALL companies from backend at startup
# -------------------------------------------------
@st.cache_data(show_spinner=True)
def load_all_companies():
    companies = {}

    if not DATA_DIR.exists():
        return companies

    # Load every *.zip file from the data folder
    for zip_path in DATA_DIR.glob("*.zip"):
        try:
            extract_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(extract_dir)

            info = load_company_zip(extract_dir)

            if info and info.get("ticker"):
                companies[info["ticker"]] = info
        except Exception as e:
            # Optional: log to console for debugging
            print(f"Error loading {zip_path.name}: {e}")

    return companies


companies = load_all_companies()


# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("📊 Youngs Capital PSX Chatbot V1.00")

if not companies:
    st.error(
        "No company data found. Please add ZIP files into the "
        "'data' folder inside the repository (e.g., psx_chatbot_v4_2/data/GAL.zip)."
    )
else:
    st.write("### Companies Available:")
    st.write(", ".join(sorted(companies.keys())))

    query = st.text_input("Ask a question (e.g., 'Net profit of GAL 2023')")

    if query:
        st.write("### Answer")
        st.write(answer_query(query, companies))
