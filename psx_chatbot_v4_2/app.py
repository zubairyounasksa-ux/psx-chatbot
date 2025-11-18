import streamlit as st
import zipfile
import tempfile
from pathlib import Path

import pandas as pd  # optional
from utils.loader import load_company_zip
from utils.engine import answer_query


# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Youngs Capital PSX Chatbot V1.00",
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

    # ----------------- Mode 1: Natural-language question -----------------
    st.markdown("#### Ask in your own words")

    nl_query = st.text_input("Ask a question (e.g., 'Net profit of GAL 2023')", key="nl_query")

    if nl_query:
        st.write("##### Answer")
        st.write(answer_query(nl_query, companies))

    st.markdown("---")

    # ----------------- Mode 2: Dropdowns (Company / Metric / Year) -----------------
    st.markdown("#### Or choose from dropdowns")

    # Layout for dropdowns
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_company = st.selectbox(
            "Company",
            sorted(companies.keys()),
            key="dd_company"
        )

    comp = companies[selected_company]
    metric_options = sorted(comp.get("metrics", []))
    period_options = sorted(comp.get("periods", []))

    with col2:
        selected_metric = st.selectbox(
            "Metric (Excel 'Heading' column)",
            metric_options,
            key="dd_metric"
        )

    with col3:
        # Allow user to select specific year/period
        selected_period = st.selectbox(
            "Year / Period",
            period_options,
            key="dd_period"
        )

    if st.button("Get value", key="dd_button"):
        # Build a precise query string so we reuse the same engine
        structured_query = f"{selected_metric} of {selected_company} in {selected_period}"
        st.write("##### Answer")
        st.write(answer_query(structured_query, companies))
