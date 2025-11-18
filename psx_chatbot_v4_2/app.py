
import streamlit as st
import zipfile, tempfile, os, pandas as pd
from utils.loader import load_company_zip
from utils.engine import answer_query

st.set_page_config(page_title="PSX Chatbot V4.2", layout="wide")
with open("themes/dark.css") as f: 
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📊 PSX Chatbot V4.2 – Multi-Company Engine (macOS ZIP Fix + Recursive Support)")

uploaded = st.file_uploader("Upload company ZIP files", type=["zip"], accept_multiple_files=True)
query = st.text_input("Ask a question (e.g., 'Net profit of AGTL 2023')")

companies = {}

if uploaded:
    st.write("### Upload Status")
    for up in uploaded:
        try:
            up.seek(0)
            data = up.read()

            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.write(data)
            tmp.flush()
            tmp.seek(0)

            extract_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(tmp.name, 'r') as z:
                z.extractall(extract_dir)

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

if query and companies:
    st.write("### Answer")
    st.write(answer_query(query, companies))
