"""
app.py
------
Streamlit dashboard for AI Dataset Documentation & Risk Analysis.

Run with:
    streamlit run app.py
"""

import os
import json
from datetime import datetime

import streamlit as st
import pandas as pd

from analyzer import load_dataset, profile_dataset, build_documentation_prompt
from risk_engine import analyze_risks, summarize_risks
from llm_engine import is_ollama_available, list_models, ask_llm


# ---------------- Page config ----------------
st.set_page_config(
    page_title="AI Dataset Documentation & Risk Analysis",
    page_icon="📊",
    layout="wide",
)

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


# ---------------- Sidebar: upload + metadata ----------------
st.sidebar.header("1. Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV or Excel file", type=["csv", "xls", "xlsx"]
)

st.sidebar.header("2. Dataset Metadata")
meta_name = st.sidebar.text_input("Dataset Name")
meta_source = st.sidebar.text_input("Source")
meta_purpose = st.sidebar.text_area("Purpose", height=80)
meta_owner = st.sidebar.text_input("Owner")
meta_license = st.sidebar.text_input("License")
meta_target = st.sidebar.text_input("Target Column (optional)")

metadata = {
    "dataset_name": meta_name,
    "source": meta_source,
    "purpose": meta_purpose,
    "owner": meta_owner,
    "license": meta_license,
    "target_column": meta_target,
}


# ---------------- Main area ----------------
st.title("📊 AI Dataset Documentation & Risk Analysis")
st.caption("Upload a dataset, review risks, and generate documentation with a local LLM.")

if uploaded_file is None:
    st.info("👈 Upload a dataset from the sidebar to begin.")
    st.stop()


# ---- Load ----
try:
    # Save uploaded file to a temp path so our loader can read it
    tmp_path = os.path.join(REPORTS_DIR, uploaded_file.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    df = load_dataset(tmp_path)
    profile = profile_dataset(df)
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()


# ---- Overview ----
st.subheader("Dataset Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", profile["rows"])
c2.metric("Columns", profile["columns"])
c3.metric("Numeric Cols", sum(1 for d in profile["dtypes"].values()
                              if "int" in d or "float" in d))
c4.metric("Missing Cells", sum(profile["missing"].values()))

with st.expander("Preview (first 5 rows)"):
    st.dataframe(df.head())

with st.expander("Column Profile"):
    col_df = pd.DataFrame({
        "dtype": pd.Series(profile["dtypes"]),
        "missing": pd.Series(profile["missing"]),
        "missing_%": pd.Series(profile["missing_pct"]),
        "unique": pd.Series(profile["unique"]),
    })
    st.dataframe(col_df)


# ---- Risk Analysis ----
st.subheader("⚠️ Risk Analysis")
findings = analyze_risks(df, metadata)
summary = summarize_risks(findings)

r1, r2, r3 = st.columns(3)
r1.metric("🔴 High", summary["High"])
r2.metric("🟠 Medium", summary["Medium"])
r3.metric("🟢 Low", summary["Low"])

if findings:
    for f in findings:
        sev = f["severity"]
        icon = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}.get(sev, "⚪")
        with st.expander(f"{icon} [{sev}] {f['type']}"):
            st.write(f["message"])
            if f["columns"]:
                st.write("Affected columns:", ", ".join(map(str, f["columns"])))
else:
    st.success("No risks detected by the rule engine.")


# ---- LLM Documentation ----
st.subheader("📝 Generated Documentation")

ollama_ok = is_ollama_available()

if not ollama_ok:
    st.warning("Ollama is not running. Start it with `ollama serve` to enable LLM features.")
else:
    models = list_models()
    if not models:
        st.warning("No Ollama models found. Pull one with: `ollama pull llama3`")
    else:
        chosen_model = st.selectbox("Choose model", models, index=0)

        if st.button("Generate Documentation"):
            with st.spinner(f"Asking {chosen_model}..."):
                prompt = build_documentation_prompt(df, metadata, profile)
                doc_text = ask_llm(prompt, model=chosen_model)

            st.markdown(doc_text)

            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report = {
                "generated_at": timestamp,
                "model": chosen_model,
                "metadata": metadata,
                "profile": {
                    "rows": profile["rows"],
                    "columns": profile["columns"],
                    "dtypes": profile["dtypes"],
                },
                "risks": findings,
                "documentation": doc_text,
            }

            json_path = os.path.join(REPORTS_DIR, f"report_{timestamp}.json")
            md_path = os.path.join(REPORTS_DIR, f"report_{timestamp}.md")

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# Dataset Report — {metadata.get('dataset_name') or 'Unnamed'}\n\n")
                f.write(f"*Generated: {timestamp}*  \n")
                f.write(f"*Model: {chosen_model}*\n\n")
                f.write("## Risks\n\n")
                for r in findings:
                    f.write(f"- **[{r['severity']}] {r['type']}** — {r['message']}\n")
                f.write("\n## Documentation\n\n")
                f.write(doc_text)

            st.success(f"Saved: `{json_path}` and `{md_path}`")
            with open(md_path, "rb") as f:
                st.download_button("⬇️ Download Markdown report", f,
                                   file_name=os.path.basename(md_path))


# ---- Footer ----
st.caption("Runs fully offline — your data never leaves your machine.")