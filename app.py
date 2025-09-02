import streamlit as st
import pandas as pd
import boto3
import json
import os
import re
import io
from dotenv import load_dotenv
import base64
import requests
 
# =========================
# CONFIG & SETUP
# =========================
load_dotenv()
region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
client = boto3.client("bedrock-runtime", region_name=region)
 
st.set_page_config(page_title="Generative Data Modeling", layout="wide")
st.title("Generative Data Modeling")
 
# Sampling limits (increase if needed)
PROFILE_SAMPLE_ROWS = 5000
SCHEMA_SAMPLE_ROWS = 100
 
# =========================
# UTILITIES
# =========================
def _read_df(file_bytes: bytes, filename: str, nrows: int | None):
    """Read CSV/Excel from bytes with optional row limit."""
    bio = io.BytesIO(file_bytes)
    if filename.lower().endswith(".csv"):
        return pd.read_csv(bio, nrows=nrows)
    return pd.read_excel(bio, nrows=nrows)
 
def extract_schema(file_bytes: bytes, filename: str):
    """Extract lightweight schema (first N rows)."""
    df = _read_df(file_bytes, filename, nrows=SCHEMA_SAMPLE_ROWS)
    cols = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample_values = df[col].dropna().unique()[:5].tolist()
        cols.append({
            "column": col,
            "dtype": dtype,
            "sample_values": [str(v) for v in sample_values]
        })
    return cols
 
def _type_bucket(dtype: str) -> str:
    dt = dtype.lower()
    if any(x in dt for x in ["int"]): return "integer"
    if any(x in dt for x in ["float", "double", "decimal", "number"]): return "number"
    if "date" in dt: return "date"
    if "time" in dt: return "timestamp"
    if "bool" in dt: return "boolean"
    return "string"
 
def profile_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Profile: null %, distinct, uniqueness, min/max (lightweight)."""
    df = _read_df(file_bytes, filename, nrows=PROFILE_SAMPLE_ROWS)
    total = max(len(df), 1)
 
    rows = []
    for col in df.columns:
        series = df[col]
        non_null = series.notna().sum()
        null_pct = round(100 * (1 - (non_null / total)), 2)
        distinct_cnt = series.nunique(dropna=True)
        is_unique = bool(distinct_cnt == non_null and null_pct == 0.0)
 
        min_val, max_val = None, None
        try:
            if non_null > 0:
                min_val = series.min()
                max_val = series.max()
        except Exception:
            pass
 
        rows.append({
            "column": col,
            "dtype": str(series.dtype),
            "bucket": _type_bucket(str(series.dtype)),
            "non_null": int(non_null),
            "null_pct": float(null_pct),
            "distinct_count": int(distinct_cnt),
            "is_unique": is_unique,
            "min": None if pd.isna(min_val) else (str(min_val) if not isinstance(min_val, (int, float)) else float(min_val)),
            "max": None if pd.isna(max_val) else (str(max_val) if not isinstance(max_val, (int, float)) else float(max_val)),
        })
    return pd.DataFrame(rows)
 
def gather_inputs(files):
    """
    Read all files once and return:
      - schemas_by_table: {table: [col-meta]}
      - profiling_by_table: {table: dataframe}
      - table_rows_estimate: {table: rowcount}
    """
    schemas_by_table = {}
    profiling_by_table = {}
    table_rows_estimate = {}
 
    for f in files:
        file_bytes = f.getvalue()
        table = os.path.splitext(os.path.basename(f.name))[0]
 
        # Schema sample
        schemas_by_table[table] = extract_schema(file_bytes, f.name)
 
        # Profiling sample
        prof_df = profile_data(file_bytes, f.name)
        profiling_by_table[table] = prof_df
 
        # Quick row estimate (full read not required)
        df_est = _read_df(file_bytes, f.name, nrows=None)
        table_rows_estimate[table] = int(len(df_est))
 
    return schemas_by_table, profiling_by_table, table_rows_estimate
 
# =========================
# BEDROCK CALL
# =========================
def call_bedrock(schemas_by_table, profiling_by_table, table_rows_estimate, target_db="Snowflake"):
    """
    Send a strong, anomaly-aware prompt so the model:
      - reconciles naming inconsistencies across files
      - normalizes to canonical business terms
      - produces LOGICAL_MODEL, ERD (Mermaid), DDL, SCD recommendations
    """
    profiling_compact = {t: df.to_dict(orient="records") for t, df in profiling_by_table.items()}
 
    prompt_parts = [
        "You are a senior enterprise data modeler specializing in dimensional (star/snowflake) modeling.",
        "You are given multiple flat input tables from a single business domain.",
        "Your job is to design a precise, analytics-friendly model that reconciles anomalies across files.",
        "",
        "### CRITICAL RULES",
        "- Consider ALL input files together (NOT individually).",
        "- Detect and reconcile column name anomalies across tables:",
        "  * Synonyms and abbreviations (e.g., customer_id, cust_id, customer_number).",
        "  * Case differences, punctuation/underscore differences, singular/plural.",
        "  * Typos with clear intent (minor edit distance).",
        "- Normalize to consistent, descriptive business terms for entities and columns.",
        "- Prefer surrogate integer keys for dimensions when natural keys are unclear.",
        "- Use conservative, high-precision mappings; flag any ambiguous mappings as assumptions.",
        "",
        "### OUTPUT REQUIREMENTS",
        "Respond in EXACTLY these sections:",
        "",
        "### LOGICAL_MODEL",
        "- Identify fact tables and define their GRAIN.",
        # "- Identify normalized dimension tables as per SNOWFLAKE SCHEMA dimension tables and list attributes, Primary Keys, Foreign Keys.",
        "- Design the model as a SNOWFLAKE SCHEMA with normalized dimensions along with list of attributes, Primary Keys, Foreign Keys .",
        "- Define PKs and FKs (facts reference dimensions).",
        "",
        "### ERD",
        "- Provide a valid Mermaid ER diagram using `erDiagram` syntax.",
        "- Use the normalized entity/column names.",
        "- Show correct cardinalities (|| for one, o{ for many).",
        "- Design the model as a SNOWFLAKE SCHEMA with normalized dimensions.",
        "",
        "### DDL",
        "- Provide SQL DDL including CREATE TABLE for all facts and dimensions.",
        "- Include PK and FK constraints explicitly.",
        "- Implement the SNOWFLAKE SCHEMA design with normalized dimension tables.",
        "",
        "### SCD_RECOMMENDATIONS",
        "- For each dimension, recommend SCD Type (1/2/3) with rationale based on profiling.",
        "",
        "### INPUTS (DO NOT ECHO IN OUTPUT)",
        "- TABLES_ROWCOUNT:",
        json.dumps(table_rows_estimate, indent=2),
        "- SCHEMAS_BY_TABLE:",
        json.dumps(schemas_by_table, indent=2),
        "- PROFILING_BY_TABLE:",
        json.dumps(profiling_compact, indent=2),
    ]
 
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 3200,
            "temperature": 0,
            "messages": [{"role": "user", "content": "\n".join(map(str, prompt_parts))}]
        })
    )
    output = json.loads(response["body"].read())
    return output["content"][0]["text"]
 
# =========================
# PARSING & RENDERING
# =========================
def parse_output(result):
    """Extract logical model, ERD, DDL, SCD from model output."""
    def pick(pattern):
        m = re.search(pattern, result)
        return m.group(1).strip() if m else ""
    logical = pick(r"### LOGICAL_MODEL([\s\S]*?)(###|$)")
    erd = pick(r"```mermaid([\s\S]*?)```")
    ddl = pick(r"```sql([\s\S]*?)```")
    scd = pick(r"### SCD_RECOMMENDATIONS([\s\S]*?)(###|$)")
    return logical, erd, ddl, scd
 
def save_erd_as_png(erd_code, filename="erd.png"):
    """Render Mermaid ERD into PNG using mermaid.ink API."""
    try:
        erd_text = f"erDiagram\n{erd_code}" if not erd_code.startswith("erDiagram") else erd_code
        graphbytes = erd_text.encode("utf8")
        base64_string = base64.urlsafe_b64encode(graphbytes).decode("ascii")
        url = f"https://mermaid.ink/img/{base64_string}"
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
            return filename
        else:
            st.error(f"Mermaid rendering failed: HTTP {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error rendering ERD: {e}")
        return None
 
# =========================
# UI FLOW (Professional, multi-file)
# =========================

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

if 'clear_files' not in st.session_state:
    st.session_state.clear_files = False
 
uploader_key = f"file_uploader_{st.session_state.clear_files}"
uploaded_files = st.file_uploader(
    "Upload one or more CSV/Excel files",
    type=["csv", "xlsx"],
    accept_multiple_files=True,
    key=uploader_key
)
 
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

    # Optional soft warning if too many
    if len(uploaded_files) > 10:
        st.warning("⚠️ Uploading more than 10 files may slow down processing.")
 
# Clear all files button
if st.session_state.uploaded_files:
    if st.button("Clear All Files"):
        st.session_state.uploaded_files = []
        st.session_state.clear_files = not st.session_state.clear_files
        st.rerun()
 
target_db = st.selectbox("Target Database", ["Snowflake", "SQL Server", "Oracle", "Redshift", "Synapse"])
 
if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded")
 
    if st.button("Generate Data Model"):
        with st.spinner("Reconciling anomalies and generating model..."):
            schemas_by_table, profiling_by_table, rowcount_by_table = gather_inputs(uploaded_files)
            result = call_bedrock(
                schemas_by_table, profiling_by_table, rowcount_by_table, target_db=target_db
            )
            logical, erd, ddl, scd = parse_output(result)
 
        tabs = st.tabs(["Overview", "Logical Model", "ERD", "SQL DDL", "SCD"])
 
        # Overview
        with tabs[0]:
            st.write("### Detected Tables")
            st.json({"tables": list(schemas_by_table.keys()), "row_counts_estimate": rowcount_by_table})
            st.write("### Column Profiling (sample-based)")
            for t, df in profiling_by_table.items():
                st.markdown(f"**{t}**")
                st.dataframe(df, use_container_width=True)
 
        # Logical
        with tabs[1]:
            st.write("### Logical Data Model")
            st.text_area("", logical, height=280)
 
        # ERD
        with tabs[2]:
            if erd:
                with st.spinner("Rendering ERD diagram..."):
                    erd_file = save_erd_as_png(erd, "erd.png")
                if erd_file:
                    st.image(erd_file, caption="Entity-Relationship Diagram", use_container_width=True)
                    with open(erd_file, "rb") as f:
                        st.download_button("Download ERD (PNG)", f, file_name="erd.png", mime="image/png")
                else:
                    st.warning("Could not render ERD")
 
        # DDL
        with tabs[3]:
            if ddl:
                st.write("### Generated SQL DDL")
                st.code(ddl, language="sql")
                st.download_button("Download SQL DDL", ddl, file_name="model.sql", mime="text/sql")
 
        # SCD
        with tabs[4]:
            if scd:
                st.write("### SCD Recommendations")
                st.text_area("", scd, height=260)
 
        with st.expander("Raw Model Output (debug)"):
            st.text(result)
 
# Sidebar help
st.sidebar.header("How to Use")
st.sidebar.markdown("""
1) Upload multiple CSV/XLSX files from the same domain  
2) Pick your target database  
3) Click **Generate Data Model**  
4) Review logical model, ERD and DDL  
5) Download ERD (PNG) and SQL
""")
st.sidebar.info("The model reconciles naming anomalies across files and outputs normalized entities/columns.")
 
