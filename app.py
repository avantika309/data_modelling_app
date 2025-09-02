# import streamlit as st
# import pandas as pd
# import boto3
# import json
# import os
# import re
# from dotenv import load_dotenv
# import base64
# import requests
# import io
 
# # =========================
# # CONFIG & SETUP
# # =========================
# load_dotenv()
# region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
# model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
# client = boto3.client("bedrock-runtime", region_name=region)
 
# st.set_page_config(page_title="Generative Data Modeling", layout="wide")
# st.title("Generative Data Modeling")
 
# # How many rows to read for profiling (kept modest for speed)
# PROFILE_SAMPLE_ROWS = 5000
# SCHEMA_SAMPLE_ROWS = 100
 
 
# # =========================
# # UTILITIES
# # =========================
# def _read_df(file_bytes: bytes, filename: str, nrows: int | None):
#     """Read CSV/Excel from bytes with optional row limit."""
#     bio = io.BytesIO(file_bytes)
#     if filename.lower().endswith(".csv"):
#         return pd.read_csv(bio, nrows=nrows)
#     return pd.read_excel(bio, nrows=nrows)
 
 
# def extract_schema(file_bytes: bytes, filename: str):
#     """Extract lightweight schema (first N rows)."""
#     df = _read_df(file_bytes, filename, nrows=SCHEMA_SAMPLE_ROWS)
 
#     schema = []
#     for col in df.columns:
#         dtype = str(df[col].dtype)
#         sample_values = df[col].dropna().unique()[:5].tolist()
#         schema.append({
#             "column": col,
#             "dtype": dtype,
#             "sample_values": [str(v) for v in sample_values]
#         })
#     return schema
 
 
# def profile_data(file_bytes: bytes, filename: str) -> tuple[pd.DataFrame, list[dict]]:
#     """Profile columns: null %, distinct count, min/max, uniqueness, sample values."""
#     df = _read_df(file_bytes, filename, nrows=PROFILE_SAMPLE_ROWS)
#     total = len(df) if len(df) > 0 else 1
 
#     rows = []
#     for col in df.columns:
#         series = df[col]
#         non_null = series.notna().sum()
#         null_pct = round(100 * (1 - (non_null / total)), 2)
#         distinct_cnt = series.nunique(dropna=True)
#         is_unique = bool(distinct_cnt == non_null and null_pct == 0.0)
 
#         # min/max only for comparable types
#         min_val = None
#         max_val = None
#         try:
#             if non_null > 0:
#                 min_val = series.min()
#                 max_val = series.max()
#         except Exception:
#             pass
 
#         sample_vals = series.dropna().unique()[:5].tolist()
 
#         row = {
#             "column": col,
#             "dtype": str(series.dtype),
#             "non_null": int(non_null),
#             "null_pct": float(null_pct),
#             "distinct_count": int(distinct_cnt),
#             "is_unique": is_unique,
#             "min": None if pd.isna(min_val) else (str(min_val) if not isinstance(min_val, (int, float)) else float(min_val)),
#             "max": None if pd.isna(max_val) else (str(max_val) if not isinstance(max_val, (int, float)) else float(max_val)),
#             "sample_values": [str(v) for v in sample_vals]
#         }
#         rows.append(row)
 
#     prof_df = pd.DataFrame(rows, columns=[
#         "column", "dtype", "non_null", "null_pct", "distinct_count", "is_unique", "min", "max", "sample_values"
#     ])
#     return prof_df, rows
 
 
# def call_bedrock(schema, profiling_rows, target_db="Snowflake"):
#     """Call Bedrock with structured prompt to generate model, ERD, DDL, and SCD recs."""
#     prompt_parts = [
#         "You are a senior enterprise data modeler specializing in dimensional modeling.",
#         "Analyze the flat schema and profiling stats to design a star/snowflake schema suitable for analytics.",
#         "",
#         "### OUTPUT REQUIREMENTS",
#         "Respond in exactly these sections:",
#         "",
#         "### LOGICAL_MODEL",
#         "- Identify fact tables and define their **grain** (e.g., sales per order, inventory per day).",
#         "- Identify dimension tables with descriptive attributes.",
#         "- Define **PKs** for each table (prefer surrogate keys for dimensions when natural keys are unclear).",
#         "- Define **FKs** and relationships between facts and dimensions.",
#         "- State assumptions clearly where needed.",
#         "",
#         "### ERD",
#         "- Provide a **valid Mermaid ER diagram** using `erDiagram` syntax.",
#         "- Show correct cardinalities (||, o{, etc).",
#         "- Use meaningful entity names (avoid placeholders).",
#         "",
#         "### DDL",
#         f"- Provide a valid SQL DDL script for {target_db}.",
#         "- Include CREATE TABLE statements for facts and dimensions.",
#         "- Include PK/FK constraints explicitly.",
#         "- Use Snowflake-compatible datatypes (VARCHAR, NUMBER, DATE, TIMESTAMP).",
#         "",
#         "### SCD_RECOMMENDATIONS",
#         "- For each dimension, recommend SCD Type (1/2/3) with rationale.",
#         "- Base this on profiling signals (e.g., columns likely to change over time like addresses, names, statuses).",
#         "",
#         "### INPUTS (DO NOT ECHO IN OUTPUT)",
#         f"- SCHEMA_SAMPLE (first {SCHEMA_SAMPLE_ROWS} rows analyzed):",
#         json.dumps(schema, indent=2),
#         "",
#         f"- COLUMN_PROFILING_SAMPLE (first {PROFILE_SAMPLE_ROWS} rows analyzed):",
#         json.dumps(profiling_rows, indent=2),
#     ]
 
#     response = client.invoke_model(
#         modelId=model_id,
#         body=json.dumps({
#             "anthropic_version": "bedrock-2023-05-31",
#             "max_tokens": 3200,
#             "temperature": 0,
#             "messages": [{"role": "user", "content": "\n".join(map(str, prompt_parts))}]
#         })
#     )
#     output = json.loads(response["body"].read())
#     return output["content"][0]["text"]
 
 
# def parse_output(result):
#     """Extract logical model, ERD, DDL, SCD from model output."""
#     def pick(pattern):
#         m = re.search(pattern, result)
#         return m.group(1).strip() if m else ""
 
#     logical = pick(r"### LOGICAL_MODEL([\s\S]*?)(###|$)")
#     erd = pick(r"```mermaid([\s\S]*?)```")
#     ddl = pick(r"```sql([\s\S]*?)```")
#     scd = pick(r"### SCD_RECOMMENDATIONS([\s\S]*?)(###|$)")
#     return logical, erd, ddl, scd
 
 
# def save_erd_as_png(erd_code, filename="erd.png"):
#     """Render Mermaid ERD into PNG using mermaid.ink API."""
#     try:
#         erd_text = f"erDiagram\n{erd_code}" if not erd_code.startswith("erDiagram") else erd_code
#         graphbytes = erd_text.encode("utf8")
#         base64_string = base64.urlsafe_b64encode(graphbytes).decode("ascii")
#         url = f"https://mermaid.ink/img/{base64_string}"
#         response = requests.get(url)
 
#         if response.status_code == 200:
#             with open(filename, "wb") as f:
#                 f.write(response.content)
#             return filename
#         else:
#             st.error(f"Mermaid rendering failed: HTTP {response.status_code}")
#             return None
#     except Exception as e:
#         st.error(f"Error rendering ERD: {e}")
#         return None
   
 
 
 
# # =========================
# # UI FLOW (same pattern)
# # =========================
# uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
# target_db = st.selectbox("Target Database", ["Snowflake", "SQL Server", "Oracle", "Redshift", "Synapse"])
 
# if uploaded_file:
#     st.success(f"File `{uploaded_file.name}` uploaded successfully")
 
#     if st.button("Generate Data Model"):
#         with st.spinner("Generating model..."):
#             # Read bytes once; reuse for schema + profiling
#             file_bytes = uploaded_file.getvalue()
#             filename = uploaded_file.name
 
#             schema = extract_schema(file_bytes, filename)
#             prof_df, prof_rows = profile_data(file_bytes, filename)
 
#             result = call_bedrock(schema, prof_rows, target_db=target_db)
#             logical, erd, ddl, scd = parse_output(result)
 
#         # Tabs (unchanged)
#         tabs = st.tabs(["Logical Model", "ERD Diagram", "SQL DDL"])
 
#         # --- Logical Model Tab ---
#         with tabs[0]:
#             st.write("### Data Profiling Summary")
#             st.caption("Computed from a sample of the uploaded file to guide modeling decisions.")
#             st.dataframe(prof_df, use_container_width=True)
 
#             # Download profiling as CSV
#             prof_csv = prof_df.to_csv(index=False).encode("utf-8")
#             st.download_button("Download Profiling CSV", prof_csv, file_name="profiling_summary.csv", mime="text/csv")
 
#             st.write("### Logical Data Model")
#             st.text_area("", logical, height=260)
 
#             if scd:
#                 st.write("### SCD Recommendations")
#                 st.text_area("", scd, height=200)
 
#         # --- ERD Tab ---
#         with tabs[1]:
#             if erd:
#                 with st.spinner("Rendering ERD diagram..."):
#                     erd_file = save_erd_as_png(erd, "erd.png")
#                 if erd_file:
#                     st.image(erd_file, caption="Entity-Relationship Diagram", use_container_width=True)
#                     with open(erd_file, "rb") as f:
#                         st.download_button("Download ERD (PNG)", f, file_name="erd.png", mime="image/png")
#                 else:
#                     st.warning("Could not render ERD")
 
#         # --- DDL Tab ---
#         with tabs[2]:
#             if ddl:
#                 st.write("### Generated SQL DDL")
#                 st.code(ddl, language="sql")
#                 st.download_button("Download SQL DDL", ddl, file_name="model.sql", mime="text/sql")
 
#         # Debug (unchanged)
#         with st.expander("Raw Model Output (for debugging)"):
#             st.text(result)
 
 
# # =========================
# # SIDEBAR HELP (same pattern)
# # =========================
# st.sidebar.header("How to Use")
# st.sidebar.markdown("""
# 1. Upload a CSV or Excel file  
# 2. Select your target database  
# 3. Click **Generate Data Model**  
# 4. Review results in tabs:  
#    - Logical model  
#    - ERD diagram  
#    - SQL DDL  
# 5. Download artifacts as needed
# """)
# st.sidebar.info("Only the first rows are analyzed for speed. Validate results before production.")
 
 
 
 
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
uploaded_files = st.file_uploader(
    "Upload one or more CSV/Excel files",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)
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
 