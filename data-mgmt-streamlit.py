# explorer.py

import os
import re
import glob
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Revenue Analysis - Data Visualization", layout="wide")

st.title("Customer Revenue Analysis — Data Visualization")
st.caption("Browse query outputs, filter data, view SQL, and generate quick interactive charts.")

DATA_DIR = "data"           # q1.csv, q2.csv, ...
SQL_PATH = "queries_shan.sql"

# ---------- Helpers ----------
QHEADER_RE = re.compile(r"^\s*--\s*\[?Q(\d+)\]?\s*:\s*(.+?)\s*$", re.IGNORECASE)

def parse_qnum(path: str):
    base = os.path.basename(path).lower()
    digits = "".join(ch for ch in base if ch.isdigit())
    return int(digits) if digits else None

def parse_sql_blocks(sql_text: str):
    """
    Parses queries_shan.sql into {qnum: {"title": str, "sql": str}}
    Header format per block (single line):
      -- [Q1]: Title here
    The block continues until the next header or EOF.
    """
    blocks = {}
    current_q = None
    current_title = None
    current_lines = []

    for line in sql_text.splitlines():
        m = QHEADER_RE.match(line)
        if m:
            # flush previous
            if current_q is not None:
                blocks[current_q] = {
                    "title": current_title.strip(),
                    "sql": "\n".join(current_lines).strip()
                }
            # start new
            current_q = int(m.group(1))
            current_title = m.group(2)
            current_lines = []
        else:
            if current_q is not None:
                current_lines.append(line)

    # flush last
    if current_q is not None:
        blocks[current_q] = {
            "title": current_title.strip(),
            "sql": "\n".join(current_lines).strip()
        }
    return blocks

# ---------- Load CSVs ----------
csv_paths = sorted(glob.glob(os.path.join(DATA_DIR, "q*.csv")))
available = [(parse_qnum(p), p) for p in csv_paths if parse_qnum(p) is not None]
available.sort(key=lambda x: x[0])

if not available:
    st.warning("No CSVs found in `data/` (expected files like q1.csv, q2.csv, ...).")
    st.stop()

# ---------- Load SQL (optional) ----------
sql_blocks = {}
if os.path.exists(SQL_PATH):
    try:
        with open(SQL_PATH, "r", encoding="utf-8", errors="ignore") as f:
            sql_blocks = parse_sql_blocks(f.read())
    except Exception:
        sql_blocks = {}

# Build labels using SQL titles if present; fallback to simple labels
labels = []
for qnum, _ in available:
    if qnum in sql_blocks and sql_blocks[qnum].get("title"):
        labels.append(f"Query {qnum} – {sql_blocks[qnum]['title']}")
    else:
        labels.append(f"Query {qnum}")

selected_label = st.sidebar.selectbox("Choose a result set", labels, index=0)
selected_idx = labels.index(selected_label)
qnum, csv_path = available[selected_idx]
st.sidebar.write(f"**Selected:** Query {qnum}")

# ---------- Load DataFrame ----------
df = pd.read_csv(csv_path)
if df.shape[0] == 0:
    st.warning("This CSV has 0 rows. Double-check that the query and export match.")

st.write(f"**Rows:** {df.shape[0]}  |  **Columns:** {df.shape[1]}")

# ---------- Search ----------
search = st.text_input("Search across all columns (case-insensitive)")
filtered = df.copy()
if search.strip():
    needle = search.strip().lower()
    mask = pd.Series([False] * len(filtered))
    for col in filtered.columns:
        mask = mask | filtered[col].astype(str).str.lower().str.contains(needle, na=False)
    filtered = filtered[mask]

# ---------- Column Filters ----------
with st.expander("Filter columns"):
    for col in filtered.columns:
        ser = filtered[col]
        if pd.api.types.is_numeric_dtype(ser):
            if len(ser) == 0:
                continue
            min_val = float(ser.min())
            max_val = float(ser.max())
            if min_val == max_val:
                st.caption(f"“{col}” has a single value ({min_val}); numeric filter skipped.")
                continue
            r = st.slider(
                f"{col} range",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),
            )
            filtered = filtered[(filtered[col] >= r[0]) & (filtered[col] <= r[1])]
        else:
            uniques = sorted(ser.dropna().astype(str).unique().tolist())
            if 0 < len(uniques) <= 200:
                picked = st.multiselect(f"{col} values", uniques, default=[])
                if picked:
                    filtered = filtered[filtered[col].astype(str).isin(picked)]

# ---------- Show Table ----------
st.subheader(f"Query {qnum} Results")
st.dataframe(filtered, use_container_width=True)

# ---------- Quick Chart ----------
with st.expander("Quick chart"):
    num_cols = [c for c in filtered.columns if pd.api.types.is_numeric_dtype(filtered[c])]
    other_cols = [c for c in filtered.columns if c not in num_cols]

    if not filtered.empty and (num_cols or other_cols):
        x = st.selectbox("X-axis", other_cols + num_cols, index=0 if other_cols else 0)
        y = st.selectbox("Y-axis (numeric)", num_cols, index=0 if num_cols else None)

        if y:
            chart_type = st.radio("Chart type", ["Bar", "Line", "Area"], horizontal=True)
            plot_df = filtered[[x, y]].copy()
            try:
                if chart_type == "Bar":
                    st.bar_chart(plot_df.set_index(x))
                elif chart_type == "Line":
                    st.line_chart(plot_df.set_index(x))
                else:
                    st.area_chart(plot_df.set_index(x))
            except Exception as e:
                st.info(f"Couldn’t plot chart: {e}")
        else:
            st.info("Pick a numeric column for Y-axis to plot.")
    else:
        st.info("No data to chart.")

# ---------- SQL Viewer ----------
with st.expander("View SQL (read-only)"):
    if qnum in sql_blocks and sql_blocks[qnum].get("sql"):
        if sql_blocks[qnum].get("title"):
            st.markdown(f"**{sql_blocks[qnum]['title']}**")
        st.code(sql_blocks[qnum]["sql"], language="sql")
    elif os.path.exists(SQL_PATH):
        st.caption("No block header found for this query. Showing entire file.")
        with open(SQL_PATH, "r", encoding="utf-8", errors="ignore") as f:
            st.code(f.read(), language="sql")
    else:
        st.caption("`queries_shan.sql` not found.")
