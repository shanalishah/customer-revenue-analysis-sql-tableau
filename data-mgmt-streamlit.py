# app.py
import os
import re
import io
import json
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import altair as alt

# =========================
# Page setup
# =========================
st.set_page_config(page_title="Customer Revenue Analysis – Data Visualization", layout="wide")
st.title("Customer Revenue Analysis — Data Visualization")
st.caption("Browse query outputs, filter data, view SQL, and generate quick interactive charts.")

# =========================
# Config
# =========================
DATA_DIR = "data"                   # where q1.csv ... q11.csv live
SQL_FILE = "queries_shan.sql"       # your SQL file with query blocks
CSV_PATTERN = r"^q(\d+)\.csv$"      # q1.csv, q2.csv, ...

# =========================
# Helpers
# =========================
def make_pretty_unique_columns(cols: List[str]) -> List[str]:
    """Title-case columns and ensure uniqueness with friendly suffixes."""
    pretty = [c.replace("_", " ").strip().title() for c in cols]
    seen = {}
    out = []
    for name in pretty:
        if name not in seen:
            seen[name] = 1
            out.append(name)
        else:
            seen[name] += 1
            out.append(f"{name} ({seen[name]})")
    return out

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = make_pretty_unique_columns(df.columns)
    return df

def list_query_csvs(data_dir: str) -> List[Tuple[int, str]]:
    """Return list of (query_number, filename) sorted by number."""
    items = []
    if not os.path.isdir(data_dir):
        return items
    for f in os.listdir(data_dir):
        m = re.match(CSV_PATTERN, f, flags=re.IGNORECASE)
        if m:
            items.append((int(m.group(1)), f))
    items.sort(key=lambda x: x[0])
    return items

@st.cache_data
def parse_sql_blocks(sql_text: str) -> Dict[int, str]:
    """
    Parse SQL file with blocks like:
    -- [Query 1] A title...
    SELECT ...
    -- [Query 2] Another title...
    """
    blocks = {}
    current_idx = None
    current_lines = []

    header_pat = re.compile(r"^\s*--\s*\[Query\s*(\d+)\b.*\]", re.IGNORECASE)

    for line in sql_text.splitlines():
        m = header_pat.match(line)
        if m:
            if current_idx is not None and current_lines:
                blocks[current_idx] = "\n".join(current_lines).strip()
            current_idx = int(m.group(1))
            current_lines = []
        else:
            if current_idx is not None:
                current_lines.append(line)
    if current_idx is not None and current_lines:
        blocks[current_idx] = "\n".join(current_lines).strip()

    return blocks

@st.cache_data
def read_sql_file(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def safe_numeric_slider(label: str, col: pd.Series) -> Tuple[float, float]:
    """
    Slider that won’t crash when min==max.
    Returns (min_val, max_val) chosen.
    When constant, returns (v, v) and shows a disabled slider substitute.
    """
    vmin = float(col.min())
    vmax = float(col.max())
    if vmin == vmax:
        st.caption(f"“{label}” is constant ({vmin}). No range filter applied.")
        return vmin, vmax
    return st.slider(label, min_value=vmin, max_value=vmax, value=(vmin, vmax))

def apply_text_search(df: pd.DataFrame, search: str) -> pd.DataFrame:
    if not search:
        return df
    search = search.strip().lower()
    mask = pd.Series([False] * len(df))
    for c in df.columns:
        # Convert to string to avoid errors
        col = df[c].astype(str).str.lower()
        mask = mask | col.str.contains(re.escape(search), na=False)
    return df[mask]

def quick_chart(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None, chart_type: str = "Bar"):
    chart_df = df[[x_col, y_col] + ([color_col] if color_col else [])].dropna()
    if chart_type == "Bar":
        ch = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X(x_col, sort='-y'),
            y=y_col,
            color=color_col if color_col else alt.value(None),
            tooltip=list(chart_df.columns)
        ).interactive()
    elif chart_type == "Line":
        ch = alt.Chart(chart_df).mark_line(point=True).encode(
            x=x_col, y=y_col,
            color=color_col if color_col else alt.value(None),
            tooltip=list(chart_df.columns)
        ).interactive()
    elif chart_type == "Scatter":
        ch = alt.Chart(chart_df).mark_circle().encode(
            x=x_col, y=y_col,
            color=color_col if color_col else alt.value(None),
            tooltip=list(chart_df.columns),
            size=alt.value(80)
        ).interactive()
    else:
        ch = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X(x_col, sort='-y'),
            y=y_col,
            tooltip=list(chart_df.columns)
        ).interactive()

    st.altair_chart(ch, use_container_width=True)

# =========================
# Load SQL
# =========================
raw_sql = read_sql_file(SQL_FILE)
sql_blocks = parse_sql_blocks(raw_sql) if raw_sql else {}

# =========================
# Sidebar
# =========================
st.sidebar.header("Select a Query")

csvs = list_query_csvs(DATA_DIR)
if not csvs:
    st.sidebar.error(f"No CSVs found in `{DATA_DIR}/`. Add q1.csv, q2.csv, …")
    st.stop()

# Build labels “Query 1”, “Query 2”, …
labels = [f"Query {idx}" for idx, _ in csvs]
selection = st.sidebar.selectbox("Query", labels, index=0)
selected_idx = csvs[labels.index(selection)][0]
selected_csv = csvs[labels.index(selection)][1]
selected_path = os.path.join(DATA_DIR, selected_csv)

# =========================
# Load DataFrame
# =========================
df = load_csv(selected_path)

# =========================
# Main — Data / Filters / Chart / SQL
# =========================
st.subheader(f"{selection}")

# Search
with st.expander("Search & Filter", expanded=True):
    search_text = st.text_input("Search across all columns", placeholder="Type to filter rows…")
    filtered = apply_text_search(df, search_text)

    # Optional: quick numeric sliders for top 2 numeric columns (robust to constant values)
    num_cols = [c for c in filtered.columns if pd.api.types.is_numeric_dtype(filtered[c])]
    if num_cols:
        st.markdown("**Quick numeric range filters**")
        # Limit to at most 2 sliders to keep UI simple
        for i, c in enumerate(num_cols[:2]):
            vmin, vmax = safe_numeric_slider(c, filtered[c])
            if vmin != vmax:  # Only filter if there’s a real range
                filtered = filtered[(filtered[c] >= vmin) & (filtered[c] <= vmax)]

    # Show table
    st.dataframe(filtered, use_container_width=True)

# Quick chart builder
with st.expander("Quick Interactive Chart", expanded=True):
    # Pick X/Y
    x_opt = st.selectbox("X axis", options=list(filtered.columns), index=0, key="x_axis")
    # Try to select a numeric default for Y
    y_default_candidates = [c for c in filtered.columns if pd.api.types.is_numeric_dtype(filtered[c])]
    y_default = y_default_candidates[0] if y_default_candidates else filtered.columns[-1]
    y_opt = st.selectbox("Y axis", options=list(filtered.columns), index=list(filtered.columns).index(y_default), key="y_axis")
    color_opt = st.selectbox("Color (optional)", options=["None"] + list(filtered.columns), index=0, key="color_axis")
    chart_type = st.radio("Chart type", options=["Bar", "Line", "Scatter"], horizontal=True)

    if x_opt and y_opt:
        color_arg = None if color_opt == "None" else color_opt
        try:
            quick_chart(filtered, x_opt, y_opt, color_arg, chart_type)
        except Exception as e:
            st.warning(f"Chart could not be rendered: {e}")

# SQL viewer
with st.expander("View SQL", expanded=False):
    if sql_blocks and selected_idx in sql_blocks:
        st.code(sql_blocks[selected_idx], language="sql")
    elif raw_sql:
        # Fallback when blocks not found: show entire file once
        st.info("No block header found for this query. Showing entire SQL file.")
        st.code(raw_sql, language="sql")
    else:
        st.write("No SQL file found.")

# Download filtered data
with st.expander("Download", expanded=False):
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data (CSV)",
        data=csv_bytes,
        file_name=f"{selection.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )
