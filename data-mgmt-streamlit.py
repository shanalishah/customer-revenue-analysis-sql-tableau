import re
import io
import glob
import os
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import altair as alt


# ---------- Config ----------
st.set_page_config(page_title="Customer Revenue Analysis – SQL Explorer", layout="wide")


# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def load_csvs(data_glob: str = "data/q*.csv") -> Dict[str, pd.DataFrame]:
    files = sorted(glob.glob(data_glob))
    out = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]  # e.g., q3
        try:
            df = pd.read_csv(f)
        except Exception:
            # Try ; as sep if needed
            df = pd.read_csv(f, sep=";")
        out[name] = df
    return out


@st.cache_data(show_spinner=False)
def read_sql_file(path: str = "queries_shan.sql") -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def best_effort_extract_query(sql_text: str, q_key: str) -> str:
    """
    Try to extract the SQL for qN from a monolithic SQL file.
    We look for either:
      - a heading like 'Query 3' (case-insensitive)
      - or the first 'SELECT' block after a line containing 'Query 3'
    If we fail, return the whole file (so something is shown).
    """
    if not sql_text:
        return ""

    # Normalize
    text = sql_text.replace("\r\n", "\n")
    q_number = re.sub(r"[^0-9]", "", q_key)  # "q3" -> "3"
    if not q_number:
        return text

    # Find "Query N" marker
    pattern = re.compile(rf"(?i)(^|\n)\s*--?\s*Query\s*{q_number}\b.*\n", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        # Try a simpler pattern
        pattern2 = re.compile(rf"(?i)(^|\n).*Query\s*{q_number}.*\n")
        m = pattern2.search(text)

    if not m:
        # fallback: return entire file
        return text.strip()

    start = m.end()
    # From start, find next 'Query \d' or end of file
    pattern_next = re.compile(r"(?i)\n\s*--?\s*Query\s*\d+\b.*\n")
    m2 = pattern_next.search(text, start)
    block = text[start: m2.start()] if m2 else text[start:]

    # Optional: try to cut at the first standalone ';' that ends a statement
    # but keep it simple—return the block
    return block.strip()


def df_text_search(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query:
        return df
    query_l = query.lower().strip()
    mask = pd.Series(False, index=df.index)
    for col in df.columns:
        # convert to string safely
        s = df[col].astype(str).str.lower()
        mask = mask | s.str.contains(query_l, na=False)
    return df[mask]


def add_column_filters(df: pd.DataFrame) -> pd.DataFrame:
    # Render per-column filters in an expander
    with st.expander("Column filters", expanded=False):
        filtered = df.copy()
        for col in df.columns:
            col_data = df[col]
            if pd.api.types.is_numeric_dtype(col_data):
                min_val, max_val = float(col_data.min()), float(col_data.max())
                sel_min, sel_max = st.slider(
                    f"{col} range", min_val, max_val, (min_val, max_val)
                )
                filtered = filtered[(filtered[col] >= sel_min) & (filtered[col] <= sel_max)]
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                min_d, max_d = col_data.min(), col_data.max()
                sel = st.date_input(
                    f"{col} between",
                    value=(min_d.date() if hasattr(min_d, 'date') else None,
                           max_d.date() if hasattr(max_d, 'date') else None),
                )
                if isinstance(sel, tuple) and len(sel) == 2 and all(sel):
                    start, end = pd.to_datetime(sel[0]), pd.to_datetime(sel[1])
                    filtered = filtered[(filtered[col] >= start) & (filtered[col] <= end)]
            else:
                # categorical / text
                # If unique small, offer multiselect; else, skip to avoid heavy UI
                uniq = col_data.dropna().unique()
                if len(uniq) > 0 and len(uniq) <= 50:
                    choices = st.multiselect(f"{col} is in…", options=sorted(map(str, uniq)))
                    if choices:
                        filtered = filtered[filtered[col].astype(str).isin(choices)]
        return filtered


def chart_builder(df: pd.DataFrame):
    st.subheader("Quick chart")
    if df.empty:
        st.info("No data to chart.")
        return

    cols = list(df.columns)
    x = st.selectbox("X axis", cols, index=0)
    y = st.selectbox("Y axis (numeric recommended)", cols, index=min(1, len(cols) - 1))
    chart_type = st.radio("Chart type", ["Bar", "Line", "Area"], horizontal=True)

    # Attempt type casting for y
    data = df.copy()
    # If y is not numeric, try to coerce
    if not pd.api.types.is_numeric_dtype(data[y]):
        data[y] = pd.to_numeric(data[y], errors="coerce")

    # If x looks like date as string, try parse
    if not pd.api.types.is_datetime64_any_dtype(data[x]):
        try:
            data[x] = pd.to_datetime(data[x])
        except Exception:
            pass

    base = alt.Chart(data).mark_bar()

    if chart_type == "Bar":
        chart = alt.Chart(data).mark_bar().encode(x=alt.X(x, sort=None), y=y, tooltip=list(df.columns))
    elif chart_type == "Line":
        chart = alt.Chart(data).mark_line(point=True).encode(x=alt.X(x, sort=None), y=y, tooltip=list(df.columns))
    else:
        chart = alt.Chart(data).mark_area(opacity=0.5).encode(x=alt.X(x, sort=None), y=y, tooltip=list(df.columns))

    st.altair_chart(chart.properties(height=360), use_container_width=True)


def download_button_for_df(df: pd.DataFrame, filename: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download filtered data (CSV)",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


# ---------- Metadata (friendly names/descriptions) ----------
DEFAULT_TITLES = {
    "q1": "VIP Customers (High Spend & Frequent Rentals)",
    "q2": "Avg Spending per Rental by Country",
    "q3": "Monthly Revenue Trend",
    "q4": "Customers Inactive in Last 6 Months",
    "q5": "Top Genre by Country (Revenue)",
    "q6": "Most Profitable Movie Category",
    "q7": "Customer Lifetime Value (CLV)",
    "q8": "Peak Rental Days",
    "q9": "Query 9",
    "q10": "Query 10",
    "q11": "Query 11",
    "q12": "Query 12",
}

DEFAULT_INTROS = {
    "q1": "Identify high-value, high-frequency renters for loyalty targeting.",
    "q2": "Compare average spend per rental across countries.",
    "q3": "Track revenue by month to reveal seasonality.",
    "q4": "Find customers who may be churning (no rentals in 6 months).",
    "q5": "See which genres drive the most revenue by country.",
    "q6": "Rank categories by total revenue contribution.",
    "q7": "Estimate CLV using revenue and active duration.",
    "q8": "Spot which weekdays drive the most rentals and revenue.",
}


# ---------- UI ----------
st.title("Customer Revenue Analysis — SQL Results Explorer")
st.caption("Browse query outputs, filter data, view SQL, and build quick charts.")

# Paths: adjust if your repo uses different folders
csv_map = load_csvs("data/q*.csv")
sql_text = read_sql_file("queries_shan.sql")

if not csv_map:
    st.warning("No CSVs found in `data/` (expected files like `q1.csv`, `q2.csv`, ...).")
    st.stop()

# Sidebar: pick a query
available_queries = sorted(csv_map.keys(), key=lambda x: (len(x), x))  # q1..q12
q_key = st.sidebar.selectbox("Select query result", options=available_queries, index=0)

# Titles and intros
nice_title = DEFAULT_TITLES.get(q_key, q_key.upper())
st.header(nice_title)
st.write(DEFAULT_INTROS.get(q_key, ""))

# Show SQL (best effort)
with st.expander("View SQL", expanded=False):
    if sql_text:
        extracted = best_effort_extract_query(sql_text, q_key)
        if extracted:
            st.code(extracted, language="sql")
        else:
            st.info("SQL not found. Showing the entire file.")
            st.code(sql_text, language="sql")
    else:
        st.info("`queries_shan.sql` not found or unreadable.")

# Load data
df = csv_map[q_key].copy()

# Try to parse date-ish columns for better filtering/charting
for c in df.columns:
    if df[c].dtype == object:
        # quick sniff for dates
        sample = df[c].dropna().astype(str).head(10)
        if any(bool(re.search(r"\d{4}-\d{1,2}-\d{1,2}", s)) for s in sample):
            try:
                df[c] = pd.to_datetime(df[c], errors="ignore")
            except Exception:
                pass

# Quick stats row
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Rows", f"{len(df):,}")
with c2:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    st.metric("Numeric columns", len(numeric_cols))
with c3:
    st.metric("Columns", len(df.columns))

# Global text search
search = st.text_input("Search across all columns", placeholder="Type to filter…")
filtered_df = df_text_search(df, search)

# Column filters
filtered_df = add_column_filters(filtered_df)

# Results table
st.subheader("Results")
st.dataframe(filtered_df, use_container_width=True)

# Download filtered CSV
download_button_for_df(filtered_df, filename=f"{q_key}_filtered.csv")

# Chart
chart_builder(filtered_df)

st.markdown("---")
st.caption("Tip: Add more CSVs to the `data/` folder and corresponding SQL to `queries_shan.sql` to extend the app.")
