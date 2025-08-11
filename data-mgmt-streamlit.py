# app.py
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st

# --------- App config ---------
st.set_page_config(
    page_title="Customer Revenue Analysis — Interactive Data Visualization",
    layout="wide",
)

st.title("Customer Revenue Analysis — Interactive Data Visualization")
st.caption("Browse query outputs, filter data, view SQL, and generate quick interactive charts.")

# --------- Paths ---------
REPO_ROOT = Path(__file__).parent if "__file__" in globals() else Path(".")
DATA_DIR = REPO_ROOT / "data"            # expects q1.csv ... q11.csv
SQL_FILE = REPO_ROOT / "queries_shan.sql"

# --------- Helpers ---------
SQL_BLOCK_RE = re.compile(
    r"""--\s*Query\s*(?P<num>\d+)\s*:\s*(?P<title>.+?)\n(?P<body>.*?)(?=(\n--\s*Query\s*\d+\s*:)|\Z)""",
    re.IGNORECASE | re.DOTALL,
)

def parse_sql_blocks(sql_text: str):
    """
    Returns ordered list of dicts: {num:int, title:str, body:str}
    Requires headers like:
      -- Query 1: Top 10 Revenue-Generating Customers
      SELECT ...
    """
    blocks = []
    for m in SQL_BLOCK_RE.finditer(sql_text):
        num = int(m.group("num"))
        title = m.group("title").strip()
        body = m.group("body").strip()
        blocks.append({"num": num, "title": title, "body": body})
    blocks.sort(key=lambda x: x["num"])
    return blocks

@st.cache_data(show_spinner=False)
def load_sql_blocks(sql_path: Path):
    if not sql_path.exists():
        return []
    text = sql_path.read_text(encoding="utf-8", errors="ignore")
    return parse_sql_blocks(text)

@st.cache_data(show_spinner=False)
def load_csv(path: Path):
    return pd.read_csv(path, low_memory=False)

def filter_df(df: pd.DataFrame, search: str) -> pd.DataFrame:
    if not search:
        return df
    s = search.strip().lower()
    if not s:
        return df
    return df[df.apply(lambda row: row.astype(str).str.lower().str.contains(s, na=False).any(), axis=1)]

def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)

def safe_number_slider(label, min_val, max_val, value):
    """Avoid Streamlit slider crash when min == max."""
    if pd.isna(min_val) or pd.isna(max_val):
        st.info(f"{label}: no numeric data to filter.")
        return value
    if min_val == max_val:
        st.info(f"{label}: single value ({round(float(min_val), 4)}) — no range filter applied.")
        return (min_val, max_val)
    return st.slider(label, float(min_val), float(max_val), (float(value[0]), float(value[1])))

def series_from(df: pd.DataFrame, col: str) -> pd.Series:
    """Always return a Series even if duplicate column names exist."""
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj

# --------- Load SQL metadata ---------
sql_blocks = load_sql_blocks(SQL_FILE)

# If no blocks were parsed, fall back to generic names for q1..q99
fallback_blocks = [{"num": i, "title": f"Query {i}", "body": ""} for i in range(1, 100)]

# Build available queries based on CSV files present
available = []
for blk in (sql_blocks if sql_blocks else fallback_blocks):
    csv_path = DATA_DIR / f"q{blk['num']}.csv"
    if csv_path.exists():
        available.append({"num": blk["num"], "title": blk["title"], "csv": csv_path, "sql": blk["body"]})

if not available:
    st.error(
        "No query outputs found. Please add CSVs to the `data/` folder (e.g., `q1.csv`, `q2.csv`…) "
        "and ensure `queries_shan.sql` includes headers like `-- Query 1: Your Title`."
    )
    st.stop()

# --------- Sidebar ---------
with st.sidebar:
    st.subheader("Select Query")
    options = {f"Query {q['num']}: {q['title']}": q for q in available}
    pick_label = st.selectbox("Query output", list(options.keys()))
    picked = options[pick_label]

# --------- Load chosen CSV ---------
df = load_csv(picked["csv"])
st.markdown(f"### {pick_label}")

# --------- Top info row ---------
c1, c2, c3, c4 = st.columns([2, 2, 2, 3])
with c1:
    st.metric("Rows", f"{len(df):,}")
with c2:
    st.metric("Columns", f"{df.shape[1]:,}")
with c3:
    st.metric("File", picked["csv"].name)
with c4:
    search = st.text_input("Search across all columns", placeholder="Type to filter…")

df_filtered = filter_df(df, search)

# --------- Numeric filters ---------
st.markdown("#### Numeric Range Filters")
num_cols = [c for c in df_filtered.columns if is_numeric(df_filtered[c])]
if num_cols:
    with st.expander("Show/hide numeric filters", expanded=False):
        for col in num_cols:
            col_num = pd.to_numeric(df_filtered[col], errors="coerce")
            col_min, col_max = col_num.min(), col_num.max()
            chosen_min, chosen_max = safe_number_slider(
                f"{col} range",
                col_min,
                col_max,
                (col_min, col_max),
            )
            if not pd.isna(col_min) and not pd.isna(col_max) and col_min != col_max:
                df_filtered = df_filtered[
                    (pd.to_numeric(df_filtered[col], errors="coerce") >= chosen_min) &
                    (pd.to_numeric(df_filtered[col], errors="coerce") <= chosen_max)
                ]
else:
    st.info("No numeric columns with a usable range to filter.")

# --------- Data preview ---------
st.markdown("#### Data Preview")
st.dataframe(df_filtered.head(100), use_container_width=True)

# --------- Quick Chart Builder ---------
st.markdown("#### Quick Interactive Chart")
chart_cols_left, chart_cols_right = st.columns([3, 2])

with chart_cols_left:
    x_col = st.selectbox("X-axis", options=list(df_filtered.columns), index=0)

    y_candidates = [c for c in df_filtered.columns if is_numeric(df_filtered[c])]
    if not y_candidates:
        st.info("No numeric columns available for Y-axis.")
        y_col = None
    else:
        default_y_idx = 0
        for i, c in enumerate(y_candidates):
            if c.lower() in ("total_revenue", "revenue", "amount", "avg_spending_per_rental", "total_late_fees"):
                default_y_idx = i
                break
        y_col = st.selectbox("Y-axis (numeric)", options=y_candidates, index=default_y_idx)

with chart_cols_right:
    chart_type = st.selectbox("Chart type", ["Bar", "Line", "Area", "Scatter"])
    # prevent duplicate with Y
    color_options = ["(none)"] + [c for c in df_filtered.columns if c != x_col and (y_col is None or c != y_col)]
    color_col = st.selectbox("Color by (optional)", options=color_options, index=0)

if y_col:
    try:
        import altair as alt

        cols = [x_col, y_col] + ([] if color_col == "(none)" else [color_col])
        data = df_filtered.loc[:, cols].copy()

        # Drop duplicate-named columns to satisfy Altair/Vega
        data = data.loc[:, ~data.columns.duplicated()].copy()

        # Ensure Series semantics for x
        x_series = series_from(data, x_col)

        # Limit cardinality for categorical X
        if not is_numeric(x_series):
            top_n = 50
            uniq = x_series.dropna().astype(str).unique()
            if len(uniq) > top_n:
                top_vals = pd.Series(x_series).value_counts().nlargest(top_n).index
                data = data[data[x_col].astype(str).isin(top_vals)]

        # Choose mark
        mark = (
            alt.Chart(data).mark_bar()
            if chart_type == "Bar" else
            alt.Chart(data).mark_line()
            if chart_type == "Line" else
            alt.Chart(data).mark_area()
            if chart_type == "Area" else
            alt.Chart(data).mark_circle(size=60)
        )

        enc = mark.encode(
            x=alt.X(x_col, sort=None),
            y=alt.Y(y_col),
            tooltip=list(data.columns)
        )

        if color_col != "(none)" and color_col in data.columns and color_col != y_col:
            enc = enc.encode(color=color_col)

        st.altair_chart(enc.properties(height=380, width="container"), use_container_width=True)

    except Exception as e:
        st.warning(f"Chart could not be rendered: {e}")

# --------- SQL Viewer ---------
with st.expander("View SQL for this query"):
    sql_body = picked["sql"].strip()
    if sql_body:
        sql_body = re.sub(r"\n{3,}", "\n\n", sql_body)
        st.code(sql_body, language="sql")
    else:
        st.info("No SQL block found in `queries_shan.sql` for this query number.")

# --------- Downloads ---------
st.markdown("#### Download")
csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download filtered CSV",
    data=csv_bytes,
    file_name=f"query_{picked['num']}_filtered.csv",
    mime="text/csv",
)
