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
    # search across all columns as strings (safe for mixed dtypes)
    mask = df.apply(lambda row: row.astype(str).str.lower().str.contains(s, na=False).any(), axis=1)
    return df[mask]

def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)

def numeric_range_slider(df: pd.DataFrame, col: str):
    """Render a range slider for a numeric column; returns (min,max) or None if skipped."""
    series = pd.to_numeric(df[col], errors="coerce")
    col_min = series.min()
    col_max = series.max()
    # Hide sliders for single-value or all-NaN columns
    if pd.isna(col_min) or pd.isna(col_max) or col_min == col_max:
        return None
    return st.slider(f"{col} range", float(col_min), float(col_max), (float(col_min), float(col_max)))

# --------- Load SQL metadata ---------
sql_blocks = load_sql_blocks(SQL_FILE)

# If no blocks were parsed, fall back to generic names for q1..q99 (based on files present)
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

# --------- Sidebar: query picker ---------
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

# --------- Numeric filters (dynamic) ---------
st.markdown("#### Numeric Range Filters")
num_cols = [c for c in df_filtered.columns if is_numeric(df_filtered[c])]
at_least_one_slider = False
if num_cols:
    with st.expander("Show filters"):
        for col in num_cols:
            rng = numeric_range_slider(df_filtered, col)
            if rng is None:
                continue  # hide single-value / non-numeric columns
            at_least_one_slider = True
            series = pd.to_numeric(df_filtered[col], errors="coerce")
            df_filtered = df_filtered[(series >= rng[0]) & (series <= rng[1])]
if num_cols and not at_least_one_slider:
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
    color_choices = ["(none)"] + [c for c in df_filtered.columns]
    color_col = st.selectbox("Color by (optional)", options=color_choices, index=0)

# --------- Render chart ---------
if y_col:
    try:
        import altair as alt

        # Build chart dataframe including only selected columns
        cols = [x_col, y_col]
        if color_col != "(none)":
            cols.append(color_col)

        data = df_filtered[cols].copy()

        # De-duplicate column names if user picked the same one twice
        # (Altair requires unique field names)
        new_cols = []
        seen = {}
        for c in data.columns:
            if c not in seen:
                seen[c] = 1
                new_cols.append(c)
            else:
                seen[c] += 1
                new_cols.append(f"{c}__{seen[c]}")
        data.columns = new_cols

        # Map back to renamed columns
        x_name = new_cols[0]
        y_name = new_cols[1]
        color_name = None
        if color_col != "(none)":
            color_name = new_cols[2]

        # Limit cardinality for categorical x
        if not is_numeric(data[x_name]):
            top_n = 50
            if data[x_name].nunique(dropna=True) > top_n:
                top_vals = data[x_name].value_counts(dropna=False).nlargest(top_n).index
                data = data[data[x_name].isin(top_vals)]

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
            x=alt.X(x_name, sort=None),
            y=alt.Y(y_name),
            tooltip=list(data.columns),
        )
        if color_name:
            enc = enc.encode(color=color_name)

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
