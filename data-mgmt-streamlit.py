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
DATA_DIR = REPO_ROOT / "data"            # expects q1.csv ... qN.csv
SQL_FILE = REPO_ROOT / "queries_shan.sql"

# --------- Helpers ---------
SQL_BLOCK_RE = re.compile(
    r"""^\s*--\s*\[?\s*Query\s*(?P<num>\d+)\s*\]?\s*:?\s*(?P<title>.*)\s*$""",
    re.IGNORECASE | re.MULTILINE,
)

def parse_sql_blocks(sql_text: str):
    headers = [(m.start(), m.end(), int(m.group("num")), (m.group("title") or "").strip())
               for m in SQL_BLOCK_RE.finditer(sql_text)]
    blocks = []
    for i, (s, e, num, title) in enumerate(headers):
        body_end = headers[i + 1][0] if i + 1 < len(headers) else len(sql_text)
        body = sql_text[e:body_end].strip()
        blocks.append({"num": num, "title": title or f"Query {num}", "body": body})
    blocks.sort(key=lambda x: x["num"])
    return blocks

@st.cache_data(show_spinner=False)
def load_sql_blocks(sql_path: Path):
    if not sql_path.exists():
        return []
    text = sql_path.read_text(encoding="utf-8", errors="ignore")
    return parse_sql_blocks(text)

def _dedupe_columns(cols):
    seen = {}
    out = []
    for c in cols:
        name = str(c)
        if name not in seen:
            seen[name] = 1
            out.append(name)
        else:
            seen[name] += 1
            out.append(f"{name} ({seen[name]})")
    return out

@st.cache_data(show_spinner=False)
def load_csv(path: Path):
    df = pd.read_csv(path, low_memory=False)
    if df.columns.duplicated().any():
        df.columns = _dedupe_columns(df.columns)
    return df

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
    if pd.isna(min_val) or pd.isna(max_val):
        st.info(f"{label}: no numeric data to filter.")
        return value
    if float(min_val) == float(max_val):
        st.info(f"{label}: single value ({min_val}) — no range filter applied.")
        return (float(min_val), float(max_val))
    return st.slider(label, float(min_val), float(max_val), (float(value[0]), float(value[1])))

# --------- Load SQL metadata ---------
sql_blocks = load_sql_blocks(SQL_FILE)

# Build available queries based on CSV files present
available = []
if DATA_DIR.exists():
    for f in sorted(DATA_DIR.glob("q*.csv"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0)):
        try:
            num = int(re.sub(r"\D", "", f.stem))
        except Exception:
            continue
        blk = next((b for b in sql_blocks if b["num"] == num), None)
        title = (blk["title"] if blk and blk.get("title") else f"Query {num}")
        body = (blk["body"] if blk and blk.get("body") else "")
        available.append({"num": num, "title": title, "csv": f, "sql": body})

if not available:
    st.error(
        "No query outputs found. Please add CSVs to the `data/` folder (e.g., `q1.csv`, `q2.csv`…) "
        "and ensure `queries_shan.sql` includes headers like `-- Query 1: Your Title`."
    )
    st.stop()

# --------- Sidebar ---------
with st.sidebar:
    st.subheader("Select Query")
    labels = [f"Query {q['num']}: {q['title']}" for q in available]
    pick_label = st.selectbox("Query output", labels)
    picked = available[labels.index(pick_label)]

    st.markdown("---")
    # was: "Use the controls below..." (confusing in sidebar)
    st.caption("Controls are in the main panel: search, filters, chart, SQL & downloads.")

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
st.markdown("#### Quick Filters")
num_cols = [c for c in df_filtered.columns if is_numeric(df_filtered[c])]
if num_cols:
    with st.expander("Numeric range filters"):
        for col in num_cols:
            ser = pd.to_numeric(df_filtered[col], errors="coerce")
            col_min, col_max = float(ser.min()), float(ser.max())
            chosen_min, chosen_max = safe_number_slider(
                f"{col} range",
                col_min,
                col_max,
                (col_min, col_max),
            )
            if col_min != col_max:
                ser2 = pd.to_numeric(df_filtered[col], errors="coerce")
                df_filtered = df_filtered[(ser2 >= chosen_min) & (ser2 <= chosen_max)]
else:
    st.info("No numeric columns detected for range filtering.")

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
        pref = {"total_revenue", "revenue", "amount", "avg_spending_per_rental", "total_late_fees"}
        idx = 0
        for i, c in enumerate(y_candidates):
            if c.lower() in pref:
                idx = i
                break
        y_col = st.selectbox("Y-axis (numeric)", options=y_candidates, index=idx)

with chart_cols_right:
    chart_type = st.selectbox("Chart type", ["Bar", "Line", "Area", "Scatter"])
    color_options = ["(none)"] + [c for c in df_filtered.columns if c != x_col]
    color_col = st.selectbox("Color by (optional)", options=color_options, index=0)

if y_col:
    try:
        import altair as alt

        # Build a unique list of required columns (avoid duplicate names)
        needed_cols = [x_col, y_col]
        if color_col != "(none)" and color_col not in needed_cols:
            needed_cols.append(color_col)

        data = df_filtered[needed_cols].copy()

        # Limit x cardinality for readability if x is categorical
        if not is_numeric(data[x_col]) and data[x_col].nunique() > 50:
            top_vals = data[x_col].value_counts().nlargest(50).index
            data = data[data[x_col].isin(top_vals)]

        if chart_type == "Bar":
            mark = alt.Chart(data).mark_bar()
        elif chart_type == "Line":
            mark = alt.Chart(data).mark_line(point=True)
        elif chart_type == "Area":
            mark = alt.Chart(data).mark_area()
        else:
            mark = alt.Chart(data).mark_circle(size=60)

        enc = mark.encode(
            x=alt.X(x_col, sort=None),
            y=alt.Y(y_col),
            tooltip=list(data.columns),
        )
        if color_col != "(none)":
            enc = enc.encode(color=color_col)

        st.altair_chart(enc.properties(height=380, width="container"), use_container_width=True)
    except Exception as e:
        st.warning(f"Chart could not be rendered: {e}")

# --------- SQL Viewer ---------
with st.expander("View SQL for this query"):
    sql_body = (picked.get("sql") or "").strip()
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
