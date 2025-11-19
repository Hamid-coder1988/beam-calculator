# -------------------------
# Beam DB loader (table: Beam, columns: Type, Size, ...)
# -------------------------

import streamlit as st
import pandas as pd

# Try psycopg2
try:
    import psycopg2
    HAS_PG = True
except:
    HAS_PG = False

# ---- DB connection ----
def get_conn():
    url = st.secrets["connections"]["postgres"]["url"]
    return psycopg2.connect(url, sslmode="require")


# ---- SAMPLE fallback ----
SAMPLE_ROWS = [
    {"Type": "IPE", "Size": "IPE 200", "A_cm2": 31.2},
    {"Type": "IPE", "Size": "IPE 300", "A_cm2": 45.0},
]


# ---- FUNCTIONS ----
@st.cache_data(show_spinner=False)
def get_types():
    """Return list of distinct Type values."""
    if not HAS_PG:
        df = pd.DataFrame(SAMPLE_ROWS)
        return sorted(df["Type"].unique().tolist())

    try:
        conn = get_conn()
        sql = 'SELECT DISTINCT "Type" FROM "Beam" ORDER BY "Type";'
        df = pd.read_sql(sql, conn)
        return df["Type"].astype(str).tolist()
    except Exception as e:
        st.sidebar.warning(f"DB Error (using sample): {e}")
        df = pd.DataFrame(SAMPLE_ROWS)
        return sorted(df["Type"].unique().tolist())


@st.cache_data(show_spinner=False)
def get_sizes(type_value):
    """Return sizes for a given Type."""
    if not HAS_PG:
        df = pd.DataFrame(SAMPLE_ROWS)
        return df[df["Type"] == type_value]["Size"].astype(str).tolist()

    try:
        conn = get_conn()
        sql = 'SELECT "Size" FROM "Beam" WHERE "Type" = %s ORDER BY "Size";'
        df = pd.read_sql(sql, conn, params=(type_value,))
        return df["Size"].astype(str).tolist()
    except Exception as e:
        st.sidebar.warning(f"DB Error (using sample): {e}")
        df = pd.DataFrame(SAMPLE_ROWS)
        return df[df["Type"] == type_value]["Size"].astype(str).tolist()


@st.cache_data(show_spinner=False)
def get_section(type_value, size_value):
    """Return full row dictionary."""
    if not HAS_PG:
        df = pd.DataFrame(SAMPLE_ROWS)
        f = df[(df["Type"] == type_value) & (df["Size"] == size_value)]
        return f.iloc[0].to_dict() if not f.empty else None

    try:
        conn = get_conn()
        sql = 'SELECT * FROM "Beam" WHERE "Type" = %s AND "Size" = %s LIMIT 1;'
        df = pd.read_sql(sql, conn, params=(type_value, size_value))
        if df.empty:
            return None
        return df.iloc[0].to_dict()
    except Exception as e:
        st.sidebar.warning(f"DB Error (using sample): {e}")
        df = pd.DataFrame(SAMPLE_ROWS)
        f = df[(df["Type"] == type_value) & (df["Size"] == size_value)]
        return f.iloc[0].to_dict() if not f.empty else None


# -------------------------
# UI
# -------------------------

st.header("Section Selection From Database")

types = get_types()
type_sel = st.selectbox("Type", ["-- choose --"] + types)

selected_row = None

if type_sel != "-- choose --":
    sizes = get_sizes(type_sel)
    size_sel = st.selectbox("Size", ["-- choose --"] + sizes)

    if size_sel != "-- choose --":
        selected_row = get_section(type_sel, size_sel)

        if selected_row:
            st.success(f"Loaded section: {size_sel}")
            st.json(selected_row)  # show properties
        else:
            st.error("Could not load this section from DB.")
