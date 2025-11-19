# Paste this (replace your previous DB-selection block)
import streamlit as st
import pandas as pd
import math
from datetime import datetime, date
from io import BytesIO

# try to import psycopg2; if missing app will warn and use in-memory sample
try:
    import psycopg2
    HAS_PG = True
except Exception:
    psycopg2 = None
    HAS_PG = False
    st.sidebar.warning("psycopg2 not installed — DB disabled. Install psycopg2-binary in your environment to enable Railway DB.")

from urllib.parse import urlparse

# ------------------------------------------------------------------
# DB connection helper: expects credentials in .streamlit/secrets.toml
# Example secrets.toml:
# [postgres]
# host = "amanote.proxy.rlwy.net"
# port = "15500"
# database = "railway"
# user = "postgres"
# password = "YOUR_PASSWORD"
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_conn():
    if not HAS_PG:
        raise RuntimeError("psycopg2 not available")
    pg = st.secrets.get("postgres")
    if not pg:
        raise RuntimeError("No postgres credentials found in st.secrets['postgres']")
    host = pg.get("host")
    port = pg.get("port")
    database = pg.get("database")
    user = pg.get("user")
    password = pg.get("password")

    # note: Railway often requires sslmode='require'
    conn = psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        sslmode='require'
    )
    return conn

# -------------------------
# Fallback sample data (used if DB not available)
# -------------------------
SAMPLE_ROWS = [
    {"type": "IPE", "name": "IPE 200",
     "A_cm2": 31.2, "S_y_cm3": 88.0, "S_z_cm3": 16.0,
     "I_y_cm4": 1760.0, "I_z_cm4": 64.0, "J_cm4": 14.0, "c_max_mm": 100.0,
     "Wpl_y_cm3": 96.8, "Wpl_z_cm3": 18.0, "alpha_curve": 0.49,
     "flange_class_db": "Class 1/2", "web_class_bending_db": "Class 2", "web_class_compression_db": "Class 3",
     "Iw_cm6": 2500.0, "It_cm4": 14.0
    },
    # add more sample entries here if you want
]

# -------------------------
# DB query helpers (safe)
# -------------------------
@st.cache_data(show_spinner=False)
def get_families_from_db():
    # If DB unavailable, return families from sample
    if not HAS_PG:
        df = pd.DataFrame(SAMPLE_ROWS)
        return sorted(df['type'].dropna().unique().tolist())
    try:
        conn = get_conn()
        sql = "SELECT DISTINCT type FROM IPE_11_25 ORDER BY Type;"
        df = pd.read_sql(sql, conn)
        return sorted(df['Type'].dropna().tolist())
    except Exception as e:
        st.sidebar.warning(f"Could not load families from DB (using sample). Error: {e}")
        df = pd.DataFrame(SAMPLE_ROWS)
        return sorted(df['type'].dropna().unique().tolist())

@st.cache_data(show_spinner=False)
def get_sizes_for_type(type):
    if not HAS_PG:
        df = pd.DataFrame(SAMPLE_ROWS)
        df_f = df[df['Type'] == type]
        return sorted(df_f['name'].astype(str).tolist())
    try:
        conn = get_conn()
        sql = """
          SELECT name
          FROM beam_sections
          WHERE type = %s
          ORDER BY name;
        """
        df = pd.read_sql(sql, conn, params=(Type,))
        return df['name'].astype(str).tolist()
    except Exception as e:
        st.sidebar.warning(f"Could not load sizes from DB (using sample). Error: {e}")
        df = pd.DataFrame(SAMPLE_ROWS)
        df_f = df[df['type'] == type]
        return sorted(df_f['name'].astype(str).tolist())

@st.cache_data(show_spinner=False)
def get_section_row(type, name):
    if not HAS_PG:
        df = pd.DataFrame(SAMPLE_ROWS)
        df_f = df[(df['Type'] == Type) & (df['name'].astype(str) == str(name))]
        if df_f.empty:
            return None
        return df_f.iloc[0].to_dict()
    try:
        conn = get_conn()
        sql = """
          SELECT *
          FROM beam_sections
          WHERE type = %s AND name = %s
          LIMIT 1;
        """
        df = pd.read_sql(sql, conn, params=(type, name))
        if df.empty:
            return None
        return df.iloc[0].to_dict()
    except Exception as e:
        st.sidebar.warning(f"Could not load section row from DB (using sample). Error: {e}")
        df = pd.DataFrame(SAMPLE_ROWS)
        df_f = df[(df['Type'] == Type) & (df['name'].astype(str) == str(name))]
        if df_f.empty:
            return None
        return df_f.iloc[0].to_dict()

# -------------------------
# UI: type -> size -> load properties
# -------------------------
st.header("Section selection (DB)")
families = get_families_from_db()
if not families:
    st.info("No families found in DB or sample. Use 'Use custom section' to enter properties manually.")
type = st.selectbox("Section type (DB)", options=["-- choose --"] + families)

selected_row = None
if type and type != "-- choose --":
    sizes = get_sizes_for_type(Type)
    if not sizes:
        st.info("No section sizes found for chosen type.")
    size = st.selectbox("Section size (DB)", options=["-- choose --"] + sizes)
    if size and size != "-- choose --":
        selected_row = get_section_row(Type, size)
        if selected_row:
            st.success(f"Loaded {selected_row.get('name')} from DB (Type={Type})")
        else:
            st.error("Could not find the selected section (check DB column names / values).")


