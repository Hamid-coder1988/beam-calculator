import streamlit as st
import pandas as pd
import math
from datetime import datetime, date
from io import BytesIO
import psycopg2
from urllib.parse import urlparse

# --- assumes get_conn() exists as in your app ---
import pandas as pd

@st.cache_data
def get_families_from_db():
    conn = get_conn()
    sql = "SELECT DISTINCT family FROM beam_sections ORDER BY family;"
    return pd.read_sql(sql, conn)['family'].dropna().tolist()

@st.cache_data
def get_sizes_for_family(family):
    conn = get_conn()
    # 'name' assumed to be the size column (first column from Excel)
    sql = """
      SELECT name
      FROM beam_sections
      WHERE family = %s
      ORDER BY name;
    """
    return pd.read_sql(sql, conn, params=(family,))['name'].tolist()

@st.cache_data
def get_section_row(family, name):
    conn = get_conn()
    sql = """
      SELECT *
      FROM beam_sections
      WHERE family = %s AND name = %s
      LIMIT 1;
    """
    df = pd.read_sql(sql, conn, params=(family, name))
    if df.empty:
        return None
    return df.iloc[0].to_dict()

# UI
families = get_families_from_db()
family = st.selectbox("Section family (DB)", options=["-- choose --"] + families)
selected_row = None
if family and family != "-- choose --":
    sizes = get_sizes_for_family(family)
    size = st.selectbox("Section size (DB)", options=["-- choose --"] + sizes)
    if size and size != "-- choose --":
        selected_row = get_section_row(family, size)
        st.success(f"Loaded {selected_row.get('name') if selected_row else 'n/a'} from DB")
