# beam_design_app.py
# EngiSnap — Standard steel beam checks (DB-backed, compact UI)

import streamlit as st
import pandas as pd
import math
from datetime import datetime, date
import traceback
import re
import numbers
import numpy as np

# -------------------------
# Optional: install psycopg2-binary:
# pip install psycopg2-binary
# -------------------------
try:
    import psycopg2
    HAS_PG = True
except Exception:
    psycopg2 = None
    HAS_PG = False

# -------------------------
# get_conn() using st.secrets (UNCHANGED)
# -------------------------
def get_conn():
    if not HAS_PG:
        raise RuntimeError("psycopg2 not installed. Install with: pip install psycopg2-binary")
    try:
        pg = st.secrets["postgres"]
        host = pg.get("host")
        port = pg.get("port")
        database = pg.get("database")
        user = pg.get("user")
        password = pg.get("password")
        sslmode = pg.get("sslmode", "require")
        if not (host and database and user and password):
            raise KeyError("postgres secrets incomplete")
        return psycopg2.connect(
            host=host, port=port, database=database,
            user=user, password=password, sslmode=sslmode
        )
    except Exception:
        try:
            from urllib.parse import urlparse
            dburl = st.secrets["DATABASE_URL"]
            parsed = urlparse(dburl)
            return psycopg2.connect(
                dbname=parsed.path.lstrip("/"),
                user=parsed.username,
                password=parsed.password,
                host=parsed.hostname,
                port=parsed.port,
                sslmode="require"
            )
        except Exception as e:
            raise RuntimeError(
                "Database credentials not found in st.secrets. "
                "Add [postgres] or DATABASE_URL to secrets."
            ) from e


# -------------------------
# SQL runner
# -------------------------
def run_sql(sql, params=None):
    try:
        conn = get_conn()
    except Exception as e:
        return None, f"Connection error: {e}"
    try:
        df = pd.read_sql(sql, conn, params=params)
        conn.close()
        return df, None
    except Exception as e:
        tb = traceback.format_exc()
        try:
            conn.close()
        except Exception:
            pass
        return None, f"{e}\n\n{tb}"

# -------------------------
# Fallback sample rows
# -------------------------
SAMPLE_ROWS = [
    {"Type": "IPE", "Size": "IPE 200", "A_cm2": 31.2, "S_y_cm3": 88.0, "S_z_cm3": 16.0,
     "I_y_cm4": 1760.0, "I_z_cm4": 64.0, "J_cm4": 14.0, "c_max_mm": 100.0,
     "Wpl_y_cm3": 96.8, "Wpl_z_cm3": 18.0, "alpha_curve": 0.49,
     "flange_class_db": "Class 1/2", "web_class_bending_db": "Class 2", "web_class_compression_db": "Class 3",
     "Iw_cm6": 2500.0, "It_cm4": 14.0
    },
    {"Type": "CHS", "Size": "CHS 150x5", "A_cm2": 17.66, "S_y_cm3": 20.0, "S_z_cm3": 20.0,
     "I_y_cm4": 100.0, "I_z_cm4": 100.0, "J_cm4": 5.0, "c_max_mm": 75.0,
     "Wpl_y_cm3": 22.0, "Wpl_z_cm3": 22.0, "alpha_curve": 0.34,
     "flange_class_db": "N/A (CHS)", "web_class_bending_db": "N/A (CHS)", "web_class_compression_db": "N/A (CHS)",
     "Iw_cm6": 0.0, "It_cm4": 0.0
    }
]

# -------------------------
# DB helpers (unchanged)
# -------------------------
def list_user_tables():
    sql = """
      SELECT schemaname, tablename
      FROM pg_tables
      WHERE schemaname NOT IN ('pg_catalog','information_schema')
      ORDER BY schemaname, tablename;
    """
    return run_sql(sql)

def table_columns(table_name):
    sql = """
      SELECT column_name, data_type, ordinal_position
      FROM information_schema.columns
      WHERE table_schema = 'public' AND table_name = %s
      ORDER BY ordinal_position;
    """
    return run_sql(sql, params=(table_name,))

def detect_table_and_columns():
    if not HAS_PG:
        return None, None, "no-db"
    tables_df, err = list_user_tables()
    if err:
        return None, None, f"list-tables-error: {err}"
    tables = tables_df['tablename'].astype(str).tolist()

    priorities = []
    if "Beam" in tables: priorities.append
