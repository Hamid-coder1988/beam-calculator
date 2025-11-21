# beam_design_app.py
# EngiSnap — Beam design Eurocode (DB-backed, final merged file)
import streamlit as st
import pandas as pd
import math
from io import BytesIO
from datetime import datetime, date
import traceback
import re
import numbers
import numpy as np

# -------------------------
# Optional: install psycopg2-binary in your environment:
# pip install psycopg2-binary
# -------------------------
try:
    import psycopg2
    HAS_PG = True
except Exception:
    psycopg2 = None
    HAS_PG = False

# -------------------------
# get_conn() using st.secrets recommended.
# If you want to test locally with hard-coded credentials,
# replace the body of get_conn() accordingly (not recommended for production).
# -------------------------
def get_conn():
    if not HAS_PG:
        raise RuntimeError("psycopg2 not installed. Install with: pip install psycopg2-binary")
    # Try structured secrets first
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
        return psycopg2.connect(host=host, port=port, database=database, user=user, password=password, sslmode=sslmode)
    except Exception:
        # Try a DATABASE_URL style secret (Railway / Heroku)
        try:
            from urllib.parse import urlparse
            dburl = st.secrets["DATABASE_URL"]
            parsed = urlparse(dburl)
            return psycopg2.connect(dbname=parsed.path.lstrip("/"), user=parsed.username, password=parsed.password, host=parsed.hostname, port=parsed.port, sslmode="require")
        except Exception as e:
            # Bubble up original error for caller to show friendly message
            raise RuntimeError("Database credentials not found in st.secrets. Add [postgres] or DATABASE_URL to .streamlit/secrets.toml or the Streamlit Cloud Secrets.") from e

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
# Small sample fallback rows (used if DB not available)
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
# DB inspection helpers
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
    if "Beam" in tables: priorities.append("Beam")
    if "beam" in tables and "beam" not in priorities: priorities.append("beam")
    for t in tables:
        if 'beam' in t.lower() and t not in priorities:
            priorities.append(t)
    for t in tables:
        if t not in priorities:
            priorities.append(t)

    for tbl in priorities:
        cols_df, err = table_columns(tbl)
        if err:
            continue
        cols = cols_df['column_name'].astype(str).tolist()
        cols_lower = [c.lower() for c in cols]
        if 'type' in cols_lower and 'size' in cols_lower:
            return tbl, cols, None
    return priorities[0] if priorities else None, None, "no-type-size-found"

# -------------------------
# Fetch Types & Sizes preserving DB insertion order (ORDER BY ctid)
# -------------------------
@st.cache_data(show_spinner=False)
def fetch_types_and_sizes():
    if not HAS_PG:
        df = pd.DataFrame(SAMPLE_ROWS)
        types = []
        sizes_map = {}
        for _, r in df.iterrows():
            t = str(r["Type"])
            s = str(r["Size"])
            if t not in types:
                types.append(t); sizes_map[t] = []
            if s not in sizes_map[t]:
                sizes_map[t].append(s)
        return types, sizes_map, "sample"

    tbl, cols, detect_err = detect_table_and_columns()
    if not tbl:
        return [], {}, f"detect-error: {detect_err}"

    sql = f'SELECT "Type", "Size" FROM "{tbl}" ORDER BY ctid;'
    rows_df, err = run_sql(sql)
    if err:
        sql2 = f"SELECT type, size FROM {tbl} ORDER BY ctid;"
        rows_df, err2 = run_sql(sql2)
        if err2:
            return [], {}, f"could-not-read-rows: {err}\n{err2}"

    if rows_df is None or rows_df.empty:
        return [], {}, f"no-rows-read-from-{tbl}"

    types = []
    sizes_map = {}
    for _, row in rows_df.iterrows():
        t = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
        s = str(row.iloc[1]) if pd.notna(row.iloc[1]) else ""
        if t == "":
            continue
        if t not in types:
            types.append(t); sizes_map[t] = []
        if s and s not in sizes_map[t]:
            sizes_map[t].append(s)

    return types, sizes_map, tbl

# -------------------------
# Fetch full DB row for chosen type & size
# -------------------------
@st.cache_data(show_spinner=False)
def get_section_row_db(type_value, size_value, table_name):
    if not HAS_PG:
        df = pd.DataFrame(SAMPLE_ROWS)
        row = df[(df["Type"] == type_value) & (df["Size"] == size_value)]
        return row.iloc[0].to_dict() if not row.empty else None
    row = None
    q_variants = []
    if table_name:
        q_variants = [
            (f'SELECT * FROM "{table_name}" WHERE "Type" = %s AND "Size" = %s LIMIT 1;', (type_value, size_value)),
            (f"SELECT * FROM {table_name} WHERE type = %s AND size = %s LIMIT 1;", (type_value, size_value)),
            (f'SELECT * FROM "{table_name}" WHERE type = %s AND size = %s LIMIT 1;', (type_value, size_value)),
        ]
    else:
        q_variants = [
            ('SELECT * FROM "Beam" WHERE "Type" = %s AND "Size" = %s LIMIT 1;', (type_value, size_value)),
            ('SELECT * FROM beam WHERE type = %s AND size = %s LIMIT 1;', (type_value, size_value)),
        ]
    for q, p in q_variants:
        df_row, err = run_sql(q, params=p)
        if err:
            continue
        if df_row is not None and not df_row.empty:
            row = df_row.iloc[0].to_dict()
            break
    return row

# -------------------------
# Robust numeric parser & pick helper
# -------------------------
def safe_float(val, default=0.0):
    if val is None:
        return default
    if isinstance(val, numbers.Number):
        try:
            return float(val)
        except Exception:
            return default
    if hasattr(val, "item"):
        try:
            v = val.item()
            if isinstance(v, numbers.Number):
                return float(v)
            val = str(v)
        except Exception:
            val = str(val)
    s = str(val).strip()
    if s == "":
        return default
    # comma decimal if no dot; else remove thousands commas
    if s.count(',') > 0 and s.count('.') == 0:
        s = s.replace(',', '.')
    else:
        s = s.replace(',', '')
    s = s.replace(' ', '')
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
    if not m:
        return default
    token = m.group(0)
    try:
        return float(token)
    except Exception:
        return default

def pick(d, *keys, default=None):
    if d is None:
        return default
    for k in keys:
        if k in d and d[k] is not None:
            v = d[k]
            try:
                if hasattr(v, "iloc"):
                    return v.iloc[0]
                if isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0:
                    return v[0]
            except Exception:
                pass
            return v
    kl = {str(k).lower(): k for k in d.keys()}
    for k in keys:
        lk = str(k).lower()
        if lk in kl and d.get(kl[lk]) is not None:
            v = d.get(kl[lk])
            try:
                if hasattr(v, "iloc"):
                    return v.iloc[0]
                if isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0:
                    return v[0]
            except Exception:
                pass
            return v
    # last attempt: search for any key that contains the requested key substring
    for k in keys:
        sk = str(k).lower()
        for col in d.keys():
            if sk in str(col).lower():
                v = d.get(col)
                if v is not None:
                    return v
    return default

# -------------------------
# Page config & header
# -------------------------
st.set_page_config(page_title="EngiSnap Beam Design Eurocode Checker", layout="wide")
st.title("EngiSnap — Design of steel memebers (Eurocode)")

st.markdown("""
**Disclaimer (read carefully)**  
This prototype performs simplified Eurocode-style screening checks. It is **not** a full EN1993 design package.
Use for screening/prototyping only — always have final results verified by a licensed structural engineer.
""")

# -------------------------
# sample df fallback
# -------------------------
df_sample_db = pd.DataFrame(SAMPLE_ROWS)

# -------------------------
# Metadata block
# -------------------------
st.markdown("## Project data")
meta_col1, meta_col2, meta_col3 = st.columns([1,1,1])
with meta_col1:
    doc_name = st.text_input("Document title", value="Beam check", help="Short title for the generated report")
    project_name = st.text_input("Project name", value="", help="Project identifier")
with meta_col2:
    position = st.text_input("Position / Location (e.g. Beam ID)", value="", help="Beam or member ID (used in reports)")
    requested_by = st.text_input("Requested by", value="", help="Name of requestor")
with meta_col3:
    revision = st.text_input("Revision", value="0.1", help="Revision number")
    run_date = st.date_input("Date", value=date.today())

st.markdown("---")

# -------------------------
# Sidebar: material & defaults
# -------------------------
st.sidebar.header("Material & Eurocode defaults")
material = st.sidebar.selectbox("Material", ("S235", "S275", "S355", "Custom"), help="Select steel grade (typical EN names). For custom, enter fy.")
if material == "S235":
    fy = 235.0
elif material == "S275":
    fy = 275.0
elif material == "S355":
    fy = 355.0
else:
    fy = st.sidebar.number_input("Enter yield stress fy (MPa)", value=355.0, min_value=1.0)

sigma_allow_MPa = 0.6 * fy
gamma_M0 = st.sidebar.number_input("γ_M0 (cross-section)", value=1.0)
gamma_M1 = st.sidebar.number_input("γ_M1 (stability/shear)", value=1.0)
st.sidebar.markdown("Buckling effective length factors (K):")
K_z = st.sidebar.number_input("K_z", value=1.0, min_value=0.1, step=0.05)
K_y = st.sidebar.number_input("K_y", value=1.0, min_value=0.1, step=0.05)
K_LT = st.sidebar.number_input("K_LT", value=1.0, min_value=0.1, step=0.05)
K_T = st.sidebar.number_input("K_T", value=1.0, min_value=0.1, step=0.05)
alpha_default_val = 0.49

# -------------------------
# Section selection UI (DB-backed)
# -------------------------
st.header("Section selection")
st.markdown('<span title="Select a standard section from DB (read-only) or use custom.">ⓘ Section selection help</span>', unsafe_allow_html=True)

col_left, col_right = st.columns([1,1])
with col_left:
    types, sizes_map, detected_table = fetch_types_and_sizes()
    if types:
        family = st.selectbox("Section family / Type (DB)", options=["-- choose --"] + types, help="Choose section family (database). If none selected, use Custom.")
    else:
        family = st.selectbox("Section family / Type (DB)", options=["-- choose --"], help="No DB types found.")
with col_right:
    selected_name = None
    selected_row = None
    if family and family != "-- choose --":
        names = sizes_map.get(family, [])
        if names:
            selected_name = st.selectbox("Section size (DB)", options=["-- choose --"] + names, help="Choose section size (database). Selecting a size loads read-only properties.")
            if selected_name and selected_name != "-- choose --":
                selected_row = get_section_row_db(family, selected_name, detected_table if detected_table != "sample" else None)

st.markdown("**Or select Custom**. Standard DB sections are read-only; custom sections are editable.")
use_custom = st.checkbox("Use custom section (enable manual inputs)", help="Tick to enter section properties manually (CUSTOM)")

if (selected_row is None) and not use_custom:
    st.info("Please select a section size from the DB above, or tick 'Use custom section' to enter properties manually.")

# -------------------------
# Cross-section image placeholder (center)
# -------------------------
st.markdown("---")
st.markdown("### Cross-section schematic (placeholder)")
center_cols = st.columns([1, 2, 1])
center_cols[1].markdown(
    """
<div style="
    border: 2px dashed #999;
    border-radius: 6px;
    width: 100%;
    max-width: 420px;
    height: 260px;
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: #fafafa;
    color:#666;
    font-weight:600;
    margin: 8px auto 12px auto;
">
    Image 1 — cross-section placeholder
</div>
<p style="font-size:12px;color:gray;margin-top:-10px;text-align:center">(Replace this box with section image from database when available.)</p>
""",
    unsafe_allow_html=True,
)
st.markdown("---")

# -------------------------
# Map selected_row into use_props and display Section properties (read-only)
# -------------------------
# We will fill the exact layout you had and put database values into the numbered boxes.
if selected_row is not None and not use_custom:
    sr = selected_row
    bad_fields = []

    def get_num_from_row(*keys, default=0.0, fieldname=None):
        raw = pick(sr, *keys, default=None)
        val = safe_float(raw, default=default)
        # flag if raw exists but produced default (likely parse failure)
        if (raw is not None) and (val == default) and (raw != default):
            bad_fields.append((fieldname or keys[0], raw))
        return val

    # numeric/popular fields (try many candidate column names)
    A_cm2 = get_num_from_row("A_cm2", "area_cm2", "area", "A", "area_cm", default=0.0, fieldname="A_cm2")
    S_y_cm3 = get_num_from_row("S_y_cm3", "Sy_cm3", "S_y", "Wy_cm3", "Wy", "Wpl_y_cm3", default=0.0, fieldname="S_y_cm3")
    S_z_cm3 = get_num_from_row("S_z_cm3", "Sz_cm3", "S_z", "Wz_cm3", "Wz", default=0.0, fieldname="S_z_cm3")
    I_y_cm4 = get_num_from_row("I_y_cm4", "Iy_cm4", "I_y", "Iy", "Iyy", default=0.0, fieldname="I_y_cm4")
    I_z_cm4 = get_num_from_row("I_z_cm4", "Iz_cm4", "I_z", "Iz", "Izz", default=0.0, fieldname="I_z_cm4")
    J_cm4 = get_num_from_row("J_cm4", "J", "torsion_const", default=0.0, fieldname="J_cm4")
    c_max_mm = get_num_from_row("c_max_mm", "c_mm", "c", "cmax", default=0.0, fieldname="c_max_mm")
    Wpl_y_cm3 = get_num_from_row("Wpl_y_cm3", "Wpl_y", "W_pl_y", "Wpl_ycm3", "Wy", default=0.0, fieldname="Wpl_y_cm3")
    Wpl_z_cm3 = get_num_from_row("Wpl_z_cm3", "Wpl_z", "W_pl_z", "Wz", default=0.0, fieldname="Wpl_z_cm3")
    Iw_cm6 = get_num_from_row("Iw_cm6", "Iw", "Iw_cm6", default=0.0, fieldname="Iw_cm6")
    It_cm4 = get_num_from_row("It_cm4", "It", "It_cm4", default=0.0, fieldname="It_cm4")
    alpha_curve = get_num_from_row("alpha_curve", "alpha", default=alpha_default_val, fieldname="alpha_curve")

    # classes (strings)
    flange_class_db = str(pick(sr, "flange_class_db", "flange_class", "flange", default="n/a"))
    web_class_bending_db = str(pick(sr, "web_class_bending_db", "web_class_bending", "web_class_bend", "web_class", default="n/a"))
    web_class_compression_db = str(pick(sr, "web_class_compression_db", "web_class_compression", "web_class_comp", default="n/a"))

    # Display in the same exact layout as before (read-only number_inputs)
    st.markdown("### Section properties (from DB — read only)")
    sr_display = {
        "family": pick(sr, "Type", "family", "type", default="DB"),
        "name": pick(sr, "Size", "name", "designation", default=str(pick(sr, "Size", "name", default="DB"))),
        "A_cm2": A_cm2,
        "S_y_cm3": S_y_cm3,
        "S_z_cm3": S_z_cm3,
        "I_y_cm4": I_y_cm4,
        "I_z_cm4": I_z_cm4,
        "J_cm4": J_cm4,
        "c_max_mm": c_max_mm,
        "Wpl_y_cm3": Wpl_y_cm3,
        "Wpl_z_cm3": Wpl_z_cm3,
        "Iw_cm6": Iw_cm6,
        "It_cm4": It_cm4,
        "alpha_curve": alpha_curve,
        "flange_class_db": flange_class_db,
        "web_class_bending_db": web_class_bending_db,
        "web_class_compression_db": web_class_compression_db
    }

    # First row
    c1, c2, c3 = st.columns(3)
    c1.number_input("A (cm²)", value=float(sr_display.get('A_cm2', 0.0)), disabled=True, key="db_A_cm2")
    c2.number_input("S_y (cm³) about y", value=float(sr_display.get('S_y_cm3', 0.0)), disabled=True, key="db_Sy_cm3")
    c3.number_input("S_z (cm³) about z", value=float(sr_display.get('S_z_cm3', 0.0)), disabled=True, key="db_Sz_cm3")

    # Second row
    c4, c5, c6 = st.columns(3)
    c4.number_input("I_y (cm⁴) about y", value=float(sr_display.get('I_y_cm4', 0.0)), disabled=True, key="db_Iy_cm4")
    c5.number_input("I_z (cm⁴) about z", value=float(sr_display.get('I_z_cm4', 0.0)), disabled=True, key="db_Iz_cm4")
    c6.number_input("J (cm⁴) (torsion const)", value=float(sr_display.get('J_cm4', 0.0)), disabled=True, key="db_J_cm4")

    # Third row
    c7, c8, c9 = st.columns(3)
    c7.number_input("c_max (mm)", value=float(sr_display.get('c_max_mm', 0.0)), disabled=True, key="db_c_max_mm")
    c8.number_input("Wpl_y (cm³)", value=float(sr_display.get('Wpl_y_cm3', 0.0)), disabled=True, key="db_Wpl_y")
    c9.number_input("Wpl_z (cm³)", value=float(sr_display.get('Wpl_z_cm3', sr_display.get('Wpl_y_cm3', 0.0))), disabled=True, key="db_Wpl_z")

    # Fourth row – Iw/It
    c10, c11, c12 = st.columns(3)
    c10.number_input("Iw (cm⁶) (warping)", value=float(sr_display.get("Iw_cm6", 0.0)), disabled=True, key="db_Iw_cm6")
    c11.number_input("It (cm⁴) (torsion moment of inertia)", value=float(sr_display.get("It_cm4", 0.0)), disabled=True, key="db_It_cm4")
    c12.empty()

    # Fifth row – class definitions
    cls1, cls2, cls3 = st.columns(3)
    cls1.text_input("Flange class (DB)", value=str(sr_display.get('flange_class_db', "n/a")), disabled=True, key="db_flange_class_txt")
    cls2.text_input("Web class (bending, DB)", value=str(sr_display.get('web_class_bending_db', "n/a")), disabled=True, key="db_web_bending_class_txt")
    cls3.text_input("Web class (compression, DB)", value=str(sr_display.get('web_class_compression_db', "n/a")), disabled=True, key="db_web_comp_class_txt")

    # Sixth row – buckling α
    a1, a2, a3 = st.columns(3)
    alpha_db_val = sr_display.get('alpha_curve', alpha_default_val)
    alpha_label_db = next((lbl for lbl, val in [
        ("0.13 (a)",0.13),("0.21 (b)",0.21),("0.34 (c)",0.34),("0.49 (d)",0.49),("0.76 (e)",0.76)
    ] if abs(val - float(alpha_db_val)) < 1e-8), f"{alpha_db_val}")
    a1.text_input("Buckling α (DB)", value=str(alpha_label_db), disabled=True, key="db_alpha_text")
    a2.empty(); a3.empty()

    # show raw DB row in an expander for debugging (you can remove this)
    with st.expander("Raw DB row (debug)"):
        st.json(sr)

    if bad_fields:
        st.warning("Some DB fields could not be parsed as numbers and default values were used. See raw DB row to inspect.")
else:
    # Custom editable section (unchanged)
    st.markdown("### Section properties (editable - Custom)")
    c1, c2, c3 = st.columns(3)
    A_cm2 = c1.number_input("Area A (cm²)", value=50.0, key="A_cm2_custom")
    S_y_cm3 = c2.number_input("S_y (cm³) about y", value=200.0, key="Sy_custom")
    S_z_cm3 = c3.number_input("S_z (cm³) about z", value=50.0, key="Sz_custom")
    c4, c5, c6 = st.columns(3)
    I_y_cm4 = c4.number_input("I_y (cm⁴) about y", value=1500.0, key="Iy_custom")
    I_z_cm4 = c5.number_input("I_z (cm⁴) about z", value=150.0, key="Iz_custom")
    J_cm4 = c6.number_input("J (cm⁴) (torsion const)", value=10.0, key="J_custom")
    c7, c8, c9 = st.columns(3)
    c_max_mm = c7.number_input("c_max (mm)", value=100.0, key="c_custom")
    Wpl_y_cm3 = c8.number_input("Wpl_y (cm³)", value=0.0, key="Wpl_custom")
    Wpl_z_cm3 = c9.number_input("Wpl_z (cm³)", value=0.0, key="Wplz_custom")
    c10, c11, c12 = st.columns(3)
    Iw_cm6 = c10.number_input("Iw (cm⁶) (warping)", value=0.0, key="Iw_custom")
    It_cm4 = c11.number_input("It (cm⁴) (torsion moment of inertia)", value=0.0, key="It_custom")
    c12.empty()
    st.markdown("Optional: set flange/web class for custom section (overrides auto estimate)")
    cls1, cls2, cls3 = st.columns(3)
    flange_class_choice = cls1.selectbox("Flange class (custom)", ["Auto (calc)", "Class 1", "Class 2", "Class 3", "Class 4"], index=0, key="flange_class_choice_custom")
    web_class_bending_choice = cls2.selectbox("Web class (bending, custom)", ["Auto (calc)", "Class 1", "Class 2", "Class 3", "Class 4"], index=0, key="web_bend_class_choice_custom")
    web_class_compression_choice = cls3.selectbox("Web class (compression, custom)", ["Auto (calc)", "Class 1", "Class 2", "Class 3", "Class 4"], index=0, key="web_comp_class_choice_custom")
    a1, a2, a3 = st.columns(3)
    alpha_options = [("0.13 (a)", 0.13), ("0.21 (b)", 0.21), ("0.34 (c)", 0.34), ("0.49 (d)", 0.49), ("0.76 (e)", 0.76)]
    alpha_labels = [t for t, v in alpha_options]
    alpha_map = {t: v for t, v in alpha_options}
    alpha_label_selected = a1.selectbox("Buckling curve α (custom)", alpha_labels, index=3, help="Choose buckling curve α (a–e).")
    alpha_custom = alpha_map[alpha_label_selected]
    a2.empty(); a3.empty()

# -------------------------
# Material design properties (read-only)
# -------------------------
st.markdown("---")
st.markdown("### Design properties of material (read-only)")
mp1, mp2, mp3 = st.columns(3)
mp1.text_input("Modulus of elasticity E (MPa)", value=f"{210000}", disabled=True)
mp1.text_input("Yield strength fy (MPa)", value=f"{fy}", disabled=True)
mp2.text_input("Shear modulus G (MPa)", value=f"{80769}", disabled=True)
mp2.text_input("Partial factor γ_M0 (cross-sectional)", value=f"{gamma_M0}", disabled=True)
mp3.text_input("Partial factor γ_M1 (buckling / stability)", value=f"{gamma_M1}", disabled=True)
st.markdown("---")

# -------------------------
# Ready cases section (keep unchanged or reuse your original ready_catalog)
# -------------------------
st.markdown("---")
st.markdown("### Ready beam & frame cases (optional)")
st.write("You can enter loads manually below **or** select a ready beam/frame case to auto-fill typical maxima. First choose whether this is a **Beam** or a **Frame**, then pick a category and case.")
use_ready = st.checkbox("Use ready case (select a template to prefill loads)", key="ready_use_case")

# Minimal ready_catalog (same as before) - keep or extend
def ss_udl(span_m: float, w_kN_per_m: float):
    Mmax = w_kN_per_m * span_m**2 / 8.0
    Vmax = w_kN_per_m * span_m / 2.0
    return (0.0, float(Mmax), 0.0, float(Vmax), 0.0)
def ss_point_center(span_m: float, P_kN: float):
    Mmax = P_kN * span_m / 4.0
    Vmax = P_kN / 2.0
    return (0.0, float(Mmax), 0.0, float(Vmax), 0.0)
def ss_point_at_a(span_m: float, P_kN: float, a_m: float):
    Mmax = P_kN * a_m * (span_m - a_m) / span_m
    Vmax = P_kN
    return (0.0, float(Mmax), 0.0, float(Vmax), 0.0)
ready_catalog = {
    "Beam": {
        "Simply supported (examples)": {
            "SS-UDL": {"label": "SS-01: UDL (w on L)", "inputs": {"L": 6.0, "w": 10.0}, "func": ss_udl},
            "SS-Point-Centre": {"label": "SS-02: Point at midspan (P)", "inputs": {"L": 6.0, "P": 20.0}, "func": ss_point_center},
            "SS-Point-a": {"label": "SS-03: Point at distance a (P at a)", "inputs": {"L": 6.0, "P": 20.0, "a": 2.0}, "func": ss_point_at_a},
        }
    },
    "Frame": {
        "Simple frame examples": {
            "F-01": {"label": "FR-01: Simple 2-member frame (placeholder)", "inputs": {"L":4.0, "P":5.0}, "func": lambda L,P: (0.0, float(P*L/4.0), 0.0, float(P/2.0), 0.0)},
        }
    }
}

if use_ready:
    st.markdown("**Step 1 — choose object type (Beam or Frame)**")
    chosen_type = st.selectbox("Type", options=["-- choose --", "Beam", "Frame"], key="ready_type")
    if chosen_type and chosen_type != "-- choose --":
        categories = sorted(ready_catalog.get(chosen_type, {}).keys())
        if categories:
            chosen_cat = st.selectbox("Category", options=["-- choose --"] + categories, key="ready_category")
            if chosen_cat and chosen_cat != "-- choose --":
                cases_dict = ready_catalog[chosen_type][chosen_cat]
                st.markdown("**Choose a case (click the Select button below the case box)**")
                case_keys = list(cases_dict.keys())
                n_cols = 3
                cols = st.columns(n_cols)
                selected_case_key = None
                for i, ck in enumerate(case_keys):
                    col = cols[i % n_cols]
                    lbl = cases_dict[ck]["label"]
                    col.markdown(f"<div style='border:2px solid #bbb;border-radius:10px;padding:18px;text-align:center;background:#fbfbfb;margin-bottom:8px;min-height:84px;display:flex;align-items:center;justify-content:center;font-weight:600;'>{lbl}</div>", unsafe_allow_html=True)
                    if col.button(f"Select {ck}", key=f"ready_select_{chosen_type}_{chosen_cat}_{i}"):
                        selected_case_key = ck
                if selected_case_key:
                    st.session_state["ready_selected_type"] = chosen_type
                    st.session_state["ready_selected_category"] = chosen_cat
                    st.session_state["ready_selected_case"] = selected_case_key
                    st.rerun()
                sel_case = st.session_state.get("ready_selected_case")
                sel_type = st.session_state.get("ready_selected_type")
                sel_cat = st.session_state.get("ready_selected_category")
                if sel_case and sel_type == chosen_type and sel_cat == chosen_cat:
                    scase_info = ready_catalog[sel_type][sel_cat].get(sel_case)
                    if scase_info:
                        st.markdown(f"**Selected case:** {scase_info['label']}")
                        inputs = scase_info.get("inputs", {})
                        input_vals = {}
                        for k, v in inputs.items():
                            input_vals[k] = st.number_input(f"{k}", value=float(v), key=f"ready_input_{sel_case}_{k}")
                        col_apply, col_clear = st.columns([1,1])
                        with col_apply:
                            if st.button("Apply case to load inputs", key=f"ready_apply_{sel_case}"):
                                func = scase_info.get("func")
                                try:
                                    args = [input_vals[k] for k in inputs.keys()]
                                    N_case, My_case, Mz_case, Vy_case, Vz_case = func(*args)
                                except Exception:
                                    N_case, My_case, Mz_case, Vy_case, Vz_case = 0.0, 0.0, 0.0, 0.0, 0.0
                                st.session_state["prefill_from_case"] = True
                                st.session_state["prefill_N_kN"] = float(N_case)
                                st.session_state["prefill_My_kNm"] = float(My_case)
                                st.session_state["prefill_Mz_kNm"] = float(Mz_case)
                                st.session_state["prefill_Vy_kN"] = float(Vy_case)
                                st.session_state["prefill_Vz_kN"] = float(Vz_case)
                                if "L" in input_vals:
                                    st.session_state["case_L"] = float(input_vals["L"])
                                st.success("Case applied — scroll to Design forces and moments (inputs updated).")
                        with col_clear:
                            if st.button("Clear selected case", key=f"ready_clear_{sel_case}"):
                                for k in ("ready_selected_type","ready_selected_category","ready_selected_case","prefill_from_case","prefill_N_kN","prefill_My_kNm","prefill_Mz_kNm","prefill_Vy_kN","prefill_Vz_kN","case_L"):
                                    if k in st.session_state: del st.session_state[k]
                                st.success("Selected case cleared.")
                                st.rerun()
        else:
            st.info(f"No ready cases defined for {chosen_type}.")
    else:
        st.info("Select 'Beam' or 'Frame' to view categorized ready cases.")

# -------------------------
# Loads & inputs
# -------------------------
st.header("Design forces and moments (ultimate state) - INPUT")
st.markdown('<span title="Enter ultimate (ULS) design forces and moments. Positive N = compression.">ⓘ Load input help</span>', unsafe_allow_html=True)

r1c1, r1c2, r1c3 = st.columns(3)
with r1c1:
    L = st.number_input("Element length L (m)", value=6.0, min_value=0.0)
with r1c2:
    N_kN = st.number_input("Axial force N (kN) (positive = compression)", value=0.0)
with r1c3:
    Vy_kN = st.number_input("Shear V_y (kN)", value=0.0)
r2c1, r2c2, r2c3 = st.columns(3)
with r2c1:
    Vz_kN = st.number_input("Shear V_z (kN)", value=0.0)
with r2c2:
    My_kNm = st.number_input("Bending M_y (kN·m) (about y)", value=0.0)
with r2c3:
    Mz_kNm = st.number_input("Bending M_z (kN·m) (about z)", value=0.0)
r3c1, r3c2, r3c3 = st.columns(3)
with r3c1:
    Tx_kNm = st.number_input("Torsion T_x (kN·m)", value=0.0)
with r3c2:
    st.write("")
with r3c3:
    st.write("")
axis_check = "both (y & z)"
st.sidebar.header("Output mode")
output_mode = st.sidebar.radio("Results output mode", ("Concise (no formulas)", "Full (with formulas & steps)"))
want_pdf = st.sidebar.checkbox("Enable PDF download (requires reportlab)")

# -------------------------
# Use properties: if DB selected we already loaded sr_display above; else custom values
# -------------------------
if selected_row is not None and not use_custom:
    # sr_display contains the numeric values used for UI (we created it above)
    use_props = {
        "family": sr_display.get("family", "DB"),
        "name": sr_display.get("name", "DB"),
        "A_cm2": sr_display.get("A_cm2", 0.0),
        "S_y_cm3": sr_display.get("S_y_cm3", 0.0),
        "S_z_cm3": sr_display.get("S_z_cm3", 0.0),
        "I_y_cm4": sr_display.get("I_y_cm4", 0.0),
        "I_z_cm4": sr_display.get("I_z_cm4", 0.0),
        "J_cm4": sr_display.get("J_cm4", 0.0),
        "c_max_mm": sr_display.get("c_max_mm", 0.0),
        "Wpl_y_cm3": sr_display.get("Wpl_y_cm3", 0.0),
        "Wpl_z_cm3": sr_display.get("Wpl_z_cm3", 0.0),
        "alpha_curve": sr_display.get("alpha_curve", alpha_default_val),
        "flange_class_db": sr_display.get("flange_class_db", "n/a"),
        "web_class_bending_db": sr_display.get("web_class_bending_db", "n/a"),
        "web_class_compression_db": sr_display.get("web_class_compression_db", "n/a"),
        "Iw_cm6": sr_display.get("Iw_cm6", 0.0),
        "It_cm4": sr_display.get("It_cm4", 0.0),
        "J_cm4": sr_display.get("J_cm4", 0.0)
    }
else:
    use_props = {
        "type": "CUSTOM", "name": "CUSTOM",
        "A_cm2": locals().get("A_cm2", 50.0),
        "S_y_cm3": locals().get("S_y_cm3", 200.0),
        "S_z_cm3": locals().get("S_z_cm3", 50.0),
        "I_y_cm4": locals().get("I_y_cm4", 1500.0),
        "I_z_cm4": locals().get("I_z_cm4", 150.0),
        "J_cm4": locals().get("J_cm4", 10.0),
        "c_max_mm": locals().get("c_max_mm", 100.0),
        "Wpl_y_cm3": locals().get("Wpl_y_cm3", 0.0),
        "Wpl_z_cm3": locals().get("Wpl_z_cm3", 0.0),
        "alpha_curve": locals().get("alpha_custom", alpha_default_val),
        "bf_mm": 0.0, "tf_mm": 0.0, "hw_mm": 0.0, "tw_mm": 0.0,
        "flange_class_db": locals().get("flange_class_choice", "Auto (calc)"),
        "web_class_bending_db": locals().get("web_class_bending_choice", "Auto (calc)"),
        "web_class_compression_db": locals().get("web_class_compression_choice", "Auto (calc)"),
        "Iw_cm6": 0.0, "It_cm4": 0.0
    }

# -------------------------
# Unit conversions & derived values
# -------------------------
N_N = N_kN * 1e3
Vy_N = Vy_kN * 1e3
Vz_N = Vz_kN * 1e3
My_Nm = My_kNm * 1e3
Mz_Nm = Mz_kNm * 1e3
T_Nm = Tx_kNm * 1e3

A_m2 = use_props.get("A_cm2", 0.0) / 1e4
S_y_m3 = use_props.get("S_y_cm3", 0.0) * 1e-6
S_z_m3 = use_props.get("S_z_cm3", 0.0) * 1e-6
I_y_m4 = use_props.get("I_y_cm4", 0.0) * 1e-8
I_z_m4 = use_props.get("I_z_cm4", 0.0) * 1e-8
J_m4 = use_props.get("J_cm4", 0.0) * 1e-8
c_max_m = use_props.get("c_max_mm", 0.0) / 1000.0
Wpl_y_m3 = (use_props.get("Wpl_y_cm3", 0.0) * 1e-6) if use_props.get("Wpl_y_cm3", 0.0) > 0 else 1.1 * S_y_m3
Wpl_z_m3 = (use_props.get("Wpl_z_cm3", 0.0) * 1e-6) if use_props.get("Wpl_z_cm3", 0.0) > 0 else 1.1 * S_z_m3
alpha_curve_db = use_props.get("alpha_curve", alpha_default_val)
flange_class_db = use_props.get("flange_class_db", "Auto (calc)")
web_class_bending_db = use_props.get("web_class_bending_db", "Auto (calc)")
web_class_compression_db = use_props.get("web_class_compression_db", "Auto (calc)")

if A_m2 <= 0:
    st.error("Section area not provided (A <= 0). Select a section or use custom inputs with A > 0.")
    st.stop()

# -------------------------
# Calculations (unchanged core logic)
# -------------------------
N_Rd_N = A_m2 * fy * 1e6 / gamma_M0
T_Rd_N = N_Rd_N
M_Rd_y_Nm = Wpl_y_m3 * fy * 1e6 / gamma_M0
Av_m2 = 0.6 * A_m2
V_Rd_N = Av_m2 * fy * 1e6 / (math.sqrt(3) * gamma_M0)

sigma_axial_Pa = N_N / A_m2
sigma_by_Pa = My_Nm / S_y_m3 if S_y_m3 > 0 else 0.0
sigma_bz_Pa = Mz_Nm / S_z_m3 if S_z_m3 > 0 else 0.0
sigma_total_Pa = sigma_axial_Pa + sigma_by_Pa + sigma_bz_Pa
sigma_total_MPa = sigma_total_Pa / 1e6

tau_y_Pa = Vz_N / Av_m2 if Av_m2 > 0 else 0.0
tau_z_Pa = Vy_N / Av_m2 if Av_m2 > 0 else 0.0
tau_torsion_Pa = 0.0
if J_m4 > 0 and c_max_m > 0:
    tau_torsion_Pa = T_Nm * c_max_m / J_m4
tau_total_Pa = math.sqrt(tau_y_Pa**2 + tau_z_Pa**2 + tau_torsion_Pa**2)
tau_total_MPa = tau_total_Pa / 1e6
sigma_eq_MPa = math.sqrt((abs(sigma_total_MPa))**2 + 3.0 * (tau_total_MPa**2))

tau_allow_Pa = 0.6 * sigma_allow_MPa * 1e6

util_axial = abs(N_N) / N_Rd_N if N_Rd_N > 0 else None
util_ten = abs(min(N_N, 0.0)) / T_Rd_N if T_Rd_N > 0 else None
util_My = abs(My_Nm) / M_Rd_y_Nm if M_Rd_y_Nm > 0 else None
util_shear_resultant = math.sqrt(Vy_N**2 + Vz_N**2) / V_Rd_N if V_Rd_N > 0 else None
util_torsion = (tau_torsion_Pa / tau_allow_Pa) if tau_allow_Pa > 0 else None

# Buckling simplified
E = 210e9
buck_results = []
I_check_list = []
if axis_check in ("both (y & z)", "y (weak)"):
    I_check_list.append(("y", I_y_m4, K_y))
if axis_check in ("both (y & z)", "z (strong)"):
    I_check_list.append(("z", I_z_m4, K_z))

for axis_label, I_axis, K_axis in I_check_list:
    if I_axis is None or I_axis <= 0:
        buck_results.append((axis_label, None, None, None, None, "No I"))
        continue
    Leff_axis = K_axis * L
    Ncr = (math.pi**2 * E * I_axis) / (Leff_axis**2)
    lambda_bar = math.sqrt((A_m2 * fy * 1e6) / Ncr) if Ncr > 0 else float('inf')
    alpha_use = alpha_curve_db if alpha_curve_db is not None else alpha_default_val
    phi = 0.5 * (1.0 + alpha_use * (lambda_bar**2))
    sqrt_term = max(phi**2 - lambda_bar**2, 0.0)
    chi = 1.0 / (phi + math.sqrt(sqrt_term)) if (phi + math.sqrt(sqrt_term)) > 0 else 0.0
    N_b_Rd_N = chi * A_m2 * fy * 1e6 / gamma_M1
    status = "OK" if abs(N_N) <= N_b_Rd_N else "EXCEEDS"
    buck_results.append((axis_label, Ncr, lambda_bar, chi, N_b_Rd_N, status))

N_b_Rd_candidates = [r[4] for r in buck_results if r[4] not in (None,)]
N_b_Rd_min_N = min(N_b_Rd_candidates) if N_b_Rd_candidates else None
compression_resistance_N = N_b_Rd_min_N if N_b_Rd_min_N is not None else N_Rd_N

# -------------------------
# Build checks rows (same logic as original)
# -------------------------
def status_and_util(applied, resistance):
    if resistance is None or resistance == 0:
        return ("n/a", None)
    util = abs(applied) / resistance
    return ("OK" if util <= 1.0 else "EXCEEDS", util)

rows = []
# Compression
applied_N = N_N if N_N >= 0 else 0.0
res_comp_N = compression_resistance_N
status_comp, util_comp = status_and_util(applied_N, res_comp_N)
rows.append({"Check":"Compression (N≥0)","Applied":f"{applied_N/1e3:.3f} kN","Resistance":(f"{res_comp_N/1e3:.3f} kN" if res_comp_N else "n/a"),"Utilization":(f"{util_comp:.3f}" if util_comp else "n/a"),"Status":status_comp})
# Tension
applied_tension_N = -N_N if N_N < 0 else 0.0
res_tension_N = T_Rd_N
status_ten, util_ten = status_and_util(applied_tension_N, res_tension_N)
rows.append({"Check":"Tension (N<0)","Applied":f"{applied_tension_N/1e3:.3f} kN","Resistance":f"{res_tension_N/1e3:.3f} kN","Utilization":(f"{util_ten:.3f}" if util_ten else "n/a"),"Status":status_ten})
# Shear
applied_shear_N = math.sqrt(Vy_N**2 + Vz_N**2)
res_shear_N = V_Rd_N
status_shear, util_shear_val = status_and_util(applied_shear_N, res_shear_N)
rows.append({"Check":"Shear (resultant Vy & Vz)","Applied":f"{applied_shear_N/1e3:.3f} kN","Resistance":f"{res_shear_N/1e3:.3f} kN","Utilization":(f"{util_shear_val:.3f}" if util_shear_val else "n/a"),"Status":status_shear})
# Torsion
applied_tau_Pa = tau_torsion_Pa
res_tau_allow_Pa = tau_allow_Pa
util_torsion = applied_tau_Pa / res_tau_allow_Pa if res_tau_allow_Pa>0 else None
status_tors = "OK" if util_torsion is not None and util_torsion <= 1.0 else ("EXCEEDS" if util_torsion is not None else "n/a")
rows.append({"Check":"Torsion (τ = T·c/J)","Applied":(f"{applied_tau_Pa/1e6:.6f} MPa" if isinstance(applied_tau_Pa, (int,float)) else "n/a"),"Resistance":f"{res_tau_allow_Pa/1e6:.6f} MPa (approx)","Utilization":(f"{util_torsion:.3f}" if util_torsion else "n/a"),"Status":status_tors})

rows.append({"Check":"Bending y-y (σ_by)","Applied":f"{sigma_by_Pa/1e6:.3f} MPa","Resistance":f"{sigma_allow_MPa:.3f} MPa","Utilization":(f"{(abs(sigma_by_Pa/1e6)/sigma_allow_MPa):.3f}" if sigma_allow_MPa>0 else "n/a"),"Status":"OK" if abs(sigma_by_Pa/1e6)/sigma_allow_MPa<=1.0 else "EXCEEDS"})
rows.append({"Check":"Bending z-z (σ_bz)","Applied":f"{sigma_bz_Pa/1e6:.3f} MPa","Resistance":f"{sigma_allow_MPa:.3f} MPa","Utilization":(f"{(abs(sigma_bz_Pa/1e6)/sigma_allow_MPa):.3f}" if sigma_allow_MPa>0 else "n/a"),"Status":"OK" if abs(sigma_bz_Pa/1e6)/sigma_allow_MPa<=1.0 else "EXCEEDS"})
rows.append({"Check":"Bending y-y & shear (indicative)","Applied":f"σ_by={sigma_by_Pa/1e6:.3f} MPa, τ_eq={tau_total_MPa:.3f} MPa","Resistance":f"{sigma_allow_MPa:.3f} MPa","Utilization":(f"{(sigma_eq_MPa/sigma_allow_MPa):.3f}" if sigma_allow_MPa>0 else "n/a"),"Status":"OK" if sigma_eq_MPa/sigma_allow_MPa<=1.0 else "EXCEEDS"})
rows.append({"Check":"Bending z-z & shear (indicative)","Applied":f"σ_bz={sigma_bz_Pa/1e6:.3f} MPa, τ_eq={tau_total_MPa:.3f} MPa","Resistance":f"{sigma_allow_MPa:.3f} MPa","Utilization":(f"{(sigma_eq_MPa/sigma_allow_MPa):.3f}" if sigma_allow_MPa>0 else "n/a"),"Status":"OK" if sigma_eq_MPa/sigma_allow_MPa<=1.0 else "EXCEEDS"})
rows.append({"Check":"Bending y-y & axial (σ_by+σ_axial)","Applied":f"{(sigma_by_Pa + sigma_axial_Pa)/1e6:.3f} MPa","Resistance":f"{sigma_allow_MPa:.3f} MPa","Utilization":(f"{(abs(sigma_by_Pa + sigma_axial_Pa)/1e6/sigma_allow_MPa):.3f}" if sigma_allow_MPa>0 else "n/a"),"Status":"OK" if abs(sigma_by_Pa + sigma_axial_Pa)/1e6/sigma_allow_MPa<=1.0 else "EXCEEDS"})
rows.append({"Check":"Bending z-z & axial (σ_bz+σ_axial)","Applied":f"{(sigma_bz_Pa + sigma_axial_Pa)/1e6:.3f} MPa","Resistance":f"{sigma_allow_MPa:.3f} MPa","Utilization":(f"{(abs(sigma_bz_Pa + sigma_axial_Pa)/1e6/sigma_allow_MPa):.3f}" if sigma_allow_MPa>0 else "n/a"),"Status":"OK" if abs(sigma_bz_Pa + sigma_axial_Pa)/1e6/sigma_allow_MPa<=1.0 else "EXCEEDS"})
rows.append({"Check":"Biaxial bending & axial (indicative)","Applied":f"{(abs(sigma_by_Pa/1e6)+abs(sigma_bz_Pa/1e6)+abs(sigma_axial_Pa/1e6)):.3f} MPa","Resistance":f"{sigma_allow_MPa:.3f} MPa","Utilization":(f"{((abs(sigma_by_Pa/1e6)+abs(sigma_bz_Pa/1e6)+abs(sigma_axial_Pa/1e6))/sigma_allow_MPa):.3f}" if sigma_allow_MPa>0 else "n/a"),"Status":"OK" if ((abs(sigma_by_Pa/1e6)+abs(sigma_bz_Pa/1e6)+abs(sigma_axial_Pa/1e6))/sigma_allow_MPa)<=1.0 else "EXCEEDS"})
rows.append({"Check":"Bending y-y & axial & shear (indicative)","Applied":f"σ_by={sigma_by_Pa/1e6:.3f} MPa, τ={tau_total_MPa:.3f} MPa","Resistance":f"{sigma_allow_MPa:.3f} MPa","Utilization":(f"{(sigma_eq_MPa/sigma_allow_MPa):.3f}" if sigma_allow_MPa>0 else "n/a"),"Status":"OK" if sigma_eq_MPa/sigma_allow_MPa<=1.0 else "EXCEEDS"})
rows.append({"Check":"Bending z-z & axial & shear (indicative)","Applied":f"σ_bz={sigma_bz_Pa/1e6:.3f} MPa, τ={tau_total_MPa:.3f} MPa","Resistance":f"{sigma_allow_MPa:.3f} MPa","Utilization":(f"{(sigma_eq_MPa/sigma_allow_MPa):.3f}" if sigma_allow_MPa>0 else "n/a"),"Status":"OK" if sigma_eq_MPa/sigma_allow_MPa<=1.0 else "EXCEEDS"})
rows.append({"Check":"Biaxial bending & axial & shear (indicative)","Applied":f"{(abs(sigma_by_Pa/1e6)+abs(sigma_bz_Pa/1e6)):.3f} MPa, τ={tau_total_MPa:.3f} MPa","Resistance":f"{sigma_allow_MPa:.3f} MPa","Utilization":(f"{(sigma_eq_MPa/sigma_allow_MPa):.3f}" if sigma_allow_MPa>0 else "n/a"),"Status":"OK" if sigma_eq_MPa/sigma_allow_MPa<=1.0 else "EXCEEDS"})

for axis_label, Ncr, lambda_bar, chi, N_b_Rd_N, status in buck_results:
    if N_b_Rd_N:
        util_buck = abs(N_N) / N_b_Rd_N
        rows.append({"Check":f"Flexural buckling {axis_label}","Applied":f"{abs(N_N)/1e3:.3f} kN","Resistance":f"{N_b_Rd_N/1e3:.3f} kN","Utilization":f"{util_buck:.3f}","Status":"OK" if util_buck<=1.0 else "EXCEEDS"})
    else:
        rows.append({"Check":f"Flexural buckling {axis_label}","Applied":"n/a","Resistance":"n/a","Utilization":"n/a","Status":"n/a"})

rows.append({"Check":"Torsional & torsional-flexural (approx)","Applied":(f"{util_torsion:.3f}" if util_torsion else "n/a"),"Resistance":"n/a","Utilization":(f"{util_torsion:.3f}" if util_torsion else "n/a"),"Status":"OK" if util_torsion and util_torsion<=1.0 else ("EXCEEDS" if util_torsion else "n/a")})

Iw_cm6 = use_props.get("Iw_cm6", 0.0)
It_cm4 = use_props.get("It_cm4", 0.0)
Iw_m6 = Iw_cm6 * 1e-12 if Iw_cm6 else 0.0
It_m4 = It_cm4 * 1e-8 if It_cm4 else 0.0
Mcr = None; chi_LT = None; M_Rd_LT = None
if Iw_m6 and Iw_m6>0:
    try:
        Leff_LT = K_LT * L
        term = (81e9 * It_m4 * (Leff_LT**2)) / (math.pi**2 * 210e9 * Iw_m6) if (Iw_m6>0) else 0.0
        Mcr = (math.pi**2 * 210e9 * Iw_m6) / (Leff_LT**2) * math.sqrt(1.0 + term)
        Mpl = Wpl_y_m3 * fy * 1e6
        lambda_LT = math.sqrt(Mpl / Mcr) if Mcr>0 else float('inf')
        alpha_LT = 0.49
        phi_LT = 0.5*(1.0 + alpha_LT * lambda_LT**2)
        sqrt_term_LT = max(phi_LT**2 - lambda_LT**2, 0.0)
        chi_LT = 1.0 / (phi_LT + math.sqrt(sqrt_term_LT)) if (phi_LT + math.sqrt(sqrt_term_LT))>0 else 0.0
        M_Rd_LT = chi_LT * Mpl / gamma_M0
    except Exception:
        Mcr = None

if M_Rd_LT and M_Rd_LT>0:
    util_LT = abs(My_Nm) / M_Rd_LT
    rows.append({"Check":"Lateral-torsional buckling (LT)","Applied":f"{abs(My_Nm)/1e3:.3f} kN·m","Resistance":f"{M_Rd_LT/1e3:.3f} kN·m","Utilization":f"{util_LT:.3f}","Status":"OK" if util_LT<=1.0 else "EXCEEDS"})
else:
    rows.append({"Check":"Lateral-torsional buckling (LT)","Applied":"n/a","Resistance":"n/a","Utilization":"n/a","Status":"n/a"})

interaction_ratio_simple = (abs(N_N)/N_Rd_N) + (abs(My_Nm)/M_Rd_y_Nm) if N_Rd_N>0 and M_Rd_y_Nm>0 else None
interaction_method2 = math.sqrt((abs(N_N)/N_Rd_N)**2 + (abs(My_Nm)/M_Rd_y_Nm)**2) if N_Rd_N>0 and M_Rd_y_Nm>0 else None
if interaction_ratio_simple is not None:
    rows.append({"Check":"Combined bending & compression - Eq6.61 method1 (approx)","Applied":f"{interaction_ratio_simple:.3f}","Resistance":"1.00","Utilization":f"{interaction_ratio_simple:.3f}","Status":"OK" if interaction_ratio_simple<=1.0 else "EXCEEDS"})
    rows.append({"Check":"Combined bending & compression - Eq6.62 method1 (approx)","Applied":f"{interaction_ratio_simple:.3f}","Resistance":"1.00","Utilization":f"{interaction_ratio_simple:.3f}","Status":"OK" if interaction_ratio_simple<=1.0 else "EXCEEDS"})
if interaction_method2 is not None:
    rows.append({"Check":"Combined bending & compression - Eq6.61 method2 (approx)","Applied":f"{interaction_method2:.3f}","Resistance":"1.00","Utilization":f"{interaction_method2:.3f}","Status":"OK" if interaction_method2<=1.0 else "EXCEEDS"})
    rows.append({"Check":"Combined bending & compression - Eq6.62 method2 (approx)","Applied":f"{interaction_method2:.3f}","Resistance":"1.00","Utilization":f"{interaction_method2:.3f}","Status":"OK" if interaction_method2<=1.0 else "EXCEEDS"})
else:
    rows.append({"Check":"Combined bending & compression (interaction)","Applied":"n/a","Resistance":"n/a","Utilization":"n/a","Status":"n/a"})

# -------------------------
# Display results table
# -------------------------
st.markdown("---")
df_rows = pd.DataFrame(rows).set_index("Check")
overall_ok = not any(df_rows["Status"] == "EXCEEDS")
if overall_ok:
    st.success("Result summary: DESIGN OK — no checks exceed capacity (preliminary).")
else:
    st.error("Result summary: DESIGN NOT OK — one or more checks exceed capacity (preliminary).")

def highlight_row(row):
    s = row["Status"]
    if s == "OK":
        color = "background-color: #e6f7e6"
    elif s == "EXCEEDS":
        color = "background-color: #fde6e6"
    else:
        color = "background-color: #f0f0f0"
    return [color] * len(row)

styled = df_rows.style.apply(highlight_row, axis=1)
st.subheader("Cross-section & member checks (detailed)")
st.write("Legend: OK — within capacity; EXCEEDS — capacity exceeded; n/a — not applicable/missing data.")
st.write(styled)

# -------------------------
# Save results (session)
# -------------------------
if "saved_results" not in st.session_state:
    st.session_state["saved_results"] = []

def build_result_record():
    rec = {
        "timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "doc_title": doc_name,
        "project_name": project_name,
        "position": position,
        "requested_by": requested_by,
        "revision": revision,
        "date": run_date.isoformat(),
        "section_type": use_props.get("family", use_props.get("type", "CUSTOM")),
        "section_name": use_props.get("name"),
        "A_cm2": use_props.get("A_cm2"),
        "S_y_cm3": use_props.get("S_y_cm3"),
        "L_m": L,
        "N_kN": N_kN,
        "My_kNm": My_kNm,
        "Vz_kN": Vz_kN,
        "N_Rd_kN": N_Rd_N/1e3,
        "M_Rd_y_kNm": M_Rd_y_Nm/1e3,
        "V_Rd_kN": V_Rd_N/1e3,
        "overall_ok": overall_ok
    }
    return rec

svc1, svc2 = st.columns([1,1])
with svc1:
    if st.button("Save results", help="Save this run to session (temporarily)."):
        st.session_state["saved_results"].append(build_result_record())
        st.success("Results saved to session state.")
with svc2:
    st.info("Saved runs are kept in this browser session. Use admin page to export DB later.")

# Full mode expanders (optional)
if output_mode.startswith("Full"):
    st.markdown("---")
    st.subheader("Full formulas & intermediate steps (click expanders to view)")
    with st.expander("Tension — formula & details (EN1993-1-1 §6.2.3)"):
        st.latex(r"N_{Rd} = \dfrac{A \cdot f_y}{\gamma_{M0}}")
        st.write(f"A = {A_m2:.6e} m², fy = {fy:.1f} MPa, γ_M0 = {gamma_M0:.2f}")
        st.write(f"N_Rd = {N_Rd_N/1e3:.3f} kN")
    with st.expander("Compression & buckling — formula & details (EN1993-1-1 §6.2.4 & 6.3)"):
        st.latex(r"N_{cr} = \dfrac{\pi^2 E I}{(K L)^2}")
        for axis_label, Ncr, lambda_bar, chi, N_b_Rd_N, status in buck_results:
            Ncr_str = f"{Ncr/1e3:.2f} kN" if (Ncr is not None and isinstance(Ncr, (int,float)) and math.isfinite(Ncr)) else "n/a"
            lambda_str = f"{lambda_bar:.3f}" if (lambda_bar is not None and isinstance(lambda_bar, (int,float)) and math.isfinite(lambda_bar)) else "n/a"
            chi_str = f"{chi:.3f}" if (chi is not None and isinstance(chi, (int,float)) and math.isfinite(chi)) else "n/a"
            Nbrd_str = f"{N_b_Rd_N/1e3:.2f} kN" if (N_b_Rd_N is not None and isinstance(N_b_Rd_N, (int,float)) and math.isfinite(N_b_Rd_N)) else "n/a"
            st.write(f"Axis {axis_label}: N_cr = {Ncr_str}, λ̄ = {lambda_str}, χ = {chi_str}, N_b,Rd = {Nbrd_str}, status={status}")
    with st.expander("Bending — formula & details (EN1993-1-1 §6.2.5)"):
        st.latex(r"\sigma = \dfrac{M}{W}")
        st.write(f"M_y = {My_Nm/1e3:.3f} kN·m, W_pl,y = {Wpl_y_m3*1e6 if Wpl_y_m3 else 'n/a'} mm³")
        st.write(f"σ_by = {sigma_by_Pa/1e6:.3f} MPa; utilization = {abs(sigma_by_Pa/1e6)/sigma_allow_MPa if sigma_allow_MPa>0 else 'n/a'}")
    with st.expander("Shear — formula & details (EN1993-1-1 §6.2.6)"):
        st.latex(r"\tau = \dfrac{V}{A_v},\quad V_{Rd} = \dfrac{A_v f_y}{\sqrt{3}\gamma_{M0}}")
        st.write(f"A_v (used) = {Av_m2:.6e} m² (Av factor = 0.6)")
        st.write(f"V_resultant = {applied_shear_N/1e3:.3f} kN; V_Rd = {V_Rd_N/1e3:.3f} kN")
    with st.expander("Torsion — formula & details (approx)"):
        st.latex(r"\tau_t = \dfrac{T \cdot c}{J} \quad\text{(approx for closed shapes)}")
        st.write(f"T = {T_Nm:.3f} N·m, c = {c_max_m:.3f} m, J = {J_m4:.6e} m⁴")
        st.write(f"τ_t (max) = {tau_torsion_Pa/1e6:.6f} MPa, allowable (approx) = {tau_allow_Pa/1e6:.3f} MPa")
    with st.expander("Lateral–torsional buckling (LT) — formula & details (EN1993-1-1 §6.3.2)"):
        st.latex(r"M_{cr} \,\text{(approx with warping)}")
        Mcr_str = f"{Mcr:.3f}" if (Mcr is not None and isinstance(Mcr, (int,float)) and math.isfinite(Mcr)) else "n/a"
        chi_LT_str = f"{chi_LT:.3f}" if (chi_LT is not None and isinstance(chi_LT, (int,float)) and math.isfinite(chi_LT)) else "n/a"
        M_Rd_LT_str = f"{M_Rd_LT/1e3:.3f} kN·m" if (M_Rd_LT is not None and isinstance(M_Rd_LT, (int,float)) and math.isfinite(M_Rd_LT)) else "n/a"
        st.write(f"K_LT used = {K_LT:.3f}; M_cr = {Mcr_str}, χ_LT = {chi_LT_str}, M_Rd_LT = {M_Rd_LT_str}")

# -------------------------
# End
# -------------------------
st.markdown("---")
st.subheader("Notes, limitations & references")
st.write("""
- This tool gives **preliminary** screening checks only. It is not a complete EN1993 implementation.
- For standard (DB) sections, classification and buckling curve α are taken from your DB and shown read-only in the UI.
- If your DB uses different column names, use the raw DB row debug expander to copy exact column names and I can add them to the auto-detection mapping.
- For stable ordering of sections, consider adding an explicit 'position' column when importing to DB and ordering by it (ctid is used as a practical fallback).
- Reference: EN1993-1-1 (Design of steel structures).
""")
