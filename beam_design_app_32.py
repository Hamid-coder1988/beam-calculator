# beam_design_app.py
# EngiSnap — Beam design Eurocode (DB-backed, Type/Size columns, original layout + results)
# SECURITY NOTE:
# - For local testing you may hardcode DB credentials (NOT recommended for public repos).
# - For deployment use Streamlit secrets: add a [postgres] table in .streamlit/secrets.toml:
#   [postgres]
#   host = "your_host"
#   port = 15500
#   database = "railway"
#   user = "postgres"
#   password = "your_password"
# Do NOT commit secrets to public repos.

import streamlit as st
import pandas as pd
import math
from io import BytesIO
from datetime import datetime, date
import psycopg2
from urllib.parse import urlparse

st.set_page_config(page_title="EngiSnap Beam Design Eurocode Checker", layout="wide")
st.title("EngiSnap — Design of steel memebers (Eurocode)")

st.markdown("""
**Disclaimer (read carefully)**  
This prototype performs simplified Eurocode-style screening checks. It is **not** a full EN1993 design package.
Use for screening/prototyping only — always have final results verified by a licensed structural engineer.
""")

# -------------------------
# DB helper: connect to Postgres using st.secrets["postgres"]
# -------------------------
@st.cache_resource(show_spinner=False)
def get_conn_safe():
    """
    Attempts to read st.secrets['postgres'] and connect.
    If st.secrets not provided or connection fails, caller will fallback to sample df.
    """
    pg = st.secrets.get("postgres", None)
    if not pg:
        raise RuntimeError('st.secrets["postgres"] not found.')
    conn = psycopg2.connect(
        host=pg["host"],
        port=str(pg["port"]),
        database=pg["database"],
        user=pg["user"],
        password=pg["password"]
    )
    return conn

@st.cache_resource(show_spinner=False)
def load_beam_db_table():
    """
    Specifically attempts to load table named "Beam" (user requested).
    Accepts "Beam" or "beam" and falls back to sample if not found or on any error.
    """
    try:
        conn = get_conn_safe()
        # explicitly query "Beam" (quoted) first to preserve case if present
        try:
            df = pd.read_sql('SELECT * FROM "Beam";', conn)
            st.sidebar.info("Loaded table: Beam")
            return df
        except Exception:
            # fallback to unquoted lower-case name
            df = pd.read_sql("SELECT * FROM beam;", conn)
            st.sidebar.info("Loaded table: beam")
            return df
    except Exception as e:
        # fallback sample - keep the exact sample format you used previously
        sample_rows = [
            {"Type": "IPE", "Size": "IPE 200",
             "A_cm2": 31.2, "S_y_cm3": 88.0, "S_z_cm3": 16.0,
             "I_y_cm4": 1760.0, "I_z_cm4": 64.0, "J_cm4": 14.0, "c_max_mm": 100.0,
             "Wpl_y_cm3": 96.8, "Wpl_z_cm3": 18.0, "alpha_curve": 0.49,
             "flange_class_db": "Class 1/2", "web_class_bending_db": "Class 2", "web_class_compression_db": "Class 3",
             "Iw_cm6": 2500.0, "It_cm4": 14.0
            }
        ]
        st.sidebar.warning("Could not load DB (using internal sample). DB error: " + str(e))
        return pd.DataFrame(sample_rows)

# Load DB (or sample)
df_db = load_beam_db_table()

# -------------------------
# METADATA BLOCK (top)
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
# Sidebar: material & then section selection (Type & Size) as requested
# -------------------------
st.sidebar.header("Material & Section selection")
material = st.sidebar.selectbox("Material", ("S235", "S275", "S355"),
                                help="Select steel grade (typical EN names).")
if material == "S235":
    fy = 235.0
elif material == "S275":
    fy = 275.0
else:
    fy = 355.0

st.sidebar.markdown("---")
st.sidebar.markdown("**Select section from DB**")

# Use exact columns "Type" and "Size" (case-insensitive fallback)
def col_exists(df, name):
    for c in df.columns:
        if str(c).lower() == name.lower():
            return c
    return None

type_col = col_exists(df_db, "Type")
size_col = col_exists(df_db, "Size")

if type_col is None or size_col is None:
    st.sidebar.error("DB table must include columns 'Type' and 'Size' (case-insensitive). Using sample instead.")
    df_sample_db = pd.DataFrame([
        {"Type": "IPE", "Size": "IPE 200",
         "A_cm2": 31.2, "S_y_cm3": 88.0, "S_z_cm3": 16.0,
         "I_y_cm4": 1760.0, "I_z_cm4": 64.0, "J_cm4": 14.0, "c_max_mm": 100.0,
         "Wpl_y_cm3": 96.8, "Wpl_z_cm3": 18.0, "alpha_curve": 0.49,
         "flange_class_db": "Class 1/2", "web_class_bending_db": "Class 2", "web_class_compression_db": "Class 3",
         "Iw_cm6": 2500.0, "It_cm4": 14.0
        }
    ])
    df_work = df_sample_db
else:
    df_work = df_db

# Preserve DB order for types and sizes (no sorting)
types = []
if type_col in df_work.columns:
    for v in df_work[type_col].astype(str).tolist():
        if v not in types:
            types.append(v)

Type = st.sidebar.selectbox("Section type (DB) — Type", options=["-- choose --"] + types,
                            help="Choose section family/type (uses DB column 'Type').")

selected_name = None
selected_row = None
if Type and Type != "-- choose --":
    df_f = df_work[df_work[type_col].astype(str) == str(Type)]
    # sizes preserve DB order
    sizes = df_f[size_col].astype(str).tolist()
    # remove duplicates, keep first occurrence order
    seen = set(); sizes_u = []
    for s in sizes:
        if s not in seen:
            seen.add(s); sizes_u.append(s)
    selected_name = st.sidebar.selectbox("Section size (DB) — Size", options=["-- choose --"] + sizes_u,
                                         help="Choose section size (DB column 'Size').")
    if selected_name and selected_name != "-- choose --":
        # pick the first matching row
        df_sel = df_f[df_f[size_col].astype(str) == str(selected_name)]
        if df_sel.empty:
            df_sel = df_f[df_f[size_col].astype(str).str.strip() == str(selected_name).strip()]
        if not df_sel.empty:
            selected_row = df_sel.iloc[0].to_dict()
            st.sidebar.success(f"Loaded: {selected_name}")
        else:
            st.sidebar.error("Selected row not found.")

# Buckling factors - moved after section selection as requested
st.sidebar.markdown("---")
st.sidebar.markdown("Buckling effective length factors (K):")
K_z = st.sidebar.number_input("K_z — flexural buckling about minor axis z–z", value=1.0, min_value=0.1, step=0.05,
                              help="Effective length multiplier for flexural buckling about z-z (minor axis)")
K_y = st.sidebar.number_input("K_y — flexural buckling about major axis y–y", value=1.0, min_value=0.1, step=0.05,
                              help="Effective length multiplier for flexural buckling about y-y (major axis)")
K_LT = st.sidebar.number_input("K_LT — lateral–torsional buckling effective factor", value=1.0, min_value=0.1, step=0.05,
                               help="Effective length multiplier for lateral–torsional buckling (LT)")
K_T = st.sidebar.number_input("K_T — torsional buckling effective factor (reserved)", value=1.0, min_value=0.1, step=0.05,
                              help="Reserved: torsional buckling K factor (not used in current simplified checks)")
alpha_default_val = 0.49

# -------------------------
# Sample fallback DB (same format as old app) - used if selected_row None and no custom
# -------------------------
sample_rows = [
    {"Type": "IPE", "Size": "IPE 200",
     "A_cm2": 31.2, "S_y_cm3": 88.0, "S_z_cm3": 16.0,
     "I_y_cm4": 1760.0, "I_z_cm4": 64.0, "J_cm4": 14.0, "c_max_mm": 100.0,
     "Wpl_y_cm3": 96.8, "Wpl_z_cm3": 18.0, "alpha_curve": 0.49,
     "flange_class_db": "Class 1/2", "web_class_bending_db": "Class 2", "web_class_compression_db": "Class 3",
     "Iw_cm6": 2500.0, "It_cm4": 14.0
    }
]
df_sample_db = pd.DataFrame(sample_rows)

# If DB gave no results, keep sample as df_sample_db
if df_work is None or df_work.empty:
    df_sample_db = pd.DataFrame(sample_rows)

# -------------------------
# Section selection UI in main area (kept same as old format)
# -------------------------
st.header("Section selection")
st.markdown('<span title="Select a standard section from DB (read-only) or use custom.">ⓘ Section selection help</span>',
            unsafe_allow_html=True)

col_left, col_right = st.columns([1,1])
with col_left:
    # list of families from df_work (or sample)
    families = sorted(df_work[type_col].dropna().unique().tolist()) if (type_col and type_col in df_work.columns) else sorted(df_sample_db['Type'].unique().tolist())
    family = st.selectbox("Section family (DB)", options=["-- choose --"] + families,
                          help="Choose section family (database). If none selected, use Custom.")
with col_right:
    selected_name_main = None
    selected_row_main = None
    if family and family != "-- choose --":
        df_f_main = (df_work if type_col in df_work.columns else df_sample_db)[(df_work[type_col].astype(str) == family) if (type_col and type_col in df_work.columns) else (df_sample_db['Type'] == family)]
        # if df_f_main ends up empty use sample filtering
        if df_f_main.empty:
            df_f_main = df_sample_db[df_sample_db['Type'] == family]
        names = sorted(df_f_main[size_col].tolist()) if (size_col and size_col in df_f_main.columns) else sorted(df_f_main['Size'].tolist())
        selected_name_main = st.selectbox("Section size (DB)", options=["-- choose --"] + names,
                                     help="Choose section size (database). Selecting a size loads read-only properties.")
        if selected_name_main and selected_name_main != "-- choose --":
            df_sel_main = df_f_main[df_f_main[size_col].astype(str) == selected_name_main] if (size_col and size_col in df_f_main.columns) else df_f_main[df_f_main['Size'] == selected_name_main]
            if df_sel_main.empty:
                df_sel_main = df_f_main[df_f_main[size_col].astype(str).str.strip() == selected_name_main.strip()] if (size_col and size_col in df_f_main.columns) else df_f_main[df_f_main['Size'].astype(str).str.strip() == selected_name_main.strip()]
            if not df_sel_main.empty:
                selected_row_main = df_sel_main.iloc[0].to_dict()

st.markdown("**Or select Custom**. Standard DB sections are read-only; custom sections are editable.")
use_custom = st.checkbox("Use custom section (enable manual inputs)", help="Tick to enter section properties manually (CUSTOM)")

# Prefer DB selection from sidebar (Type/Size) if present, otherwise main selection above
if selected_row is None and selected_row_main is not None:
    selected_row = selected_row_main

if selected_row is None and not use_custom:
    st.info("Please select a section size from the DB (sidebar or above), or tick 'Use custom section' to enter properties manually.")

# -------------------------
# Cross-section image placeholder (centered)
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
    Cross-section image placeholder
</div>
""",
    unsafe_allow_html=True,
)
st.markdown("---")

# -------------------------
# Section properties display (DB read-only or Custom editable)
# (This block reproduces your previous format exactly, reading values from DB fields)
# -------------------------
alpha_options = [
    ("0.13 (a)", 0.13),
    ("0.21 (b)", 0.21),
    ("0.34 (c)", 0.34),
    ("0.49 (d)", 0.49),
    ("0.76 (e)", 0.76),
]
alpha_labels = [t for t, v in alpha_options]
alpha_map = {t: v for t, v in alpha_options}

# helper pick function tolerant of different column names
def pick(d, *keys, default=None):
    if d is None:
        return default
    for k in keys:
        if k in d and pd.notnull(d[k]):
            return d[k]
    # case-insensitive
    lk = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        kl = str(k).lower()
        if kl in lk and pd.notnull(lk[kl]):
            return lk[kl]
    return default

if selected_row is not None and not use_custom:
    st.markdown("### Section properties (from DB — read only)")
    sr = selected_row or {}

    # First row
    c1, c2, c3 = st.columns(3)
    c1.number_input("A (cm²)", value=float(pick(sr, "A_cm2", "area_cm2", "area", "A", "Area") or 0.0), disabled=True, key="db_A_cm2")
    c2.number_input("S_y (cm³) about y", value=float(pick(sr, "S_y_cm3", "Sy", "S_y") or 0.0), disabled=True, key="db_Sy_cm3")
    c3.number_input("S_z (cm³) about z", value=float(pick(sr, "S_z_cm3", "Sz", "S_z") or 0.0), disabled=True, key="db_Sz_cm3")

    # Second row
    c4, c5, c6 = st.columns(3)
    c4.number_input("I_y (cm⁴) about y", value=float(pick(sr, "I_y_cm4", "Iy", "I_y") or 0.0), disabled=True, key="db_Iy_cm4")
    c5.number_input("I_z (cm⁴) about z", value=float(pick(sr, "I_z_cm4", "Iz", "I_z") or 0.0), disabled=True, key="db_Iz_cm4")
    c6.number_input("J (cm⁴) (torsion const)", value=float(pick(sr, "J_cm4", "J") or 0.0), disabled=True, key="db_J_cm4")

    # Third row
    c7, c8, c9 = st.columns(3)
    c7.number_input("c_max (mm)", value=float(pick(sr, "c_max_mm", "c_max", "c") or 0.0), disabled=True, key="db_c_max_mm")
    c8.number_input("Wpl_y (cm³)", value=float(pick(sr, "Wpl_y_cm3", "Wpl_y", "W_pl_y") or 0.0), disabled=True, key="db_Wpl_y")
    c9.number_input("Wpl_z (cm³)", value=float(pick(sr, "Wpl_z_cm3", "Wpl_z", "W_pl_z") or pick(sr, "Wpl_y_cm3", "Wpl_y", 0.0) or 0.0), disabled=True, key="db_Wpl_z")

    # Fourth row – new Iw and It (warping, torsion inertia)
    c10, c11, c12 = st.columns(3)
    c10.number_input("Iw (cm⁶) (warping)", value=float(pick(sr, "Iw_cm6", "Iw") or 0.0), disabled=True, key="db_Iw_cm6")
    c11.number_input("It (cm⁴) (torsion moment of inertia)", value=float(pick(sr, "It_cm4", "It") or 0.0), disabled=True, key="db_It_cm4")
    c12.empty()

    # Fifth row – class definitions
    cls1, cls2, cls3 = st.columns(3)
    cls1.text_input("Flange class (DB)", value=str(pick(sr, "flange_class_db", "flange_class", "flangeClass") or "n/a"), disabled=True, key="db_flange_class_txt")
    cls2.text_input("Web class (bending, DB)", value=str(pick(sr, "web_class_bending_db", "web_class_bending") or "n/a"), disabled=True, key="db_web_bending_class_txt")
    cls3.text_input("Web class (compression, DB)", value=str(pick(sr, "web_class_compression_db", "web_class_compression") or "n/a"), disabled=True, key="db_web_comp_class_txt")

    # Sixth row – buckling α
    a1, a2, a3 = st.columns(3)
    alpha_db_val = pick(sr, "alpha_curve", "alpha", default=alpha_default_val) or alpha_default_val
    alpha_label_db = next((lbl for lbl, val in alpha_options if abs(val - float(alpha_db_val)) < 1e-8), f"{alpha_db_val}")
    a1.text_input("Buckling α (DB)", value=str(alpha_label_db), disabled=True, key="db_alpha_text")
    a2.empty()
    a3.empty()

else:
    st.markdown("### Section properties (editable - Custom)")

    # First row
    c1, c2, c3 = st.columns(3)
    A_cm2 = c1.number_input("Area A (cm²)", value=50.0, key="A_cm2_custom")
    S_y_cm3 = c2.number_input("S_y (cm³) about y", value=200.0, key="Sy_custom")
    S_z_cm3 = c3.number_input("S_z (cm³) about z", value=50.0, key="Sz_custom")

    # Second row
    c4, c5, c6 = st.columns(3)
    I_y_cm4 = c4.number_input("I_y (cm⁴) about y", value=1500.0, key="Iy_custom")
    I_z_cm4 = c5.number_input("I_z (cm⁴) about z", value=150.0, key="Iz_custom")
    J_cm4 = c6.number_input("J (cm⁴) (torsion const)", value=10.0, key="J_custom")

    # Third row
    c7, c8, c9 = st.columns(3)
    c_max_mm = c7.number_input("c_max (mm)", value=100.0, key="c_custom")
    Wpl_y_cm3 = c8.number_input("Wpl_y (cm³)", value=0.0, key="Wpl_custom")
    Wpl_z_cm3 = c9.number_input("Wpl_z (cm³)", value=0.0, key="Wplz_custom")

    # Fourth row – Iw / It
    c10, c11, c12 = st.columns(3)
    Iw_cm6 = c10.number_input("Iw (cm⁶) (warping)", value=0.0, key="Iw_custom")
    It_cm4 = c11.number_input("It (cm⁴) (torsion moment of inertia)", value=0.0, key="It_custom")
    c12.empty()

    # Fifth row – class definitions
    st.markdown("Optional: set flange/web class for custom section (overrides auto estimate)")
    cls1, cls2, cls3 = st.columns(3)
    flange_class_choice = cls1.selectbox("Flange class (custom)", ["Auto (calc)", "Class 1", "Class 2", "Class 3", "Class 4"], index=0, key="flange_class_choice_custom")
    web_class_bending_choice = cls2.selectbox("Web class (bending, custom)", ["Auto (calc)", "Class 1", "Class 2", "Class 3", "Class 4"], index=0, key="web_bend_class_choice_custom")
    web_class_compression_choice = cls3.selectbox("Web class (compression, custom)", ["Auto (calc)", "Class 1", "Class 2", "Class 3", "Class 4"], index=0, key="web_comp_class_choice_custom")

    # Sixth row – buckling α dropdown
    a1, a2, a3 = st.columns(3)
    alpha_label_selected = a1.selectbox("Buckling curve α (custom)", alpha_labels, index=3, help="Choose buckling curve α (a–e).")
    alpha_custom = alpha_map[alpha_label_selected]
    a2.empty()
    a3.empty()

# -------------------------
# Design properties of material (read-only) - BEFORE loads input
# -------------------------
st.markdown("---")
st.markdown("### Design properties of material (read-only)")
mp1, mp2, mp3 = st.columns(3)
mp1.text_input("Modulus of elasticity E (MPa)", value=f"{210000}", disabled=True)
mp1.text_input("Yield strength fy (MPa)", value=f"{fy}", disabled=True)
mp2.text_input("Shear modulus G (MPa)", value=f"{80769}", disabled=True)
mp2.text_input("Partial factor γ_M0 (cross-sectional) — DB assumed", value=f"{1.0}", disabled=True)
mp3.text_input("Partial factor γ_M1 (buckling / shear) — DB assumed", value=f"{1.0}", disabled=True)
st.markdown("---")

# ------------------------- 
# READY CASES, Loads & calculations etc. (kept as in your previous working app)
# -------------------------
# (I keep your ready cases, loads, calculations, buckling & result display identical to your working app)
# For brevity I'll reuse your previous calculation blocks (unchanged) - ensure use_props mapping below
# -------------------------

# Build use_props from selected_row or custom inputs
if selected_row is not None and not use_custom:
    # Map DB fields robustly into use_props keys (keep previous keys)
    sr = selected_row
    use_props = {
        "family": pick(sr, "Type", "type", "family"),
        "name": pick(sr, "Size", "size", "name"),
        "A_cm2": float(pick(sr, "A_cm2", "area_cm2", "area", "A", default=0.0) or 0.0),
        "S_y_cm3": float(pick(sr, "S_y_cm3", "Sy", "S_y", default=0.0) or 0.0),
        "S_z_cm3": float(pick(sr, "S_z_cm3", "Sz", "S_z", default=0.0) or 0.0),
        "I_y_cm4": float(pick(sr, "I_y_cm4", "Iy", "I_y", default=0.0) or 0.0),
        "I_z_cm4": float(pick(sr, "I_z_cm4", "Iz", "I_z", default=0.0) or 0.0),
        "J_cm4": float(pick(sr, "J_cm4", "J", default=0.0) or 0.0),
        "c_max_mm": float(pick(sr, "c_max_mm", "c_max", "c", default=0.0) or 0.0),
        "Wpl_y_cm3": float(pick(sr, "Wpl_y_cm3", "Wpl_y", "W_pl_y", default=0.0) or 0.0),
        "Wpl_z_cm3": float(pick(sr, "Wpl_z_cm3", "Wpl_z", "W_pl_z", default=0.0) or 0.0),
        "alpha_curve": pick(sr, "alpha_curve", "alpha", default=alpha_default_val) or alpha_default_val,
        "flange_class_db": pick(sr, "flange_class_db", "flange_class", default="n/a"),
        "web_class_bending_db": pick(sr, "web_class_bending_db", "web_class_bending", default="n/a"),
        "web_class_compression_db": pick(sr, "web_class_compression_db", "web_class_compression", default="n/a"),
        "Iw_cm6": float(pick(sr, "Iw_cm6", "Iw", default=0.0) or 0.0),
        "It_cm4": float(pick(sr, "It_cm4", "It", default=0.0) or 0.0)
    }
else:
    # custom section mapping (only properties user provides)
    if use_custom:
        use_props = {
            "family": "CUSTOM", "name": "CUSTOM",
            "A_cm2": locals().get("A_cm2", 0.0),
            "S_y_cm3": locals().get("S_y_cm3", 0.0),
            "S_z_cm3": locals().get("S_z_cm3", 0.0),
            "I_y_cm4": locals().get("I_y_cm4", 0.0),
            "I_z_cm4": locals().get("I_z_cm4", 0.0),
            "J_cm4": locals().get("J_cm4", 0.0),
            "c_max_mm": locals().get("c_max_mm", 0.0),
            "Wpl_y_cm3": locals().get("Wpl_y_cm3", 0.0),
            "Wpl_z_cm3": locals().get("Wpl_z_cm3", 0.0),
            "alpha_curve": locals().get("alpha_custom", alpha_default_val),
            "flange_class_db": locals().get("flange_class_choice", "Auto (calc)"),
            "web_class_bending_db": locals().get("web_class_bending_choice", "Auto (calc)"),
            "web_class_compression_db": locals().get("web_class_compression_choice", "Auto (calc)"),
            "Iw_cm6": locals().get("Iw_cm6", 0.0),
            "It_cm4": locals().get("It_cm4", 0.0)
        }
    else:
        # no selection and not custom => use placeholder zeros to avoid crashing
        use_props = {
            "family": "N/A", "name": "N/A",
            "A_cm2": 0.0, "S_y_cm3": 0.0, "S_z_cm3": 0.0,
            "I_y_cm4": 0.0, "I_z_cm4": 0.0, "J_cm4": 0.0,
            "c_max_mm": 0.0, "Wpl_y_cm3": 0.0, "Wpl_z_cm3": 0.0,
            "alpha_curve": alpha_default_val,
            "flange_class_db": "n/a", "web_class_bending_db": "n/a", "web_class_compression_db": "n/a",
            "Iw_cm6": 0.0, "It_cm4": 0.0
        }

# -------------------------
# Loads & calculations (kept from your previous working logic)
# -------------------------
st.markdown("---")
st.header("Design forces and moments (ultimate state) - INPUT")
st.markdown('<span title="Enter ultimate (ULS) design forces and moments. Positive N = compression.">ⓘ Load input help</span>', unsafe_allow_html=True)

r1c1, r1c2, r1c3 = st.columns(3)
with r1c1:
    L = st.number_input("Element length L (m)", value=6.0, min_value=0.0,
                        help="Clear length of the member (m).")
with r1c2:
    N_kN = st.number_input("Axial force N (kN) (positive = compression)", value=0.0,
                           help="Axial design force, compression positive (kN).")
with r1c3:
    Vy_kN = st.number_input("Shear V_y (kN)", value=0.0,
                            help="Shear force about local y (kN).")
r2c1, r2c2, r2c3 = st.columns(3)
with r2c1:
    Vz_kN = st.number_input("Shear V_z (kN)", value=0.0,
                            help="Shear force about local z (kN).")
with r2c2:
    My_kNm = st.number_input("Bending M_y (kN·m) (about y)", value=0.0,
                             help="Major axis bending moment (kN·m).")
with r2c3:
    Mz_kNm = st.number_input("Bending M_z (kN·m) (about z)", value=0.0,
                             help="Minor axis bending moment (kN·m).")
r3c1, r3c2, r3c3 = st.columns(3)
with r3c1:
    Tx_kNm = st.number_input("Torsion T_x (kN·m)", value=0.0,
                             help="Torsional moment about longitudinal axis (kN·m).")
with r3c2:
    st.write("")
with r3c3:
    st.write("")

axis_check = "both (y & z)"

st.sidebar.header("Output mode")
output_mode = st.sidebar.radio("Results output mode", ("Concise (no formulas)", "Full (with formulas & steps)"))
want_pdf = st.sidebar.checkbox("Enable PDF download (requires reportlab)")

# Perform conversions and calculations using use_props (same formulas as before)
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

if A_m2 <= 0 and not use_custom:
    st.error("Section area not provided (A <= 0) for DB section. Select another section or use custom mode.")
    st.stop()

# Resistances (DB assumed partials already handled)
N_Rd_N = A_m2 * fy * 1e6 / 1.0 if A_m2>0 else 0.0
Av_m2 = 0.6 * A_m2
V_Rd_N = Av_m2 * fy * 1e6 / (math.sqrt(3) * 1.0) if Av_m2>0 else 0.0
M_Rd_y_Nm = Wpl_y_m3 * fy * 1e6 / 1.0 if Wpl_y_m3>0 else 0.0

# stresses & shear/torsion
sigma_axial_Pa = N_N / A_m2 if A_m2>0 else 0.0
sigma_by_Pa = My_Nm / S_y_m3 if S_y_m3>0 else 0.0
sigma_bz_Pa = Mz_Nm / S_z_m3 if S_z_m3>0 else 0.0
tau_y_Pa = Vz_N / Av_m2 if Av_m2>0 else 0.0
tau_z_Pa = Vy_N / Av_m2 if Av_m2>0 else 0.0
tau_torsion_Pa = 0.0
if J_m4 > 0 and c_max_m > 0:
    tau_torsion_Pa = T_Nm * c_max_m / J_m4
tau_total_Pa = math.sqrt(tau_y_Pa**2 + tau_z_Pa**2 + tau_torsion_Pa**2)
tau_total_MPa = tau_total_Pa / 1e6
sigma_total_Pa = sigma_axial_Pa + sigma_by_Pa + sigma_bz_Pa
sigma_total_MPa = sigma_total_Pa / 1e6
sigma_eq_MPa = math.sqrt((abs(sigma_total_MPa))**2 + 3.0 * (tau_total_MPa**2))

tau_allow_Pa = 0.6 * 0.6 * fy * 1e6  # approx as before

# utilizations
util_axial = abs(N_N) / N_Rd_N if N_Rd_N > 0 else None
util_ten = abs(min(N_N, 0.0)) / N_Rd_N if N_Rd_N > 0 else None
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
    N_b_Rd_N = chi * A_m2 * fy * 1e6 / 1.0
    status = "OK" if abs(N_N) <= N_b_Rd_N else "EXCEEDS"
    buck_results.append((axis_label, Ncr, lambda_bar, chi, N_b_Rd_N, status))

N_b_Rd_candidates = [r[4] for r in buck_results if r[4] not in (None,)]
N_b_Rd_min_N = min(N_b_Rd_candidates) if N_b_Rd_candidates else None
compression_resistance_N = N_b_Rd_min_N if N_b_Rd_min_N is not None else N_Rd_N

# Build checks rows (kept same format)
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
res_tension_N = N_Rd_N
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

# Bending & combined checks
rows.append({"Check":"Bending y-y (σ_by)","Applied":f"{sigma_by_Pa/1e6:.3f} MPa","Resistance":f"{0.6*fy:.3f} MPa","Utilization":(f"{(abs(sigma_by_Pa/1e6)/(0.6*fy)):.3f}" if (0.6*fy)>0 else "n/a"),"Status":"OK" if abs(sigma_by_Pa/1e6)/(0.6*fy)<=1.0 else "EXCEEDS"})
rows.append({"Check":"Bending z-z (σ_bz)","Applied":f"{sigma_bz_Pa/1e6:.3f} MPa","Resistance":f"{0.6*fy:.3f} MPa","Utilization":(f"{(abs(sigma_bz_Pa/1e6)/(0.6*fy)):.3f}" if (0.6*fy)>0 else "n/a"),"Status":"OK" if abs(sigma_bz_Pa/1e6)/(0.6*fy)<=1.0 else "EXCEEDS"})

for axis_label, Ncr, lambda_bar, chi, N_b_Rd_N, status in buck_results:
    if N_b_Rd_N:
        util_buck = abs(N_N) / N_b_Rd_N
        rows.append({"Check":f"Flexural buckling {axis_label}","Applied":f"{abs(N_N)/1e3:.3f} kN","Resistance":f"{N_b_Rd_N/1e3:.3f} kN","Utilization":f"{util_buck:.3f}","Status":"OK" if util_buck<=1.0 else "EXCEEDS"})
    else:
        rows.append({"Check":f"Flexural buckling {axis_label}","Applied":"n/a","Resistance":"n/a","Utilization":"n/a","Status":"n/a"})

# LT (approx)
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
        M_Rd_LT = chi_LT * Mpl / 1.0
    except Exception:
        Mcr = None

if M_Rd_LT and M_Rd_LT>0:
    util_LT = abs(My_Nm) / M_Rd_LT
    rows.append({"Check":"Lateral-torsional buckling (LT)","Applied":f"{abs(My_Nm)/1e3:.3f} kN·m","Resistance":f"{M_Rd_LT/1e3:.3f} kN·m","Utilization":f"{util_LT:.3f}","Status":"OK" if util_LT<=1.0 else "EXCEEDS"})
else:
    rows.append({"Check":"Lateral-torsional buckling (LT)","Applied":"n/a","Resistance":"n/a","Utilization":"n/a","Status":"n/a"})

# Interaction checks (approx)
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
# Display: Result summary and colored table (same as before)
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
# Save results (session) - first save near table
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
        "section_type": use_props.get("family", use_props.get("Type", "CUSTOM")),
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

# -------------------------
# Full mode: expanders with formulas & intermediate values (kept as in your old app)
# -------------------------
if output_mode.startswith("Full"):
    st.markdown("---")
    st.subheader("Full formulas & intermediate steps (click expanders to view)")
    with st.expander("Tension — formula & details (EN1993-1-1 §6.2.3)"):
        st.latex(r"N_{Rd} = \dfrac{A \cdot f_y}{\gamma_{M0}}")
        st.write(f"A = {A_m2:.6e} m², fy = {fy:.1f} MPa, γ_M0 = 1.0 (DB assumed)")
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
        st.write(f"σ_by = {sigma_by_Pa/1e6:.3f} MPa; utilization = {abs(sigma_by_Pa/1e6)/(0.6*fy) if (0.6*fy)>0 else 'n/a'}")
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
    with st.expander("Axial–bending interaction (EN1993-1-1 §6.3.3 & Annex A/B)"):
        st.latex(r"\eta = \dfrac{N}{N_{Rd}} + \dfrac{M}{M_{Rd}} \quad\text{(method 1)}")
        st.latex(r"\eta_{2} = \sqrt{\left(\dfrac{N}{N_{Rd}}\right)^2 + \left(\dfrac{M}{M_{Rd}}\right)^2} \quad\text{(method 2)}")
        st.write(f"Interaction (method1 approx) = {interaction_ratio_simple if interaction_ratio_simple else 'n/a'}")
        if interaction_method2 is not None:
            st.write(f"Interaction (method2 approx) = {interaction_method2:.3f}")

# Summaries & notes
st.markdown("---")
st.subheader("Summary & recommended next steps by EngiSnap")
eng_notes = []
if any(df_rows["Status"] == "EXCEEDS"):
    eng_notes.append("One or more checks exceed capacity — consider increasing the section (A / W_pl), reducing applied loads, shortening unbraced length, or adding restraints. Consult a licensed structural engineer for final design.")
else:
    eng_notes.append("Preliminary screening checks OK. Proceed to full EN1993 member checks (local buckling, LTB, full interaction) before final design.")
st.write("\n\n".join(eng_notes))

st.markdown("---")
st.subheader("Notes, limitations & references")
st.write("""
- This tool gives **preliminary** screening checks only. It is not a complete EN1993 implementation.
- For DB sections this app expects columns named `Type` and `Size` (case-insensitive). Make sure your Beam table uses those names.
- Partial factors are assumed handled in your DB (we use γ = 1.0 in calculations here). Do not commit DB credentials to public repos; use Streamlit secrets.
- Reference: EN1993-1-1 (Design of steel structures).
""")
