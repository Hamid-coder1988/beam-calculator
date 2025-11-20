# beam_design_app.py
# EngiSnap — Beam design Eurocode (DB-backed Type -> Size selection)
import streamlit as st
import pandas as pd
import math
from io import BytesIO
from datetime import datetime, date
import traceback

# -------------------------
# Try DB driver (psycopg2)
# -------------------------
try:
    import psycopg2
    HAS_PG = True
except Exception:
    psycopg2 = None
    HAS_PG = False

# -------------------------
# DB connection (LOCAL testing - hard-coded)
# Replace values or convert to st.secrets for deployment
# -------------------------
def get_conn():
    if not HAS_PG:
        raise RuntimeError("psycopg2 not installed")
    return psycopg2.connect(
        host="yamanote.proxy.rlwy.net",
        port=15500,
        database="railway",
        user="postgres",
        password="KcMoXOMMbbOQITUHrdJMOiwyNBDGyrFy",
        sslmode="require"
    )

# -------------------------
# Sample fallback table rows
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
# SQL helper: run SQL and return DataFrame or error
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
        except:
            pass
        return None, f"{e}\n\n{tb}"

# -------------------------
# Detect table & columns (tries to find the beam table)
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
# Fetch types & sizes preserving table order (ORDER BY ctid)
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

    # try to read Type and Size preserving insertion order via ctid
    sql = f'SELECT "Type", "Size" FROM "{tbl}" ORDER BY ctid;'
    rows_df, err = run_sql(sql)
    if err:
        # fallback unquoted variant
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
# Fetch a single section row by Type & Size (robust variants)
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
# UI: header & metadata
# -------------------------
st.set_page_config(page_title="EngiSnap Beam Design Eurocode Checker", layout="wide")
st.title("EngiSnap — Design of steel memebers (Eurocode)")

st.markdown("""
**Disclaimer (read carefully)**  
This prototype performs simplified Eurocode-style screening checks. It is **not** a full EN1993 design package.
Use for screening/prototyping only — always have final results verified by a licensed structural engineer.
""")

# -------------------------
# Sample DB loaded for non-DB fallback (kept for other parts)
# -------------------------
df_sample_db = pd.DataFrame(SAMPLE_ROWS)

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
# Sidebar: material & defaults
# -------------------------
st.sidebar.header("Material & Eurocode defaults")
material = st.sidebar.selectbox("Material", ("S235", "S275", "S355", "Custom"),
                                help="Select steel grade (typical EN names). For custom, enter fy.")
if material == "S235":
    fy = 235.0
elif material == "S275":
    fy = 275.0
elif material == "S355":
    fy = 355.0
else:
    fy = st.sidebar.number_input("Enter yield stress fy (MPa)", value=355.0, min_value=1.0,
                                 help="Yield strength fy (MPa) used for resistance calculations")

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
# Section selection (DB-backed Type -> Size)
# -------------------------
st.header("Section selection")
st.markdown('<span title="Select a standard section from DB (read-only) or use custom.">ⓘ Section selection help</span>',
            unsafe_allow_html=True)

col_left, col_right = st.columns([1,1])
with col_left:
    # Fetch types & sizes (order preserved)
    types, sizes_map, detected_table = fetch_types_and_sizes()
    if types:
        family = st.selectbox("Section family / Type (DB)", options=["-- choose --"] + types,
                              help="Choose section family (database). If none selected, use Custom.")
    else:
        family = st.selectbox("Section family / Type (DB)", options=["-- choose --"], help="No DB types found.")
with col_right:
    selected_name = None
    selected_row = None
    if family and family != "-- choose --":
        names = sizes_map.get(family, [])
        if names:
            selected_name = st.selectbox("Section size (DB)", options=["-- choose --"] + names,
                                         help="Choose section size (database). Selecting a size loads read-only properties.")
            if selected_name and selected_name != "-- choose --":
                # load the row from DB (or sample)
                selected_row = get_section_row_db(family, selected_name, detected_table if detected_table != "sample" else None)

st.markdown("**Or select Custom**. Standard DB sections are read-only; custom sections are editable.")
use_custom = st.checkbox("Use custom section (enable manual inputs)", help="Tick to enter section properties manually (CUSTOM)")

if (selected_row is None) and not use_custom:
    st.info("Please select a section size from the DB above, or tick 'Use custom section' to enter properties manually.")

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
    Image 1 — cross-section placeholder
</div>
<p style="font-size:12px;color:gray;margin-top:-10px;text-align:center">(Replace this box with section image from database when available.)</p>
""",
    unsafe_allow_html=True,
)
st.markdown("---")

# -------------------------
# Map selected_row -> use_props (robust)
# -------------------------
def pick(d, *keys, default=None):
    """Return first existing key from d (case-insensitive), or default."""
    if d is None:
        return default
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    kl = {str(k).lower(): k for k in d.keys()}
    for k in keys:
        lk = str(k).lower()
        if lk in kl and d.get(kl[lk]) is not None:
            return d.get(kl[lk])
    return default

if selected_row is not None and not use_custom:
    sr = selected_row
    use_props = {
        "family": pick(sr, "Type", "family", "type", default="DB"),
        "name": pick(sr, "Size", "name", "designation", default=str(pick(sr, "Size", "name", default="DB"))),
        "A_cm2": float(pick(sr, "A_cm2", "area_cm2", "area", "A", "Area", default=0.0) or 0.0),
        "S_y_cm3": float(pick(sr, "S_y_cm3", "Sy_cm3", "S_y", "S_y_mm3", default=0.0) or 0.0),
        "S_z_cm3": float(pick(sr, "S_z_cm3", "Sz_cm3", "S_z", default=0.0) or 0.0),
        "I_y_cm4": float(pick(sr, "I_y_cm4", "Iy_cm4", "I_y", "Iy", default=0.0) or 0.0),
        "I_z_cm4": float(pick(sr, "I_z_cm4", "Iz_cm4", "I_z", "Iz", default=0.0) or 0.0),
        "J_cm4": float(pick(sr, "J_cm4", "J", default=0.0) or 0.0),
        "c_max_mm": float(pick(sr, "c_max_mm", "c_mm", "c", default=0.0) or 0.0),
        "Wpl_y_cm3": float(pick(sr, "Wpl_y_cm3", "Wpl_y", default=0.0) or 0.0),
        "Wpl_z_cm3": float(pick(sr, "Wpl_z_cm3", "Wpl_z", default=0.0) or 0.0),
        "alpha_curve": float(pick(sr, "alpha_curve", "alpha", default=alpha_default_val) or alpha_default_val),
        "flange_class_db": pick(sr, "flange_class_db", "flange_class", default="n/a"),
        "web_class_bending_db": pick(sr, "web_class_bending_db", "web_class_bending", "web_class", default="n/a"),
        "web_class_compression_db": pick(sr, "web_class_compression_db", "web_class_compression", default="n/a"),
        "Iw_cm6": float(pick(sr, "Iw_cm6", "Iw", default=0.0) or 0.0),
        "It_cm4": float(pick(sr, "It_cm4", "It", default=0.0) or 0.0),
    }
    # show small summary of loaded props
    st.markdown("**Loaded section properties from DB:**")
    st.write(f"Family: `{use_props['family']}` — Name: `{use_props['name']}`")
    st.write(pd.DataFrame([use_props]).T.rename(columns={0:"value"}))
else:
    # use the custom inputs if no DB row selected
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
# Section properties display (DB read-only or Custom editable)
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

if selected_row is not None and not use_custom:
    st.markdown("### Section properties (from DB — read only)")
    sr = use_props  # mapped values

    # First row
    c1, c2, c3 = st.columns(3)
    c1.number_input("A (cm²)", value=float(sr.get('A_cm2', 0.0)), disabled=True, key="db_A_cm2")
    c2.number_input("S_y (cm³) about y", value=float(sr.get('S_y_cm3', 0.0)), disabled=True, key="db_Sy_cm3")
    c3.number_input("S_z (cm³) about z", value=float(sr.get('S_z_cm3', 0.0)), disabled=True, key="db_Sz_cm3")

    # Second row
    c4, c5, c6 = st.columns(3)
    c4.number_input("I_y (cm⁴) about y", value=float(sr.get('I_y_cm4', 0.0)), disabled=True, key="db_Iy_cm4")
    c5.number_input("I_z (cm⁴) about z", value=float(sr.get('I_z_cm4', 0.0)), disabled=True, key="db_Iz_cm4")
    c6.number_input("J (cm⁴) (torsion const)", value=float(sr.get('J_cm4', 0.0)), disabled=True, key="db_J_cm4")

    # Third row
    c7, c8, c9 = st.columns(3)
    c7.number_input("c_max (mm)", value=float(sr.get('c_max_mm', 0.0)), disabled=True, key="db_c_max_mm")
    c8.number_input("Wpl_y (cm³)", value=float(sr.get('Wpl_y_cm3', 0.0)), disabled=True, key="db_Wpl_y")
    c9.number_input("Wpl_z (cm³)", value=float(sr.get('Wpl_z_cm3', sr.get('Wpl_y_cm3', 0.0))), disabled=True, key="db_Wpl_z")

    # Fourth row – Iw/It
    c10, c11, c12 = st.columns(3)
    c10.number_input("Iw (cm⁶) (warping)", value=float(sr.get("Iw_cm6", 0.0)), disabled=True, key="db_Iw_cm6")
    c11.number_input("It (cm⁴) (torsion moment of inertia)", value=float(sr.get("It_cm4", 0.0)), disabled=True, key="db_It_cm4")
    c12.empty()

    # Classes and alpha
    cls1, cls2, cls3 = st.columns(3)
    cls1.text_input("Flange class (DB)", value=str(sr.get('flange_class_db', "n/a")), disabled=True, key="db_flange_class_txt")
    cls2.text_input("Web class (bending, DB)", value=str(sr.get('web_class_bending_db', "n/a")), disabled=True, key="db_web_bending_class_txt")
    cls3.text_input("Web class (compression, DB)", value=str(sr.get('web_class_compression_db', "n/a")), disabled=True, key="db_web_comp_class_txt")
    a1, a2, a3 = st.columns(3)
    alpha_db_val = sr.get('alpha_curve', 0.49)
    alpha_label_db = next((lbl for lbl, val in alpha_options if abs(val - float(alpha_db_val)) < 1e-8), f"{alpha_db_val}")
    a1.text_input("Buckling α (DB)", value=str(alpha_label_db), disabled=True, key="db_alpha_text")
    a2.empty(); a3.empty()

else:
    st.markdown("### Section properties (editable - Custom)")
    # Custom inputs (unchanged from your original)
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
    alpha_label_selected = a1.selectbox("Buckling curve α (custom)", alpha_labels, index=3, help="Choose buckling curve α (a–e).")
    alpha_custom = alpha_map[alpha_label_selected]
    a2.empty(); a3.empty()

# -------------------------
# Design properties of material (read-only)
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
# READY CASES SECTION (unchanged)
# -------------------------
st.markdown("---")
st.markdown("### Ready beam & frame cases (optional)")
st.write("You can enter loads manually below **or** select a ready beam/frame case to auto-fill typical maxima. First choose whether this is a **Beam** or a **Frame**, then pick a category and case.")
use_ready = st.checkbox("Use ready case (select a template to prefill loads)", key="ready_use_case")

# (ready_catalog and UI unchanged — omitted in this snippet for brevity)
# For full app keep your ready_catalog and logic from your original file here.
# -------------------------
# Loads & inputs (unchanged)
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
# Choose properties to use (DB vs custom)
# -------------------------
# This block is intentionally kept simple because use_props was already set above
if use_props is None:
    st.error("No section properties available. Select DB section or enable custom inputs.")
    st.stop()

# -------------------------
# Unit conversions & derived values (unchanged)
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
# Calculations (unchanged from your original app)
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

# (The remaining calculation & output section is unchanged - include your original code here)
# For brevity in this response I omit re-pasting the long results table-generation and final outputs.
# Paste your original rows-building, display and save-results code here (unchanged).

st.markdown("---")
st.write("Calculation complete. (Results table and remaining UI preserved from original app.)")
