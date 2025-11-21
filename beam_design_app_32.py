# beam_design_app.py
# EngiSnap — Beam design Eurocode Checker (DB-backed)
# SECURITY NOTE:
# - For local testing you can add a local secrets.toml file with your DB credentials.
# - DO NOT hard-code credentials into this file for a public repo.
# Example (in .streamlit/secrets.toml):
# [postgres]
# host = "yamanote.proxy.rlwy.net"
# port = "15500"
# database = "railway"
# user = "postgres"
# password = "YOURPASSWORD"

import streamlit as st
import pandas as pd
import math
from io import BytesIO
from datetime import datetime, date
import psycopg2
from urllib.parse import urlparse

st.set_page_config(page_title="EngiSnap Beam Design Eurocode Checker", layout="wide")
st.title("EngiSnap — Design of steel members (Eurocode)")

st.markdown("""
**Disclaimer (read carefully)**  
This prototype performs simplified Eurocode-style screening checks. It is **not** a full EN1993 design package.
Use for screening/prototyping only — always have final results verified by a licensed structural engineer.
""")

# -------------------------
# DB helper: get connection (uses st.secrets)
# -------------------------
@st.cache_resource(show_spinner=False)
def get_conn():
    """
    Build psycopg2 connection from st.secrets['postgres'].
    Ensure you configured Streamlit secrets or provide a local secrets.toml.
    """
    pg = st.secrets.get("postgres", None)
    if not pg:
        raise RuntimeError('st.secrets["postgres"] not found. Add DB connection under st.secrets["postgres"].')
    conn = psycopg2.connect(
        host=pg["host"],
        port=str(pg["port"]),
        database=pg["database"],
        user=pg["user"],
        password=pg["password"]
    )
    return conn

# -------------------------
# Safe helpers for mapping DB columns
# -------------------------
def pick(d, *keys, default=None):
    """Return first non-null value found in dict d for any key in keys (case-sensitive and common variants)."""
    if d is None:
        return default
    # try exact keys first
    for k in keys:
        if k in d and pd.notnull(d[k]):
            return d[k]
    # try case-insensitive fallback
    lk = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        kl = str(k).lower()
        if kl in lk and pd.notnull(lk[kl]):
            return lk[kl]
    return default

def to_float(x, default=0.0):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default

# -------------------------
# Load DB into DataFrame (fallback sample if DB not available)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_beam_db():
    try:
        conn = get_conn()
        sql = "SELECT * FROM beam ORDER BY id NULLS LAST;"  # attempt common table name 'beam'
        df = pd.read_sql(sql, conn)
        # preserve DB ordering as-is (do NOT sort)
        return df
    except Exception:
        # try alternative table names
        try:
            conn = get_conn()
            sql2 = "SELECT * FROM \"Beam\" ORDER BY id NULLS LAST;"
            df = pd.read_sql(sql2, conn)
            return df
        except Exception:
            # fallback sample
            sample_rows = [
                {"family": "IPE", "name": "IPE 200",
                 "A_cm2": 31.2, "S_y_cm3": 88.0, "S_z_cm3": 16.0,
                 "I_y_cm4": 1760.0, "I_z_cm4": 64.0, "J_cm4": 14.0, "c_max_mm": 100.0,
                 "Wpl_y_cm3": 96.8, "Wpl_z_cm3": 18.0, "alpha_curve": 0.49,
                 "flange_class_db": "Class 1/2", "web_class_bending_db": "Class 2", "web_class_compression_db": "Class 3",
                 "Iw_cm6": 2500.0, "It_cm4": 14.0
                },
                {"family": "RHS", "name": "RHS 150x100x6.3",
                 "A_cm2": 25.0, "S_y_cm3": 60.0, "S_z_cm3": 40.0,
                 "I_y_cm4": 900.0, "I_z_cm4": 450.0, "J_cm4": 10.0, "c_max_mm": 75.0,
                 "Wpl_y_cm3": 48.0, "Wpl_z_cm3": 32.0, "alpha_curve": 0.34,
                 "flange_class_db": "N/A (RHS)", "web_class_bending_db": "N/A (RHS)", "web_class_compression_db": "N/A (RHS)",
                 "Iw_cm6": 0.0, "It_cm4": 0.0
                }
            ]
            return pd.DataFrame(sample_rows)

# Load DB
df_sample_db = load_beam_db()

# -------------------------
# Sidebar: material & section selection (moved here)
# -------------------------
st.sidebar.header("Material & section selection")

material = st.sidebar.selectbox("Material", ("S235", "S275", "S355"),
                                help="Select steel grade (typical EN names).")
if material == "S235":
    fy = 235.0
elif material == "S275":
    fy = 275.0
else:
    fy = 355.0

# Partial factors removed from UI (assumed handled in DB)
gamma_M0 = 1.0
gamma_M1 = 1.0

# Buckling K factors (left in sidebar)
st.sidebar.markdown("Buckling effective length factors (K):")
K_z = st.sidebar.number_input("K_z — flexural buckling about z–z", value=1.0, min_value=0.1, step=0.05)
K_y = st.sidebar.number_input("K_y — flexural buckling about y–y", value=1.0, min_value=0.1, step=0.05)
K_LT = st.sidebar.number_input("K_LT — lateral–torsional buckling", value=1.0, min_value=0.1, step=0.05)
K_T = st.sidebar.number_input("K_T — torsional buckling factor (reserved)", value=1.0, min_value=0.1, step=0.05)

alpha_default_val = 0.49

# Section selection in sidebar (preserve DB order; do NOT sort sizes)
st.sidebar.markdown("---")
st.sidebar.subheader("Section selection (DB)")

families_list = df_sample_db['family'].dropna().unique().tolist()
family = st.sidebar.selectbox("Section family", options=["-- choose --"] + families_list)

selected_row = None
selected_name = None
if family and family != "-- choose --":
    df_f = df_sample_db[df_sample_db['family'] == family]
    names = df_f['name'].dropna().tolist()
    selected_name = st.sidebar.selectbox("Section size", options=["-- choose --"] + names)
    if selected_name and selected_name != "-- choose --":
        selected_row = df_f[df_f['name'] == selected_name].iloc[0].to_dict()
        st.sidebar.success(f"Loaded: {selected_name}")

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
# Cross-section schematic (center)
# -------------------------
st.markdown("### Cross-section schematic")
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
    Image — cross-section placeholder
</div>
<p style="font-size:12px;color:gray;margin-top:-10px;text-align:center">(Replace with image from DB when available.)</p>
""",
    unsafe_allow_html=True,
)

# -------------------------
# Section properties display (grouped)
# -------------------------
st.markdown("---")
st.header("Section properties")

use_custom = st.checkbox("Use custom section (enable manual inputs)", help="Tick to enter section properties manually (CUSTOM)")

# Map DB row to consistent keys (use pick with many name variants)
if selected_row is not None and not use_custom:
    sr = selected_row
    # area candidates
    A_cm2 = to_float(pick(sr, "A_cm2", "area_cm2", "area", "A", "Area", "AREA", "area_cm"), default=0.0)
    # profile dims - many possible column names
    bf_mm = to_float(pick(sr, "bf_mm", "bf", "bf_mm_", "bf", "flange_width", "b_f", "b"), default=0.0)
    tf_mm = to_float(pick(sr, "tf_mm", "tf", "flange_thickness", "t_f", "tf"), default=0.0)
    hw_mm = to_float(pick(sr, "hw_mm", "hw", "web_height", "h_w", "hw_mm_"), default=0.0)
    tw_mm = to_float(pick(sr, "tw_mm", "tw", "web_thickness", "t_w", "tw_mm_"), default=0.0)
    h_mm = to_float(pick(sr, "h_mm", "h", "depth_mm", "overall_height", "height_mm"), default=0.0)
    # inertia & section modulus
    I_y_cm4 = to_float(pick(sr, "I_y_cm4", "Iy_cm4", "I_y", "Iyy", "I_y_mm4", "I_y_cm4_"), default=0.0)
    I_z_cm4 = to_float(pick(sr, "I_z_cm4", "Iz_cm4", "I_z", "Izz", "I_z_mm4"), default=0.0)
    S_y_cm3 = to_float(pick(sr, "S_y_cm3", "Sy_cm3", "S_y", "W_el_y", "section_mod_y", "S_y_mm3"), default=0.0)
    S_z_cm3 = to_float(pick(sr, "S_z_cm3", "Sz_cm3", "S_z", "W_el_z", "section_mod_z", "S_z_mm3"), default=0.0)
    Wpl_y_cm3 = to_float(pick(sr, "Wpl_y_cm3", "Wpl_y", "Wpl_y_cm3_", "Wpl_y_mm3", "W_pl_y"), default=0.0)
    Wpl_z_cm3 = to_float(pick(sr, "Wpl_z_cm3", "Wpl_z", "W_pl_z"), default=0.0)
    # torsion & warping
    J_cm4 = to_float(pick(sr, "J_cm4", "J_cm", "J", "J_mm4", "torsion_const"), default=0.0)
    It_cm4 = to_float(pick(sr, "It_cm4", "It", "It_cm", "It_mm4"), default=0.0)
    Iw_cm6 = to_float(pick(sr, "Iw_cm6", "Iw", "Iw_cm6_", "Iw_mm6"), default=0.0)
    c_max_mm = to_float(pick(sr, "c_max_mm", "c_max", "c_max_mm_", "c_max"), default=0.0)
    # resistances (if DB stores)
    N_Rd_kN_db = to_float(pick(sr, "N_Rd_kN", "N_Rd", "Nrd_kN", "N_Rd_kN_db"), default=None)
    V_Rd_kN_db = to_float(pick(sr, "V_Rd_kN", "V_Rd", "Vrd_kN"), default=None)
    M_Rd_y_kNm_db = to_float(pick(sr, "M_Rd_y_kNm", "M_Rd_y", "Mrd_y_kNm", "M_Rd_y_kNm_db"), default=None)
    # buckling & classes
    alpha_curve_db = pick(sr, "alpha_curve", "alpha", "buckling_alpha", default=alpha_default_val)
    flange_class_db = pick(sr, "flange_class_db", "flange_class", "flange_class_txt", default="n/a")
    web_class_bending_db = pick(sr, "web_class_bending_db", "web_class_bending", default="n/a")
    web_class_compression_db = pick(sr, "web_class_compression_db", "web_class_compression", default="n/a")
    # map to use_props structure
    selected_row_mapped = {
        "family": pick(sr, "family", "type", default=family),
        "name": pick(sr, "name", "size", default=selected_name),
        "A_cm2": A_cm2,
        "bf_mm": bf_mm, "tf_mm": tf_mm, "hw_mm": hw_mm, "tw_mm": tw_mm, "h_mm": h_mm,
        "I_y_cm4": I_y_cm4, "I_z_cm4": I_z_cm4,
        "S_y_cm3": S_y_cm3, "S_z_cm3": S_z_cm3,
        "Wpl_y_cm3": Wpl_y_cm3, "Wpl_z_cm3": Wpl_z_cm3,
        "J_cm4": J_cm4, "It_cm4": It_cm4, "Iw_cm6": Iw_cm6, "c_max_mm": c_max_mm,
        "N_Rd_kN_db": N_Rd_kN_db, "V_Rd_kN_db": V_Rd_kN_db, "M_Rd_y_kNm_db": M_Rd_y_kNm_db,
        "alpha_curve": alpha_curve_db,
        "flange_class_db": flange_class_db,
        "web_class_bending_db": web_class_bending_db,
        "web_class_compression_db": web_class_compression_db
    }
else:
    selected_row_mapped = None

# If custom, show editable inputs (same layout as before)
if use_custom:
    st.markdown("### Section properties (Custom - editable)")
    # Profile dimensions
    with st.expander("Profile dimensions (editable)"):
        p1, p2, p3 = st.columns(3)
        bf_mm = p1.number_input("Flange width bf (mm)", value=100.0, key="bf_custom")
        tf_mm = p2.number_input("Flange thickness tf (mm)", value=8.0, key="tf_custom")
        hw_mm = p3.number_input("Web height hw (mm)", value=100.0, key="hw_custom")
        tw_mm = p1.number_input("Web thickness tw (mm)", value=6.0, key="tw_custom")
        h_mm = p2.number_input("Overall depth h (mm)", value=200.0, key="h_custom")
    # Area & inertia
    with st.expander("Area & inertia (editable)"):
        q1, q2, q3 = st.columns(3)
        A_cm2 = q1.number_input("Area A (cm²)", value=50.0, key="A_custom")
        I_y_cm4 = q2.number_input("I_y (cm⁴)", value=1500.0, key="Iy_custom")
        I_z_cm4 = q3.number_input("I_z (cm⁴)", value=150.0, key="Iz_custom")
        S_y_cm3 = q1.number_input("S_y (cm³)", value=200.0, key="Sy_custom")
        S_z_cm3 = q2.number_input("S_z (cm³)", value=50.0, key="Sz_custom")
    with st.expander("Torsion & warping (editable)"):
        J_cm4 = st.number_input("J (cm⁴)", value=10.0, key="J_custom")
        It_cm4 = st.number_input("It (cm⁴)", value=0.0, key="It_custom")
        Iw_cm6 = st.number_input("Iw (cm⁶)", value=0.0, key="Iw_custom")
        c_max_mm = st.number_input("c_max (mm)", value=100.0, key="c_custom")
    flange_class_db = st.selectbox("Flange class (custom)", ["Auto","Class 1","Class 2","Class 3","Class 4"], index=0)
    web_class_bending_db = st.selectbox("Web class (bending, custom)", ["Auto","Class 1","Class 2","Class 3","Class 4"], index=0)
    web_class_compression_db = st.selectbox("Web class (compression, custom)", ["Auto","Class 1","Class 2","Class 3","Class 4"], index=0)
else:
    # Display DB properties grouped
    st.markdown("### Section properties (from DB — read only)")
    if selected_row_mapped is None:
        st.info("Please select a section family and size in the sidebar, or tick 'Use custom section'.")
    else:
        sr = selected_row_mapped
        # Top: Profile dimensions (next to schematic visually)
        st.write("**Profile dimensions**")
        d1, d2, d3 = st.columns(3)
        d1.number_input("Flange width bf (mm)", value=float(sr.get("bf_mm", 0.0)), disabled=True, key="db_bf_mm")
        d2.number_input("Flange thickness tf (mm)", value=float(sr.get("tf_mm", 0.0)), disabled=True, key="db_tf_mm")
        d3.number_input("Web height hw (mm)", value=float(sr.get("hw_mm", 0.0)), disabled=True, key="db_hw_mm")
        e1, e2, e3 = st.columns(3)
        e1.number_input("Web thickness tw (mm)", value=float(sr.get("tw_mm", 0.0)), disabled=True, key="db_tw_mm")
        e2.number_input("Overall depth h (mm)", value=float(sr.get("h_mm", 0.0)), disabled=True, key="db_h_mm")
        e3.empty()

        # Area & basic section properties (visible)
        st.write("**Area & basic properties**")
        a1, a2, a3 = st.columns(3)
        a1.number_input("Area A (cm²)", value=float(sr.get("A_cm2", 0.0)), disabled=True, key="db_A_cm2")
        a2.number_input("Centroid y (mm) (if stored)", value=float(pick(selected_row, "y_bar_mm", "centroid_y", default=0.0) or 0.0), disabled=True, key="db_ybar")
        a3.number_input("Centroid z (mm) (if stored)", value=float(pick(selected_row, "z_bar_mm", "centroid_z", default=0.0) or 0.0), disabled=True, key="db_zbar")

        # Inertia & section modulus
        st.write("**Inertia & section modulus**")
        b1, b2, b3 = st.columns(3)
        b1.number_input("I_y (cm⁴)", value=float(sr.get("I_y_cm4", 0.0)), disabled=True, key="db_Iy_cm4")
        b2.number_input("I_z (cm⁴)", value=float(sr.get("I_z_cm4", 0.0)), disabled=True, key="db_Iz_cm4")
        b3.number_input("J (cm⁴) (torsion const)", value=float(sr.get("J_cm4", 0.0)), disabled=True, key="db_J_cm4")
        c1, c2, c3 = st.columns(3)
        c1.number_input("S_y (cm³)", value=float(sr.get("S_y_cm3", 0.0)), disabled=True, key="db_Sy_cm3")
        c2.number_input("S_z (cm³)", value=float(sr.get("S_z_cm3", 0.0)), disabled=True, key="db_Sz_cm3")
        c3.number_input("W_pl,y (cm³) (if stored)", value=float(sr.get("Wpl_y_cm3", 0.0)), disabled=True, key="db_Wpl_y")

        # Torsion & warping expander
        with st.expander("Torsion & warping"):
            t1, t2, t3 = st.columns(3)
            t1.number_input("Iw (cm⁶) (warping)", value=float(sr.get("Iw_cm6", 0.0)), disabled=True, key="db_Iw_cm6")
            t2.number_input("It (cm⁴) (torsion moment of inertia)", value=float(sr.get("It_cm4", 0.0)), disabled=True, key="db_It_cm4")
            t3.number_input("c_max (mm)", value=float(sr.get("c_max_mm", 0.0)), disabled=True, key="db_c_max_mm")

        # Resistances expander (computed and DB-provided if present)
        with st.expander("Resistances (computed & DB values)"):
            # We'll compute derived resistances later in same units; show DB stored ones if present
            st.write("DB-provided resistances (if any):")
            st.write(f"N_Rd (kN): {sr.get('N_Rd_kN_db','n/a')}")
            st.write(f"V_Rd (kN): {sr.get('V_Rd_kN_db','n/a')}")
            st.write(f"M_Rd,y (kN·m): {sr.get('M_Rd_y_kNm_db','n/a')}")

        # Buckling & classes expander
        with st.expander("Buckling curves & section classes"):
            st.text_input("Buckling α (DB)", value=str(sr.get("alpha_curve", alpha_default_val)), disabled=True)
            st.text_input("Flange class (DB)", value=str(sr.get("flange_class_db", "n/a")), disabled=True)
            st.text_input("Web class (bending, DB)", value=str(sr.get("web_class_bending_db", "n/a")), disabled=True)
            st.text_input("Web class (compression, DB)", value=str(sr.get("web_class_compression_db", "n/a")), disabled=True)

        # DB metadata (collapsed)
        with st.expander("DB metadata (debug)"):
            st.write(selected_row)  # show raw row dict for mapping/debugging

# -------------------------
# Design properties of material (read-only) - BEFORE loads input
# -------------------------
st.markdown("---")
st.markdown("### Design properties of material (read-only)")
mp1, mp2, mp3 = st.columns(3)
mp1.text_input("Modulus of elasticity E (MPa)", value=f"{210000}", disabled=True)
mp1.text_input("Yield strength fy (MPa)", value=f"{fy}", disabled=True)
mp2.text_input("Shear modulus G (MPa)", value=f"{80769}", disabled=True)
mp2.text_input("Partial factor γ_M0 (cross-sectional) — from DB or default", value=f"{gamma_M0}", disabled=True)
mp3.text_input("Partial factor γ_M1 (stability / shear) — from DB or default", value=f"{gamma_M1}", disabled=True)
st.markdown("---")

# -------------------------
# READY CASES (kept as before)
# -------------------------
st.markdown("---")
st.markdown("### Ready beam & frame cases (optional)")
use_ready = st.checkbox("Use ready case (select a template to prefill loads)", key="ready_use_case")
# (minimal ready catalog — same helper functions as earlier)
def ss_udl(span_m: float, w_kN_per_m: float):
    Mmax = w_kN_per_m * span_m**2 / 8.0
    Vmax = w_kN_per_m * span_m / 2.0
    return (0.0, float(Mmax), 0.0, float(Vmax), 0.0)
def ss_point_center(span_m: float, P_kN: float):
    Mmax = P_kN * span_m / 4.0
    Vmax = P_kN / 2.0
    return (0.0, float(Mmax), 0.0, float(Vmax), 0.0)
ready_catalog = {
    "Beam": {
        "Simply supported (examples)": {
            "SS-UDL": {"label":"SS-01: UDL (w on L)","inputs":{"L":6.0,"w":10.0},"func":ss_udl},
            "SS-Point-Centre": {"label":"SS-02: Point at midspan (P)","inputs":{"L":6.0,"P":20.0},"func":ss_point_center},
        }
    }
}
# UI for ready cases (same logic as earlier, but compact)
if use_ready:
    st.markdown("**Step 1 — choose object type (Beam or Frame)**")
    chosen_type = st.selectbox("Type", options=["-- choose --", "Beam", "Frame"], key="ready_type")
    if chosen_type and chosen_type != "-- choose --":
        categories = sorted(ready_catalog.get(chosen_type, {}).keys())
        if categories:
            chosen_cat = st.selectbox("Category", options=["-- choose --"] + categories, key="ready_category")
            if chosen_cat and chosen_cat != "-- choose --":
                cases_dict = ready_catalog[chosen_type][chosen_cat]
                st.markdown("**Choose a case**")
                case_keys = list(cases_dict.keys())
                n_cols = 3
                cols = st.columns(n_cols)
                selected_case_key = None
                for i, ck in enumerate(case_keys):
                    col = cols[i % n_cols]
                    lbl = cases_dict[ck]["label"]
                    col.markdown(f"<div style='border:1px solid #ddd;padding:8px;border-radius:6px;text-align:center'>{lbl}</div>", unsafe_allow_html=True)
                    if col.button(f"Select {ck}", key=f"ready_select_{i}"):
                        selected_case_key = ck
                if selected_case_key:
                    st.session_state["ready_selected_case"] = selected_case_key
                sel_case = st.session_state.get("ready_selected_case")
                if sel_case:
                    scase_info = cases_dict[sel_case]
                    st.markdown(f"**Selected case:** {scase_info['label']}")
                    inputs = scase_info.get("inputs", {})
                    input_vals = {}
                    for k, v in inputs.items():
                        input_vals[k] = st.number_input(f"{k}", value=float(v), key=f"ready_input_{sel_case}_{k}")
                    if st.button("Apply case to load inputs", key=f"ready_apply_{sel_case}"):
                        func = scase_info.get("func")
                        args = [input_vals[k] for k in inputs.keys()]
                        try:
                            N_case, My_case, Mz_case, Vy_case, Vz_case = func(*args)
                        except Exception:
                            N_case, My_case, Mz_case, Vy_case, Vz_case = 0.0,0.0,0.0,0.0,0.0
                        st.session_state["prefill_from_case"] = True
                        st.session_state["prefill_N_kN"] = float(N_case)
                        st.session_state["prefill_My_kNm"] = float(My_case)
                        st.session_state["prefill_Mz_kNm"] = float(Mz_case)
                        st.session_state["prefill_Vy_kN"] = float(Vy_case)
                        st.session_state["prefill_Vz_kN"] = float(Vz_case)
                        if "L" in input_vals:
                            st.session_state["case_L"] = float(input_vals["L"])
                        st.success("Case applied — scroll to loads and moments.")

# -------------------------
# Loads & inputs
# -------------------------
st.header("Design forces and moments (ultimate state) - INPUT")
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
if not use_custom and selected_row_mapped is not None:
    use_props = dict(selected_row_mapped)
    # If DB stores gamma, override defaults
    gamma_M0 = to_float(pick(selected_row, "gamma_M0", "gammaM0", default=gamma_M0))
    gamma_M1 = to_float(pick(selected_row, "gamma_M1", "gammaM1", default=gamma_M1))
else:
    # custom or nothing selected
    use_props = {
        "family": "CUSTOM", "name": "CUSTOM",
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
        "bf_mm": locals().get("bf_mm", 0.0), "tf_mm": locals().get("tf_mm", 0.0),
        "hw_mm": locals().get("hw_mm", 0.0), "tw_mm": locals().get("tw_mm", 0.0),
        "flange_class_db": locals().get("flange_class_db", "Auto"),
        "web_class_bending_db": locals().get("web_class_bending_db", "Auto"),
        "web_class_compression_db": locals().get("web_class_compression_db", "Auto"),
        "Iw_cm6": locals().get("Iw_cm6", 0.0), "It_cm4": locals().get("It_cm4", 0.0)
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
# Basic simplified Eurocode-style resistances
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

tau_allow_Pa = 0.6 * (0.6 * fy) * 1e6  # approximate allowable (0.6 * sigma_allow where sigma_allow=0.6*fy)

# utilizations
util_axial = abs(N_N) / N_Rd_N if N_Rd_N > 0 else None
util_ten = abs(min(N_N, 0.0)) / T_Rd_N if T_Rd_N > 0 else None
util_My = abs(My_Nm) / M_Rd_y_Nm if M_Rd_y_Nm > 0 else None
util_shear_resultant = math.sqrt(Vy_N**2 + Vz_N**2) / V_Rd_N if V_Rd_N > 0 else None
util_torsion = (tau_torsion_Pa / tau_allow_Pa) if tau_allow_Pa > 0 else None

# -------------------------
# Buckling simplified
# -------------------------
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
# Build checks rows
# -------------------------
def status_and_util(applied, resistance):
    if resistance is None or resistance == 0:
        return ("n/a", None)
    util = abs(applied) / resistance
    return ("OK" if util <= 1.0 else "EXCEEDS", util)

rows = []
applied_N = N_N if N_N >= 0 else 0.0
res_comp_N = compression_resistance_N
status_comp, util_comp = status_and_util(applied_N, res_comp_N)
rows.append({"Check":"Compression (N≥0)","Applied":f"{applied_N/1e3:.3f} kN","Resistance":(f"{res_comp_N/1e3:.3f} kN" if res_comp_N else "n/a"),"Utilization":(f"{util_comp:.3f}" if util_comp else "n/a"),"Status":status_comp})
applied_tension_N = -N_N if N_N < 0 else 0.0
res_tension_N = T_Rd_N
status_ten, util_ten = status_and_util(applied_tension_N, res_tension_N)
rows.append({"Check":"Tension (N<0)","Applied":f"{applied_tension_N/1e3:.3f} kN","Resistance":f"{res_tension_N/1e3:.3f} kN","Utilization":(f"{util_ten:.3f}" if util_ten else "n/a"),"Status":status_ten})
applied_shear_N = math.sqrt(Vy_N**2 + Vz_N**2)
res_shear_N = V_Rd_N
status_shear, util_shear_val = status_and_util(applied_shear_N, res_shear_N)
rows.append({"Check":"Shear (resultant Vy & Vz)","Applied":f"{applied_shear_N/1e3:.3f} kN","Resistance":f"{res_shear_N/1e3:.3f} kN","Utilization":(f"{util_shear_val:.3f}" if util_shear_val else "n/a"),"Status":status_shear})
applied_tau_Pa = tau_torsion_Pa
res_tau_allow_Pa = tau_allow_Pa
util_torsion = applied_tau_Pa / res_tau_allow_Pa if res_tau_allow_Pa>0 else None
status_tors = "OK" if util_torsion is not None and util_torsion <= 1.0 else ("EXCEEDS" if util_torsion is not None else "n/a")
rows.append({"Check":"Torsion (τ = T·c/J)","Applied":(f"{applied_tau_Pa/1e6:.6f} MPa" if isinstance(applied_tau_Pa, (int,float)) else "n/a"),"Resistance":f"{res_tau_allow_Pa/1e6:.6f} MPa (approx)","Utilization":(f"{util_torsion:.3f}" if util_torsion else "n/a"),"Status":status_tors})

rows.append({"Check":"Bending y-y (σ_by)","Applied":f"{sigma_by_Pa/1e6:.3f} MPa","Resistance":f"{0.6*fy:.3f} MPa","Utilization":(f"{(abs(sigma_by_Pa/1e6)/(0.6*fy)):.3f}" if (0.6*fy)>0 else "n/a"),"Status":"OK" if abs(sigma_by_Pa/1e6)/(0.6*fy)<=1.0 else "EXCEEDS"})
rows.append({"Check":"Bending z-z (σ_bz)","Applied":f"{sigma_bz_Pa/1e6:.3f} MPa","Resistance":f"{0.6*fy:.3f} MPa","Utilization":(f"{(abs(sigma_bz_Pa/1e6)/(0.6*fy)):.3f}" if (0.6*fy)>0 else "n/a"),"Status":"OK" if abs(sigma_bz_Pa/1e6)/(0.6*fy)<=1.0 else "EXCEEDS"})
rows.append({"Check":"Bending & shear (indicative)","Applied":f"σ_by={sigma_by_Pa/1e6:.3f} MPa, τ_eq={tau_total_MPa:.3f} MPa","Resistance":f"{0.6*fy:.3f} MPa","Utilization":(f"{(sigma_eq_MPa/(0.6*fy)):.3f}" if (0.6*fy)>0 else "n/a"),"Status":"OK" if sigma_eq_MPa/(0.6*fy)<=1.0 else "EXCEEDS"})

for axis_label, Ncr, lambda_bar, chi, N_b_Rd_N, status in buck_results:
    if N_b_Rd_N:
        util_buck = abs(N_N) / N_b_Rd_N
        rows.append({"Check":f"Flexural buckling {axis_label}","Applied":f"{abs(N_N)/1e3:.3f} kN","Resistance":f"{N_b_Rd_N/1e3:.3f} kN","Utilization":f"{util_buck:.3f}","Status":"OK" if util_buck<=1.0 else "EXCEEDS"})
    else:
        rows.append({"Check":f"Flexural buckling {axis_label}","Applied":"n/a","Resistance":"n/a","Utilization":"n/a","Status":"n/a"})

rows.append({"Check":"Lateral-torsional buckling (LT)","Applied":"n/a","Resistance":"n/a","Utilization":"n/a","Status":"n/a"})

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
    st.info("Saved runs are kept in this browser session (temporary).")

# Full formulas expanders (optional, same as earlier)
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
- For standard (DB) sections, classification and buckling curve α are taken from your DB and cannot be changed in the UI. For custom sections, the user may enter α and classes.
- Buckling effective length factors (K_y, K_z, K_LT, K_T) are provided in the sidebar and used to compute effective length (K·L) for buckling checks.
- Missing/approximate functionality: exact EN1993 interaction formula variants, detailed local buckling classification tables, full lateral–torsional buckling (M_cr with warping/torsion coupling) in all cases, and national annex values. Replace the approximations with the full EN1993 clauses when integrating final formulas.
- Reference: EN1993-1-1 (Design of steel structures).
""")
