# beam_design_app.py
# EngiSnap — Beam design Eurocode (cross-section placeholder moved before properties, centered)
import streamlit as st
import pandas as pd
import math
from io import BytesIO
from datetime import datetime, date
# --- DB helper: connect to Postgres on Railway using st.secrets ---
import psycopg2
from urllib.parse import urlparse

# -------------------------
# DB connection & loaders
# -------------------------
@st.cache_resource(show_spinner=False)
def get_conn():
    """
    Returns a psycopg2 connection built from st.secrets.
    Assumes you have this in your .streamlit/secrets.toml:
    [postgres]
    host = "amanote.proxy.rlwy.net"
    port = "15500"
    database = "railway"
    user = "postgres"
    password = "KcMoXOMMbbOQITUHrdJMOiwyNBDGyrFy"
    """
    pg = st.secrets["postgres"]
    host = pg["host"]
    port = pg["port"]
    database = pg["database"]
    user = pg["user"]
    password = pg["password"]

    conn = psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password
    )
    return conn

@st.cache_data(show_spinner=False)
def load_beam_db():
    """
    Query DB and return pandas DataFrame for section DB.
    The query returns the columns your app expects (names used later).
    """
    conn = get_conn()
    sql = """
    SELECT
      family, name, A_cm2, S_y_cm3, S_z_cm3,
      I_y_cm4, I_z_cm4, J_cm4, c_max_mm, Wpl_y_cm3, Wpl_z_cm3,
      alpha_curve, flange_class_db, web_class_bending_db, web_class_compression_db,
      COALESCE(Iw_cm6,0) AS Iw_cm6, COALESCE(It_cm4,0) AS It_cm4
    FROM beam_sections
    ORDER BY family, name;
    """
    df = pd.read_sql(sql, conn)
    return df

# -------------------------
# Page setup & load DB (fallback to sample)
# -------------------------
st.set_page_config(page_title="EngiSnap Beam Design Eurocode Checker", layout="wide")
st.title("EngiSnap — Design of steel memebers (Eurocode)")

st.markdown("""
**Disclaimer (read carefully)**  
This prototype performs simplified Eurocode-style screening checks. It is **not** a full EN1993 design package.
Use for screening/prototyping only — always have final results verified by a licensed structural engineer.
""")

# Try to load DB; on failure fall back to small in-memory sample
try:
    df_sample_db = load_beam_db()
    st.sidebar.success(f"Loaded {len(df_sample_db)} sections from DB")
except Exception as e:
    st.sidebar.warning("Could not load DB — using internal sample. DB error: " + str(e))
    sample_rows = [
        {"family": "IPE", "name": "IPE 200",
         "A_cm2": 31.2, "S_y_cm3": 88.0, "S_z_cm3": 16.0,
         "I_y_cm4": 1760.0, "I_z_cm4": 64.0, "J_cm4": 14.0, "c_max_mm": 100.0,
         "Wpl_y_cm3": 96.8, "Wpl_z_cm3": 18.0, "alpha_curve": 0.49,
         "flange_class_db": "Class 1/2", "web_class_bending_db": "Class 2", "web_class_compression_db": "Class 3",
         "Iw_cm6": 2500.0, "It_cm4": 14.0
        }
    ]
    df_sample_db = pd.DataFrame(sample_rows)

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

gamma_M0 = st.sidebar.number_input("γ_M0 (cross-section)", value=1.0,
                                   help="Partial factor γ_M0 (cross-section) — set per project/NA")
gamma_M1 = st.sidebar.number_input("γ_M1 (stability/shear)", value=1.0,
                                   help="Partial factor γ_M1 (stability / shear) — set per project/NA")

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
# Section selection UI (family -> size -> load properties from DB)
# -------------------------
st.header("Section selection")
st.markdown('<span title="Select a standard section from DB (read-only) or use custom.">ⓘ Section selection help</span>',
            unsafe_allow_html=True)

col_left, col_right = st.columns([1,1])
with col_left:
    families = sorted(df_sample_db['family'].dropna().unique().tolist())
    family = st.selectbox("Section family (DB)", options=["-- choose --"] + families,
                          help="Choose section family (database). If none selected, use Custom.")
with col_right:
    selected_name = None
    selected_row = None
    if family and family != "-- choose --":
        df_f = df_sample_db[df_sample_db['family'] == family]
        # names sorted and unique
        names = sorted(df_f['name'].dropna().unique().tolist())
        selected_name = st.selectbox("Section size (DB)", options=["-- choose --"] + names,
                                     help="Choose section size (database). Selecting a size loads read-only properties.")
        if selected_name and selected_name != "-- choose --":
            # pull the row (first match)
            selected_row = df_f[df_f['name'] == selected_name].iloc[0].to_dict()

st.markdown("**Or select Custom**. Standard DB sections are read-only; custom sections are editable.")
use_custom = st.checkbox("Use custom section (enable manual inputs)", help="Tick to enter section properties manually (CUSTOM)")

if selected_row is None and not use_custom:
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
    sr = selected_row or {}

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

    # Fourth row – new Iw and It (warping, torsion inertia)
    c10, c11, c12 = st.columns(3)
    c10.number_input("Iw (cm⁶) (warping)", value=float(sr.get("Iw_cm6", 0.0)), disabled=True, key="db_Iw_cm6")
    c11.number_input("It (cm⁴) (torsion moment of inertia)", value=float(sr.get("It_cm4", 0.0)), disabled=True, key="db_It_cm4")
    c12.empty()  # leave blank for consistent layout

    # Fifth row – class definitions
    cls1, cls2, cls3 = st.columns(3)
    cls1.text_input("Flange class (DB)", value=str(sr.get('flange_class_db', "n/a")), disabled=True, key="db_flange_class_txt")
    cls2.text_input("Web class (bending, DB)", value=str(sr.get('web_class_bending_db', "n/a")), disabled=True, key="db_web_bending_class_txt")
    cls3.text_input("Web class (compression, DB)", value=str(sr.get('web_class_compression_db', "n/a")), disabled=True, key="db_web_comp_class_txt")

    # Sixth row – buckling α
    a1, a2, a3 = st.columns(3)
    alpha_db_val = sr.get('alpha_curve', 0.49)
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
mp2.text_input("Partial factor γ_M0 (cross-sectional)", value=f"{gamma_M0}", disabled=True)
mp3.text_input("Partial factor γ_M1 (buckling / shear)", value=f"{gamma_M1}", disabled=True)
st.markdown("---")

# ------------------------- 
# READY CASES SECTION (unchanged) - keep as in your original code
# -------------------------
# ... (I kept your ready cases and ready catalog exactly; paste that section here unchanged)
# For brevity in this snippet I assume you paste the same "READY CASES" block you already have.
# In your actual file, keep the "READY CASES" code from your original app here.
# -------------------------

# -------------------------
# Loads & inputs
# -------------------------
st.header("Design forces and moments (ultimate state) - INPUT")
st.markdown('<span title="Enter ultimate (ULS) design forces and moments. Positive N = compression.">ⓘ Load input help</span>', unsafe_allow_html=True)

# Row 1 (3 columns)
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

# Row 2 (3 columns)
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

# Row 3 (3 columns) — torsion now on the LEFT side
r3c1, r3c2, r3c3 = st.columns(3)
with r3c1:
    Tx_kNm = st.number_input("Torsion T_x (kN·m)", value=0.0,
                             help="Torsional moment about longitudinal axis (kN·m).")
with r3c2:
    st.write("")  # keep spacing consistent
with r3c3:
    st.write("")  # keep spacing consistent

axis_check = "both (y & z)"

st.sidebar.header("Output mode")
output_mode = st.sidebar.radio("Results output mode", ("Concise (no formulas)", "Full (with formulas & steps)"))
want_pdf = st.sidebar.checkbox("Enable PDF download (requires reportlab)")

# -------------------------
# Choose properties to use (DB vs custom)
# -------------------------
if selected_row is not None and not use_custom:
    # Use the DB row but normalize keys expected by rest of app
    use_props = {
        "family": selected_row.get("family"),
        "name": selected_row.get("name"),
        "A_cm2": selected_row.get("A_cm2", 0.0),
        "S_y_cm3": selected_row.get("S_y_cm3", 0.0),
        "S_z_cm3": selected_row.get("S_z_cm3", 0.0),
        "I_y_cm4": selected_row.get("I_y_cm4", 0.0),
        "I_z_cm4": selected_row.get("I_z_cm4", 0.0),
        "J_cm4": selected_row.get("J_cm4", 0.0),
        "c_max_mm": selected_row.get("c_max_mm", 0.0),
        "Wpl_y_cm3": selected_row.get("Wpl_y_cm3", 0.0),
        "Wpl_z_cm3": selected_row.get("Wpl_z_cm3", 0.0),
        "alpha_curve": selected_row.get("alpha_curve", alpha_default_val),
        "flange_class_db": selected_row.get("flange_class_db", "n/a"),
        "web_class_bending_db": selected_row.get("web_class_bending_db", "n/a"),
        "web_class_compression_db": selected_row.get("web_class_compression_db", "n/a"),
        "Iw_cm6": selected_row.get("Iw_cm6", 0.0),
        "It_cm4": selected_row.get("It_cm4", 0.0)
    }
else:
    # custom (editable) properties: refer to the variables defined earlier in custom UI
    # note: ensure these names match the custom inputs above
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
        "bf_mm": 0.0, "tf_mm": 0.0, "hw_mm": 0.0, "tw_mm": 0.0,
        "flange_class_db": locals().get("flange_class_choice", "Auto (calc)"),
        "web_class_bending_db": locals().get("web_class_bending_choice", "Auto (calc)"),
        "web_class_compression_db": locals().get("web_class_compression_choice", "Auto (calc)"),
        "Iw_cm6": locals().get("Iw_cm6", 0.0),
        "It_cm4": locals().get("It_cm4", 0.0)
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

tau_allow_Pa = 0.6 * sigma_allow_MPa * 1e6

# utilizations
util_axial = abs(N_N) / N_Rd_N if N_Rd_N > 0 else None
util_ten = abs(min(N_N, 0.0)) / T_Rd_N if T_Rd_N > 0 else None
util_My = abs(My_Nm) / M_Rd_y_Nm if M_Rd_y_Nm > 0 else None
util_shear_resultant = math.sqrt(Vy_N**2 + Vz_N**2) / V_Rd_N if V_Rd_N > 0 else None
util_torsion = (tau_torsion_Pa / tau_allow_Pa) if tau_allow_Pa > 0 else None

# -------------------------
# Buckling simplified (using axis-specific K)
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
    Ncr = (math.pi**2 * E * I_axis) / (Leff_axis**2) if Leff_axis > 0 else None
    lambda_bar = math.sqrt((A_m2 * fy * 1e6) / Ncr) if (Ncr and Ncr > 0) else float('inf')
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
# Build checks rows (unchanged) and display (unchanged)
# -------------------------
rows = []
def status_and_util(applied, resistance):
    if resistance is None or resistance == 0:
        return ("n/a", None)
    util = abs(applied) / resistance
    return ("OK" if util <= 1.0 else "EXCEEDS", util)

# Compression
applied_N = N_N if N_N >= 0 else 0.0
res_comp_N = compression_resistance_N
status_comp, util_comp = status_and_util(applied_N, res_comp_N)
rows.append({"Check":"Compression (N≥0)","Applied":f"{applied_N/1e3:.3f} kN","Resistance":(f"{res_comp_N/1e3:.3f} kN" if res_comp_N else "n/a"),"Utilization":(f"{util_comp:.3f}" if util_comp else "n/a"),"Status":status_comp})
# ... (keep the rest of your checks exactly as before)
# For brevity, in this snippet assume you paste the remainder of your original 'rows' construction and presentation code unchanged.

# Display results exactly as you had (the final table, summary, save buttons, expanders, notes, etc.)
# Ensure the rest of your original code (presentation, full-mode expanders, summaries, saves) remains below unchanged.

# NOTE:
# - The critical changes are: caching the DB DataFrame (load_beam_db), mapping selected DB row into `use_props`
#   and keeping custom inputs separate.
# - If you want to show an image from DB, add an 'image_url' column to beam_sections and show it in the cross-section box.
