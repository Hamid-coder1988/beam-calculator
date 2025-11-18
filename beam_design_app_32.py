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

@st.cache_resource(show_spinner=False)
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
    # Get postgres info from Streamlit secrets
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

def load_beam_db():
    """
    Query DB and return pandas DataFrame for section DB.
    Assumes you have a table 'beam_sections' with columns:
      family, name, A_cm2, S_y_cm3, S_z_cm3,
      I_y_cm4, I_z_cm4, J_cm4, c_max_mm, Wpl_y_cm3, Wpl_z_cm3,
      alpha_curve, flange_class_db, web_class_bending_db, web_class_compression_db,
      Iw_cm6, It_cm4
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

st.set_page_config(page_title="EngiSnap Beam Design Eurocode Checker", layout="wide")
st.title("EngiSnap — Design of steel memebers (Eurocode)")

st.markdown("""
**Disclaimer (read carefully)**  
This prototype performs simplified Eurocode-style screening checks. It is **not** a full EN1993 design package.
Use for screening/prototyping only — always have final results verified by a licensed structural engineer.
""")

# -------------------------
# Sample DB (placeholder)
# -------------------------
# Try to load DB; on failure fall back to small in-memory sample
try:
    df_sample_db = load_beam_db()
    st.sidebar.success(f"Loaded {len(df_sample_db)} sections from DB")
except Exception as e:
    st.sidebar.warning("Could not load DB — using internal sample. DB error: " + str(e))
    # fallback sample (keep minimal sample you used before)
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

# NOTE: per request, design allowable (sigma_allow) removed as editable.
sigma_allow_MPa = 0.6 * fy

gamma_M0 = st.sidebar.number_input("γ_M0 (cross-section)", value=1.0,
                                   help="Partial factor γ_M0 (cross-section) — set per project/NA")
gamma_M1 = st.sidebar.number_input("γ_M1 (stability/shear)", value=1.0,
                                   help="Partial factor γ_M1 (stability / shear) — set per project/NA")

# Buckling K factors (user-supplied)
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
# Section selection UI
# -------------------------
st.header("Section selection")
st.markdown('<span title="Select a standard section from DB (read-only) or use custom.">ⓘ Section selection help</span>',
            unsafe_allow_html=True)

col_left, col_right = st.columns([1,1])
with col_left:
    families = sorted(df_sample_db['family'].unique().tolist())
    family = st.selectbox("Section family (DB)", options=["-- choose --"] + families,
                          help="Choose section family (database). If none selected, use Custom.")
with col_right:
    selected_name = None
    selected_row = None
    if family and family != "-- choose --":
        df_f = df_sample_db[df_sample_db['family'] == family]
        names = sorted(df_f['name'].tolist())
        selected_name = st.selectbox("Section size (DB)", options=["-- choose --"] + names,
                                     help="Choose section size (database). Selecting a size loads read-only properties.")
        if selected_name and selected_name != "-- choose --":
            selected_row = df_f[df_f['name'] == selected_name].iloc[0].to_dict()

st.markdown("**Or select Custom**. Standard DB sections are read-only; custom sections are editable.")
use_custom = st.checkbox("Use custom section (enable manual inputs)", help="Tick to enter section properties manually (CUSTOM)")

if selected_row is None and not use_custom:
    st.info("Please select a section size from the DB above, or tick 'Use custom section' to enter properties manually.")

# -------------------------
# NEW: Cross-section image placeholder (layout frame)
# Moved BEFORE Section properties and CENTERED as requested
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
mp3.text_input("Partial factor γ_M1 (buckling / stability)", value=f"{gamma_M1}", disabled=True)
st.markdown("---")








# ------------------------- 
# READY CASES SECTION (drop-in)
# Put this after "Design properties of material (read-only)" and before the Loads INPUT block
# -------------------------
st.markdown("---")
st.markdown("### Ready beam & frame cases (optional)")
st.write("You can enter loads manually below **or** select a ready beam/frame case to auto-fill typical maxima. First choose whether this is a **Beam** or a **Frame**, then pick a category and case.")

# Option to use ready-cases (single checkbox)
use_ready = st.checkbox("Use ready case (select a template to prefill loads)", key="ready_use_case")

# --- helper case functions (return N_kN, My_kNm, Mz_kNm, Vy_kN, Vz_kN) ---
def ss_udl(span_m: float, w_kN_per_m: float):
    """Simply supported uniform load w (kN/m) on span L: Mmax = w L^2 / 8, Vmax = w L / 2"""
    Mmax = w_kN_per_m * span_m**2 / 8.0
    Vmax = w_kN_per_m * span_m / 2.0
    return (0.0, float(Mmax), 0.0, float(Vmax), 0.0)

def ss_point_center(span_m: float, P_kN: float):
    """Simply supported point load at midspan: Mmax = P L / 4, Vmax = P / 2"""
    Mmax = P_kN * span_m / 4.0
    Vmax = P_kN / 2.0
    return (0.0, float(Mmax), 0.0, float(Vmax), 0.0)

def ss_point_at_a(span_m: float, P_kN: float, a_m: float):
    """Simply supported single point load at distance a from left support:
       Mmax occurs either at point or between supports; for simplicity use M = P * a * (L - a) / L
       Shear at left/right = P * (L - a)/L and P * a / L but for max shear use P
    """
    Mmax = P_kN * a_m * (span_m - a_m) / span_m
    Vmax = P_kN
    return (0.0, float(Mmax), 0.0, float(Vmax), 0.0)

# Minimal catalog example: expand with all your cases later
ready_catalog = {
    "Beam": {
        "Simply supported (examples)": {
            "SS-UDL": {"label": "SS-01: UDL (w on L)", "inputs": {"L": 6.0, "w": 10.0}, "func": ss_udl},
            "SS-Point-Centre": {"label": "SS-02: Point at midspan (P)", "inputs": {"L": 6.0, "P": 20.0}, "func": ss_point_center},
            "SS-Point-a": {"label": "SS-03: Point at distance a (P at a)", "inputs": {"L": 6.0, "P": 20.0, "a": 2.0}, "func": ss_point_at_a},
        },
        # You can add more categories (Cantilever, Fixed ended, Overhang, etc.)
        "Cantilever (examples)": {
            "C-Point": {"label": "C-01: Point at free end", "inputs": {"L": 3.0, "P": 5.0, "a": 3.0}, "func": lambda L,P,a: (0.0, float(P*a), 0.0, float(P), 0.0)},
            "C-UDL": {"label": "C-02: UDL on cantilever", "inputs": {"L": 3.0, "w": 5.0}, "func": lambda L,w: (0.0, float(w*L**2/2.0), 0.0, float(w*L), 0.0)},
            "C-Point-a": {"label": "C-03: Point at a from support", "inputs": {"L": 3.0, "P": 5.0, "a": 1.5}, "func": lambda L,P,a: (0.0, float(P*a), 0.0, float(P), 0.0)},
        }
    },
    "Frame": {
        "Simple frame examples": {
            "F-01": {"label": "FR-01: Simple 2-member frame (placeholder)", "inputs": {"L":4.0, "P":5.0}, "func": lambda L,P: (0.0, float(P*L/4.0), 0.0, float(P/2.0), 0.0)},
            "F-02": {"label": "FR-02: Simple 3-member frame (placeholder)", "inputs": {"L":4.0, "P":8.0}, "func": lambda L,P: (0.0, float(P*L/3.0), 0.0, float(P/2.0), 0.0)},
            "F-03": {"label": "FR-03: Simple frame (placeholder)", "inputs": {"L":5.0, "P":6.0}, "func": lambda L,P: (0.0, float(P*L/4.0), 0.0, float(P/2.0), 0.0)},
        }
    }
}

# UI: choose type and category
if use_ready:
    st.markdown("**Step 1 — choose object type (Beam or Frame)**")
    chosen_type = st.selectbox("Type", options=["-- choose --", "Beam", "Frame"], key="ready_type")
    if chosen_type and chosen_type != "-- choose --":
        categories = sorted(ready_catalog.get(chosen_type, {}).keys())
        if not categories:
            st.warning(f"No ready cases defined yet for {chosen_type}.")
        else:
            chosen_cat = st.selectbox("Category", options=["-- choose --"] + categories, key="ready_category")
            if chosen_cat and chosen_cat != "-- choose --":
                cases_dict = ready_catalog[chosen_type][chosen_cat]
                st.markdown("**Choose a case (click the Select button below the case box)**")
                # display boxes 3 per row
                case_keys = list(cases_dict.keys())
                n_cols = 3
                cols = st.columns(n_cols)
                selected_case_key = None
                for i, ck in enumerate(case_keys):
                    col = cols[i % n_cols]
                    lbl = cases_dict[ck]["label"]
                    # placeholder box (later replace with images)
                    col.markdown(
                        f"""
                        <div style='
                            border:2px solid #bbb;
                            border-radius:10px;
                            padding:18px;
                            text-align:center;
                            background:#fbfbfb;
                            margin-bottom:8px;
                            min-height:84px;
                            display:flex;
                            align-items:center;
                            justify-content:center;
                            font-weight:600;
                        '>
                            {lbl}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if col.button(f"Select {ck}", key=f"ready_select_{chosen_type}_{chosen_cat}_{i}"):
                        selected_case_key = ck

                # store selection in session and rerun so selection panel appears
                if selected_case_key:
                    st.session_state["ready_selected_type"] = chosen_type
                    st.session_state["ready_selected_category"] = chosen_cat
                    st.session_state["ready_selected_case"] = selected_case_key
                    st.rerun()

                # If a case is already selected in session, show its details & inputs
                sel_case = st.session_state.get("ready_selected_case")
                sel_type = st.session_state.get("ready_selected_type")
                sel_cat = st.session_state.get("ready_selected_category")
                if sel_case and sel_type == chosen_type and sel_cat == chosen_cat:
                    scase_info = ready_catalog[sel_type][sel_cat].get(sel_case)
                    if scase_info:
                        st.markdown(f"**Selected case:** {scase_info['label']}")
                        # show inputs for this case (editable)
                        inputs = scase_info.get("inputs", {})
                        input_vals = {}
                        # create inputs with namespaced keys to avoid collisions
                        for k, v in inputs.items():
                            input_vals[k] = st.number_input(f"{k}", value=float(v), key=f"ready_input_{sel_case}_{k}")
                        # Apply & Clear buttons
                        col_apply, col_clear = st.columns([1,1])
                        with col_apply:
                            if st.button("Apply case to load inputs", key=f"ready_apply_{sel_case}"):
                                # call function with args read from same inputs order
                                func = scase_info.get("func")
                                try:
                                    args = [input_vals[k] for k in inputs.keys()]
                                    N_case, My_case, Mz_case, Vy_case, Vz_case = func(*args)
                                except Exception:
                                    # fallback to zeros if signature mismatch
                                    N_case, My_case, Mz_case, Vy_case, Vz_case = 0.0, 0.0, 0.0, 0.0, 0.0
                                # Save prefill values to session_state for your Load INPUT widgets to use
                                st.session_state["prefill_from_case"] = True
                                st.session_state["prefill_N_kN"] = float(N_case)
                                st.session_state["prefill_My_kNm"] = float(My_case)
                                st.session_state["prefill_Mz_kNm"] = float(Mz_case)
                                st.session_state["prefill_Vy_kN"] = float(Vy_case)
                                st.session_state["prefill_Vz_kN"] = float(Vz_case)
                                # store L if present
                                if "L" in input_vals:
                                    st.session_state["case_L"] = float(input_vals["L"])
                                st.success("Case applied — scroll to Design forces and moments (inputs updated).")
                        with col_clear:
                            if st.button("Clear selected case", key=f"ready_clear_{sel_case}"):
                                for k in ("ready_selected_type", "ready_selected_category", "ready_selected_case",
                                          "prefill_from_case","prefill_N_kN","prefill_My_kNm","prefill_Mz_kNm","prefill_Vy_kN","prefill_Vz_kN","case_L"):
                                    if k in st.session_state:
                                        del st.session_state[k]
                                st.success("Selected case cleared.")
                                st.rerun()
    else:
        st.info("Select 'Beam' or 'Frame' to view categorized ready cases.")
else:
    # if not using ready cases: ensure no leftover selected case remains visible unexpectedly
    # (do not automatically delete persisted selection; user can re-enable ready use to restore)
    pass

# ------------------------- 
# End of READY CASES SECTION
# -------------------------


# -------------------------
# Loads & inputs (3 items per line)
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

# Define axis_check (since the old selectbox was removed)
axis_check = "both (y & z)"



# axis_check removed from sidebar per request; default to check both axes
axis_check = "both (y & z)"

st.sidebar.header("Output mode")
output_mode = st.sidebar.radio("Results output mode", ("Concise (no formulas)", "Full (with formulas & steps)"))
want_pdf = st.sidebar.checkbox("Enable PDF download (requires reportlab)")








# -------------------------
# Choose properties to use (DB vs custom)
# -------------------------
if selected_row is not None and not use_custom:
    use_props = dict(selected_row)
    for k in ("Iw_cm6","It_cm4","bf_mm","tf_mm","hw_mm","tw_mm","flange_class_db","web_class_bending_db","web_class_compression_db"):
        use_props.setdefault(k, 0.0 if k.endswith("_cm6") or k.endswith("_cm4") or k.endswith("_mm") else use_props.get(k, "n/a"))
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

# Bending & combined checks (same as before)...
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

# Buckling & interaction rows
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
# Display: Result summary and colored table
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

# -------------------------
# Full mode: expanders with formulas & intermediate values
# -------------------------
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
    with st.expander("Axial–bending interaction (EN1993-1-1 §6.3.3 & Annex A/B)"):
        st.latex(r"\eta = \dfrac{N}{N_{Rd}} + \dfrac{M}{M_{Rd}} \quad\text{(method 1)}")
        st.latex(r"\eta_{2} = \sqrt{\left(\dfrac{N}{N_{Rd}}\right)^2 + \left(\dfrac{M}{M_{Rd}}\right)^2} \quad\text{(method 2)}")
        st.write(f"Interaction (method1 approx) = {interaction_ratio_simple if interaction_ratio_simple else 'n/a'}")
        if interaction_method2 is not None:
            st.write(f"Interaction (method2 approx) = {interaction_method2:.3f}")

# -------------------------
# Summaries & Notes (end)
# -------------------------
st.markdown("---")
st.subheader("Summary & recommended next steps by EngiSnap")
eng_notes = []
if any(df_rows["Status"] == "EXCEEDS"):
    eng_notes.append("One or more checks exceed capacity — consider increasing the section (A / W_pl), reducing applied loads, shortening unbraced length, or adding restraints. Consult a licensed structural engineer for final design.")
else:
    eng_notes.append("Preliminary screening checks OK. Proceed to full EN1993 member checks (local buckling, LTB, full interaction) before final design.")
st.write("\n\n".join(eng_notes))

st.markdown("---")
st.subheader("Summary & recommended next steps by AI")
ai_notes = []
if util_axial is not None and util_axial > 0.85:
    ai_notes.append("Axial utilization high (>85%): review buckling length, increase stiffness or reduce compressive load.")
if util_My is not None and util_My > 0.9:
    ai_notes.append("Bending utilization near limit (>90%): consider larger W_pl or improve lateral restraint.")
if util_shear_resultant is not None and util_shear_resultant > 0.75:
    ai_notes.append("Shear utilization moderately high — check web thickness and connection capacity.")
if util_torsion is not None and util_torsion > 0.3:
    ai_notes.append("Torsion significant relative to allowable — evaluate torsional effects in connections.")
if not ai_notes:
    ai_notes.append("No immediate hotspots detected by these automated rules.")
st.write("\n\n".join(ai_notes))

st.markdown("---")
st.subheader("Notes, limitations & references")
st.write("""
- This tool gives **preliminary** screening checks only. It is not a complete EN1993 implementation.
- For standard (DB) sections, classification and buckling curve α are taken from your DB and cannot be changed in the UI. For custom sections, the user may enter α and classes.
- Buckling effective length factors (K_y, K_z, K_LT, K_T) are provided in the sidebar and used to compute effective length (K·L) for buckling checks.
- Missing/approximate functionality: exact EN1993 interaction formula variants, detailed local buckling classification tables, full lateral–torsional buckling (M_cr with warping/torsion coupling) in all cases, and national annex values. Replace the approximations with the full EN1993 clauses when integrating final formulas.
- Reference: EN1993-1-1 (Design of steel structures).
""")

# second Save results button (end)
end_col1, end_col2 = st.columns([1,1])
with end_col1:
    if st.button("Save results (end)", help="Also save this run to session state"):
        if "saved_results" not in st.session_state:
            st.session_state["saved_results"] = []
        st.session_state["saved_results"].append(build_result_record())
        st.success("Results saved to session state (end button).")
with end_col2:
    st.info("Saved runs are kept in this browser session (temporary).")





