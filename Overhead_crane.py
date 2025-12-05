# Overhead_crane.py
# EngiSnap — Overhead crane design & assessment (with Excel-based load calculations)

import streamlit as st
from datetime import date

# -------------------------
# PAGE CONFIG & GLOBAL CSS
# -------------------------
st.set_page_config(
    page_title="EngiSnap Overhead Crane Calculator",
    page_icon="EngiSnap-Logo.png",
    layout="wide",
)

custom_css = """
<style>
html, body, [class*="css"]  {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
h1 {font-size: 1.6rem !important; font-weight: 650 !important;}
h2 {font-size: 1.25rem !important; font-weight: 600 !important;}
h3 {font-size: 1.05rem !important; font-weight: 600 !important;}

div.block-container {
    padding-top: 1.6rem;
    max-width: 1250px;
}

.stExpander {
    border-radius: 8px !important;
    border: 1px solid #e0e0e0 !important;
}

.stNumberInput > label,
.stTextInput > label,
.stSelectbox > label {
    font-size: 0.85rem;
    font-weight: 500;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------
header_col1, header_col2 = st.columns([1, 4])

with header_col1:
    try:
        st.image("EngiSnap-Logo.png", width=140)
    except Exception:
        st.write("EngiSnap")

with header_col2:
    st.markdown(
        """
        <div style="padding-top:10px;">
            <div style="font-size:1.5rem;font-weight:650;margin-bottom:0.1rem;">
                EngiSnap — Overhead crane design & assessment
            </div>
            <div style="color:#555;font-size:0.9rem;">
                Layout with main load calculations ported from Excel (97-05-24)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------
# SMALL HELPERS
# -------------------------
def small_title(text: str):
    st.markdown(
        f"<div style='font-weight:600; margin-bottom:6px; font-size:0.95rem;'>{text}</div>",
        unsafe_allow_html=True,
    )


def init_default_state():
    ss = st.session_state

    # --- Project meta ---
    ss.setdefault("proj_title", "Crane calculation")
    ss.setdefault("proj_name", "")
    ss.setdefault("proj_client", "")
    ss.setdefault("proj_location", "")
    ss.setdefault("proj_revision", "A")
    ss.setdefault("proj_date", date.today())
    ss.setdefault("proj_notes", "")

    # --- Geometry & classes (global crane) ---
    ss.setdefault("crane_type", "Double-girder overhead crane")
    ss.setdefault("span_L", 25.0)          # m
    ss.setdefault("runway_gauge", 15.0)    # m

    ss.setdefault("crane_SWL", 65.0)       # t (Excel D13 = 65000 kg)
    # Mass of bridge (without hoist / trolley), t
    ss.setdefault("bridge_mass_t", 24.0)
    # Total end-carriage mass for both sides, t (Excel D32 ~ 1640 kg per EC)
    ss.setdefault("endcar_mass_t", 3.3)

    ss.setdefault("num_wheels_per_side", 2)
    ss.setdefault("wheel_spacing_end_carriage", 5.9)   # m
    ss.setdefault("wheel_spacing_between_endcarriages", 24.0)

    ss.setdefault("crane_duty_class", "FEM 2m / ISO M5")
    ss.setdefault("class_utilization", "U5")
    ss.setdefault("class_load_spectrum", "Q3")
    ss.setdefault("structure_class", "HC2")

    # Hoisting group for φ2 (Excel D8)
    ss.setdefault("hoisting_group", "H2")  # H1/H2/H3/H4-like

    # --- Hoist, trolley, LT ---
    ss.setdefault("hoist_mass", 5.3)       # t (Excel D43 5300 kg)
    ss.setdefault("trolley_mass", 2.0)     # t
    ss.setdefault("hoist_speed", 4.0)      # m/min (Excel D44)
    ss.setdefault("trolley_speed", 16.0)   # m/min

    # Trolley position range along span (for future use)
    ss.setdefault("crab_min_pos", 1.0)
    ss.setdefault("crab_max_pos", 24.0)

    ss.setdefault("lt_speed", 20.0)        # m/min (Excel D20)
    ss.setdefault("lt_accel", 0.5)         # m/s² (Excel D19)

    ss.setdefault("wheel_diameter", 400.0)
    ss.setdefault("wheel_material", "Steel")

    # Hoist wheel arrangement (Excel D45)
    ss.setdefault("hoist_axles", 2)

    # Approaches (Excel D59, D60)
    ss.setdefault("left_approach_mm", 1000.0)
    ss.setdefault("right_approach_mm", 1200.0)

    # --- Buffer & special loads (Excel D39, D40, D131–D132) ---
    ss.setdefault("buffer_type", "CELLULAR ELASTO")   # ELASTIC SPRING / CELLULAR ELASTO / Other
    ss.setdefault("buffer_k", 9.0)                    # kN/m (Excel D40)

    # --- Loads & factors ---
    ss.setdefault("gamma_Q", 1.35)
    ss.setdefault("gamma_G", 1.10)
    ss.setdefault("phi_hoist", 1.1)
    ss.setdefault("phi_LT", 1.1)
    ss.setdefault("phi_CT", 1.1)
    ss.setdefault("fatigue_safety_gamma_Ff", 1.0)
    # Dynamic factor φ1 (Excel D9)
    ss.setdefault("phi1_dynamic", 1.1)

    ss.setdefault("selected_load_case", "LC1 – Hoisting, crab at midspan")
    ss.setdefault("serviceability_defl_limit", "L/700")

    # --- Results (main loads) ---
    ss.setdefault("crane_main_loads", None)

init_default_state()

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.markdown("## Workflow")
    st.markdown(
        """
1. **Project info**  
2. **Geometry & classes**  
3. **Hoist & end carriages**  
4. **Loads & combinations**  
5. **Checks & results**  
6. **Report**
        """
    )
    st.markdown("---")
    st.markdown("## Current setup")

    st.write(f"**Span L:** {st.session_state['span_L']:.2f} m")
    st.write(f"**SWL:** {st.session_state['crane_SWL']:.1f} t")
    st.write(f"**Crane type:** {st.session_state['crane_type']}")
    st.write(f"**Duty class:** {st.session_state['crane_duty_class']}")
    st.write(
        f"**Wheels:** {2 * st.session_state['num_wheels_per_side']} "
        f"({st.session_state['num_wheels_per_side']} per side)"
    )

    st.markdown("---")
    st.markdown("## Quick notes")
    st.caption(
        "Main load formulas (dead load, wheel loads, buffer, test load) "
        "are based on your Excel sheet 97-05-24. Stresses/buckling will be added next."
    )

# =========================================================
# CALCULATION CORE – based on Excel 97-05-24
# =========================================================
def compute_crane_main_loads(ss):
    """
    Port of the 'MAIN LOADS' block from Excel sheet 97-05-24:
    D90–D105 (dead load, wheel loads),
    D131–D132 (buffer),
    D134–D135 (test loads).
    Some parts (bridge self-weight from plate geometry, 'DETAILED' sheet)
    are replaced by explicit mass inputs.
    """
    # --- Basic geometry ---
    span_m = float(ss["span_L"])
    span_mm = span_m * 1000.0 if span_m is not None else 0.0

    # --- Masses (t) ---
    SWL_t = float(ss["crane_SWL"])             # capacity
    bridge_mass_t = float(ss["bridge_mass_t"]) # bridge w/o hoist/trolley
    endcar_mass_t = float(ss["endcar_mass_t"]) # both end carriages total
    trolley_mass_t = float(ss["trolley_mass"])
    hoist_mass_t = float(ss["hoist_mass"])

    # Convert to kN (Excel uses 9.81 * mass[kg] / 1000)
    def mass_t_to_kN(m_t: float) -> float:
        return 9.81 * m_t

    # Dead loads (simplified Excel D92–D94)
    bridge_kN = mass_t_to_kN(bridge_mass_t)
    endcar_kN = mass_t_to_kN(endcar_mass_t)
    dead_total_kN = bridge_kN + endcar_kN       # Excel also adds panels/others via D63/D64
    dead_per_m_kN = (1000.0 * dead_total_kN / span_mm) if span_mm > 0 else 0.0

    # Live loads: trolley + hoist load (Excel D96, D97)
    trolley_kN = mass_t_to_kN(trolley_mass_t)
    hoist_load_kN = mass_t_to_kN(SWL_t)  # using capacity as hoisted load

    # --- Dynamic factors ---
    phi1 = float(ss.get("phi1_dynamic", 1.1))  # D9

    hoist_speed = float(ss["hoist_speed"])     # D44 (m/min)
    hoisting_group = ss.get("hoisting_group", "H2")  # D8

    # Excel D10:
    # IF(H1, 1.1+0.13*v/60;
    #    IF(H2, 1.2+0.27*v/60;
    #       IF(H3, 1.3+0.40*v/60;
    #          1.4+0.53*v/60)))
    v_ratio = hoist_speed / 60.0
    if hoisting_group == "H1":
        phi2 = 1.1 + 0.13 * v_ratio
    elif hoisting_group == "H2":
        phi2 = 1.2 + 0.27 * v_ratio
    elif hoisting_group == "H3":
        phi2 = 1.3 + 0.40 * v_ratio
    else:
        phi2 = 1.4 + 0.53 * v_ratio

    # --- Wheel loads (Excel D99, D100, D104, D105) ---
    total_axles = int(ss["num_wheels_per_side"]) * 2  # D18
    hoist_axles = int(ss.get("hoist_axles", 2))       # D45

    left_app = float(ss.get("left_approach_mm", 1000.0))   # D60
    right_app = float(ss.get("right_approach_mm", 1000.0)) # D59
    min_app = min(left_app, right_app)                     # D61

    live_sum = trolley_kN + hoist_load_kN

    if span_mm <= 0:
        span_mm = 1.0  # avoid div/0; shouldn't happen in practice

    # Helper to avoid div/0 for axles
    def safe_div(value, n_axles):
        return value / n_axles if n_axles > 0 else 0.0

    # D99: MAXIMUM DYNAMIC VERTICAL WHEEL LOAD
    max_dyn_wheel = safe_div(
        dead_total_kN / 2.0
        + phi2 * ((span_mm - min_app) / span_mm) * live_sum,
        total_axles,
    )

    # D100: MAXIMUM ACCOMPANYING DYNAMIC VERTICAL WHEEL LOAD (trolley side)
    max_comp_dyn_wheel = safe_div(
        dead_total_kN / 2.0
        + phi2 * (min_app / span_mm) * live_sum,
        hoist_axles,
    )

    # Static versions (same but φ2 = 1.0) → D104, D105
    max_stat_wheel = safe_div(
        dead_total_kN / 2.0
        + ((span_mm - min_app) / span_mm) * live_sum,
        total_axles,
    )

    max_comp_stat_wheel = safe_div(
        dead_total_kN / 2.0
        + (min_app / span_mm) * live_sum,
        hoist_axles,
    )

    # --- Buffer force (Excel D131–D132) ---
    # D131:
    # IF(D39="ELASTIC SPRING",1.25,
    #    IF(D39="CELLULAR ELASTO",1.25,1.6))
    buffer_type = ss.get("buffer_type", "CELLULAR ELASTO")
    if buffer_type == "ELASTIC SPRING":
        phi_buffer = 1.25
    elif buffer_type == "CELLULAR ELASTO":
        phi_buffer = 1.25
    else:
        phi_buffer = 1.6

    buffer_k = float(ss.get("buffer_k", 9.0))  # D40 [kN/m]
    lt_speed = float(ss["lt_speed"])           # D20 [m/min]

    # D132:
    # BUFFER FORCE = (φ_buffer/1000)*(0.7*V) * ( 1000*(D97+D96+D94)*D40/9.81 )^0.5
    if lt_speed > 0 and buffer_k > 0:
        buffer_force = (
            (phi_buffer / 1000.0)
            * (0.7 * lt_speed)
            * (1000.0 * (hoist_load_kN + trolley_kN + dead_total_kN) * buffer_k / 9.81)
            ** 0.5
        )
    else:
        buffer_force = 0.0

    # --- Test loads (Excel D134–D135) ---
    # D134: φ5 = 0.5*(1+φ2)
    phi5 = 0.5 * (1.0 + phi2)

    # D135: LARGE TEST LOAD = (φ5*1.1*D97) + (φ1*D96)
    large_test_load = (phi5 * 1.1 * hoist_load_kN) + (phi1 * trolley_kN)

    return {
        "span_mm": span_mm,
        "dead_total_kN": dead_total_kN,
        "dead_per_m_kN": dead_per_m_kN,
        "trolley_kN": trolley_kN,
        "hoist_load_kN": hoist_load_kN,
        "phi1": phi1,
        "phi2": phi2,
        "max_dyn_wheel_kN": max_dyn_wheel,
        "max_comp_dyn_wheel_kN": max_comp_dyn_wheel,
        "max_stat_wheel_kN": max_stat_wheel,
        "max_comp_stat_wheel_kN": max_comp_stat_wheel,
        "buffer_force_kN": buffer_force,
        "phi5": phi5,
        "large_test_load_kN": large_test_load,
    }

# =========================================================
# TAB 1 – PROJECT INFO
# =========================================================
def render_tab_project_info():
    st.subheader("Project information")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state["proj_title"] = st.text_input(
            "Document title",
            value=st.session_state["proj_title"],
            key="proj_title_in",
        )
        st.session_state["proj_name"] = st.text_input(
            "Project name",
            value=st.session_state["proj_name"],
            key="proj_name_in",
        )
    with c2:
        st.session_state["proj_client"] = st.text_input(
            "Client / End user",
            value=st.session_state["proj_client"],
            key="proj_client_in",
        )
        st.session_state["proj_location"] = st.text_input(
            "Plant / Location",
            value=st.session_state["proj_location"],
            key="proj_location_in",
        )
    with c3:
        st.session_state["proj_revision"] = st.text_input(
            "Revision",
            value=st.session_state["proj_revision"],
            key="proj_revision_in",
        )
        st.session_state["proj_date"] = st.date_input(
            "Date",
            value=st.session_state["proj_date"],
            key="proj_date_in",
        )

    st.session_state["proj_notes"] = st.text_area(
        "Notes / comments",
        value=st.session_state["proj_notes"],
        key="proj_notes_in",
        height=100,
    )

    st.markdown("---")
    st.caption("This meta will later show up in the report / PDF header.")

# =========================================================
# TAB 2 – GEOMETRY & CLASSES
# =========================================================
def render_tab_geometry_classes():
    st.subheader("Crane geometry & classification")

    # --- 2.1 Crane type & geometry ---
    small_title("2.1 Crane type & basic geometry")

    g1, g2, g3 = st.columns(3)
    with g1:
        st.session_state["crane_type"] = st.selectbox(
            "Crane type",
            [
                "Double-girder overhead crane",
                "Single-girder overhead crane",
                "Underslung crane",
                "Gantry crane",
            ],
            key="crane_type_sel",
            index=0,
        )
        st.session_state["span_L"] = st.number_input(
            "Span L (m) – centre to centre of runway",
            min_value=0.0,
            value=float(st.session_state["span_L"]),
            step=0.1,
            key="span_L_in",
        )
    with g2:
        st.session_state["runway_gauge"] = st.number_input(
            "Runway gauge (m) – rail to rail",
            min_value=0.0,
            value=float(st.session_state["runway_gauge"]),
            step=0.1,
            key="runway_gauge_in",
        )
        st.session_state["crane_SWL"] = st.number_input(
            "Rated lifting capacity SWL (t)",
            min_value=0.0,
            value=float(st.session_state["crane_SWL"]),
            step=0.5,
            key="crane_SWL_in",
        )
    with g3:
        st.session_state["bridge_mass_t"] = st.number_input(
            "Bridge mass (without hoist/trolley) (t)",
            min_value=0.0,
            value=float(st.session_state["bridge_mass_t"]),
            step=0.5,
            key="bridge_mass_in",
        )

    st.markdown("---")

    # --- 2.2 Wheel arrangement & end carriages ---
    small_title("2.2 Wheel arrangement & end carriages")

    w1, w2, w3 = st.columns(3)
    with w1:
        st.session_state["num_wheels_per_side"] = st.number_input(
            "Number of wheels per side",
            min_value=1,
            max_value=8,
            value=int(st.session_state["num_wheels_per_side"]),
            step=1,
            key="num_wheels_per_side_in",
        )
    with w2:
        st.session_state["wheel_spacing_end_carriage"] = st.number_input(
            "Wheel spacing in each end carriage (m)",
            min_value=0.0,
            value=float(st.session_state["wheel_spacing_end_carriage"]),
            step=0.1,
            key="wheel_spacing_ec_in",
        )
    with w3:
        st.session_state["endcar_mass_t"] = st.number_input(
            "Total end carriage mass (both) (t)",
            min_value=0.0,
            value=float(st.session_state["endcar_mass_t"]),
            step=0.1,
            key="endcar_mass_in",
        )

    st.caption("Wheel arrangement is used to distribute calculated wheel loads.")

    st.markdown("---")

    # --- 2.3 Classes & hoisting group ---
    small_title("2.3 Duty classes & hoisting group")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.session_state["crane_duty_class"] = st.selectbox(
            "Duty class (FEM / ISO)",
            ["FEM 1Bm / ISO M3", "FEM 2m / ISO M5", "FEM 3m / ISO M6", "FEM 4m / ISO M8"],
            index=1,
            key="crane_duty_sel",
        )
    with c2:
        st.session_state["class_utilization"] = st.selectbox(
            "Class of utilization (U)",
            [f"U{i}" for i in range(1, 9)],
            index=4,
            key="class_util_sel",
        )
    with c3:
        st.session_state["class_load_spectrum"] = st.selectbox(
            "Load spectrum class (Q)",
            ["Q1", "Q2", "Q3", "Q4"],
            index=2,
            key="class_load_sel",
        )
    with c4:
        st.session_state["structure_class"] = st.selectbox(
            "Structure class (EN 13001)",
            ["HC1", "HC2", "HC3"],
            index=1,
            key="structure_class_sel",
        )

    c5, c6, c7 = st.columns(3)
    with c5:
        st.session_state["hoisting_group"] = st.selectbox(
            "Hoisting group (for φ₂)",
            ["H1", "H2", "H3", "H4"],
            index=["H1", "H2", "H3", "H4"].index(st.session_state["hoisting_group"]),
            key="hoisting_group_sel",
        )
    with c6:
        st.session_state["phi1_dynamic"] = st.number_input(
            "φ₁ (dynamic factor for dead load)",
            min_value=1.0,
            value=float(st.session_state["phi1_dynamic"]),
            step=0.05,
            key="phi1_dynamic_in",
        )
    with c7:
        st.session_state["hoist_axles"] = st.number_input(
            "Number of hoist axles",
            min_value=1,
            max_value=4,
            value=int(st.session_state["hoist_axles"]),
            step=1,
            key="hoist_axles_in",
        )

# =========================================================
# TAB 3 – HOIST & END CARRIAGES
# =========================================================
def render_tab_hoist_endcarriages():
    st.subheader("Hoist, trolley & end carriages")

    # --- 3.1 Hoist & trolley ---
    small_title("3.1 Hoist & trolley")

    h1, h2, h3 = st.columns(3)
    with h1:
        st.session_state["hoist_mass"] = st.number_input(
            "Hoist mass (t)",
            min_value=0.0,
            value=float(st.session_state["hoist_mass"]),
            step=0.1,
            key="hoist_mass_in",
        )
    with h2:
        st.session_state["trolley_mass"] = st.number_input(
            "Trolley / crab mass (t)",
            min_value=0.0,
            value=float(st.session_state["trolley_mass"]),
            step=0.1,
            key="trolley_mass_in",
        )
    with h3:
        st.session_state["hoist_speed"] = st.number_input(
            "Hoisting speed (m/min)",
            min_value=0.0,
            value=float(st.session_state["hoist_speed"]),
            step=0.5,
            key="hoist_speed_in",
        )

    h4, h5, h6 = st.columns(3)
    with h4:
        st.session_state["trolley_speed"] = st.number_input(
            "Cross travel speed (m/min)",
            min_value=0.0,
            value=float(st.session_state["trolley_speed"]),
            step=0.5,
            key="trolley_speed_in",
        )
    with h5:
        st.session_state["crab_min_pos"] = st.number_input(
            "Crab min position from left end carriage (m)",
            min_value=0.0,
            value=float(st.session_state["crab_min_pos"]),
            step=0.1,
            key="crab_min_pos_in",
        )
    with h6:
        st.session_state["crab_max_pos"] = st.number_input(
            "Crab max position from left end carriage (m)",
            min_value=0.0,
            value=float(st.session_state["crab_max_pos"]),
            step=0.1,
            key="crab_max_pos_in",
        )

    st.markdown("---")

    # --- 3.2 Long travel, approaches & buffer ---
    small_title("3.2 Long travel, approaches & buffer")

    e1, e2, e3 = st.columns(3)
    with e1:
        st.session_state["lt_speed"] = st.number_input(
            "Long travel speed (m/min)",
            min_value=0.0,
            value=float(st.session_state["lt_speed"]),
            step=0.5,
            key="lt_speed_in",
        )
    with e2:
        st.session_state["lt_accel"] = st.number_input(
            "Long travel nominal acceleration (m/s²)",
            min_value=0.0,
            value=float(st.session_state["lt_accel"]),
            step=0.05,
            key="lt_accel_in",
        )
    with e3:
        st.session_state["buffer_k"] = st.number_input(
            "Buffer spring stiffness k (kN/m)",
            min_value=0.0,
            value=float(st.session_state["buffer_k"]),
            step=0.5,
            key="buffer_k_in",
        )

    e4, e5, e6 = st.columns(3)
    with e4:
        st.session_state["buffer_type"] = st.selectbox(
            "Buffer type",
            ["ELASTIC SPRING", "CELLULAR ELASTO", "Other"],
            index=["ELASTIC SPRING", "CELLULAR ELASTO", "Other"].index(
                st.session_state["buffer_type"]
            ),
            key="buffer_type_sel",
        )
    with e5:
        st.session_state["left_approach_mm"] = st.number_input(
            "Left approach (mm)",
            min_value=0.0,
            value=float(st.session_state["left_approach_mm"]),
            step=10.0,
            key="left_approach_in",
        )
    with e6:
        st.session_state["right_approach_mm"] = st.number_input(
            "Right approach (mm)",
            min_value=0.0,
            value=float(st.session_state["right_approach_mm"]),
            step=10.0,
            key="right_approach_in",
        )

# =========================================================
# TAB 4 – LOADS & COMBINATIONS
# =========================================================
def render_tab_loads_combinations():
    st.subheader("Loads & combinations")

    # --- 4.1 Partial safety & dynamic factors (global EN factors, separate from Excel φ1/φ2) ---
    small_title("4.1 Partial safety & dynamic factors")

    l1, l2, l3, l4 = st.columns(4)
    with l1:
        st.session_state["gamma_Q"] = st.number_input(
            "γ_Q (variable / live loads)",
            min_value=1.0,
            value=float(st.session_state["gamma_Q"]),
            step=0.05,
            key="gamma_Q_in",
        )
    with l2:
        st.session_state["gamma_G"] = st.number_input(
            "γ_G (permanent / dead loads)",
            min_value=1.0,
            value=float(st.session_state["gamma_G"]),
            step=0.05,
            key="gamma_G_in",
        )
    with l3:
        st.session_state["phi_hoist"] = st.number_input(
            "φ (hoisting, global)",
            min_value=1.0,
            value=float(st.session_state["phi_hoist"]),
            step=0.05,
            key="phi_hoist_in",
        )
    with l4:
        st.session_state["fatigue_safety_gamma_Ff"] = st.number_input(
            "γ_Ff (fatigue load factor)",
            min_value=0.8,
            value=float(st.session_state["fatigue_safety_gamma_Ff"]),
            step=0.05,
            key="gamma_Ff_in",
        )

    l5, l6, l7 = st.columns(3)
    with l5:
        st.session_state["phi_LT"] = st.number_input(
            "φ (long travel)",
            min_value=1.0,
            value=float(st.session_state["phi_LT"]),
            step=0.05,
            key="phi_LT_in",
        )
    with l6:
        st.session_state["phi_CT"] = st.number_input(
            "φ (cross travel)",
            min_value=1.0,
            value=float(st.session_state["phi_CT"]),
            step=0.05,
            key="phi_CT_in",
        )
    with l7:
        st.session_state["serviceability_defl_limit"] = st.selectbox(
            "Serviceability deflection limit (crane girder)",
            ["L/500", "L/600", "L/700", "L/800"],
            index=["L/500", "L/600", "L/700", "L/800"].index(
                st.session_state["serviceability_defl_limit"]
            ),
            key="sls_limit_sel",
        )

    st.markdown("---")

    # --- 4.2 Conceptual load cases ---
    small_title("4.2 Crane load cases (conceptual)")

    st.session_state["selected_load_case"] = st.selectbox(
        "Select conceptual load case (for reporting later)",
        [
            "LC1 – Hoisting, crab at midspan",
            "LC2 – Hoisting + LT travel, braking at end",
            "LC3 – Hoisting + CT travel, crab near end carriage",
            "LC4 – Crane parked with max crab offset",
            "LC5 – Fatigue spectrum (repeated cycles)",
        ],
        key="load_case_sel",
    )

    st.markdown("---")

    # --- 4.3 Run Excel-style main load calculations ---
    small_title("4.3 Excel-style main load calculations")

    if st.button("Run crane load calculations (from current inputs)", key="run_calc_btn"):
        st.session_state["crane_main_loads"] = compute_crane_main_loads(st.session_state)

    res = st.session_state.get("crane_main_loads", None)

    if res is None:
        st.info("Press the button above to compute dead loads, wheel loads, buffer and test load.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total dead load D94 [kN]", f"{res['dead_total_kN']:.2f}")
            st.metric("Dead load per meter D95 [kN/m]", f"{res['dead_per_m_kN']:.2f}")
        with c2:
            st.metric("Trolley load D96 [kN]", f"{res['trolley_kN']:.2f}")
            st.metric("Hoisting load D97 [kN]", f"{res['hoist_load_kN']:.2f}")
        with c3:
            st.metric("Dynamic factor φ₂ (D10)", f"{res['phi2']:.3f}")
            st.metric("Test factor φ₅ (D134)", f"{res['phi5']:.3f}")

        st.markdown("#### Wheel loads (per wheel)")
        w1, w2, w3, w4 = st.columns(4)
        with w1:
            st.write("**Max dynamic wheel (D99)**")
            st.write(f"{res['max_dyn_wheel_kN']:.2f} kN")
        with w2:
            st.write("**Companion dynamic wheel (D100)**")
            st.write(f"{res['max_comp_dyn_wheel_kN']:.2f} kN")
        with w3:
            st.write("**Max static wheel (D104)**")
            st.write(f"{res['max_stat_wheel_kN']:.2f} kN")
        with w4:
            st.write("**Companion static wheel (D105)**")
            st.write(f"{res['max_comp_stat_wheel_kN']:.2f} kN")

        st.markdown("#### Special loads")
        s1, s2 = st.columns(2)
        with s1:
            st.write("**Buffer force D132**")
            st.write(f"{res['buffer_force_kN']:.2f} kN")
        with s2:
            st.write("**Large test load D135**")
            st.write(f"{res['large_test_load_kN']:.2f} kN")

# =========================================================
# TAB 5 – CHECKS & RESULTS
# =========================================================
def render_tab_checks_results():
    st.subheader("Checks & results")

    res = st.session_state.get("crane_main_loads", None)

    # Summary metrics
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Serviceability (deflection)", "–", "pending")
    with s2:
        st.metric("Static resistance", "–", "pending")
    with s3:
        st.metric("Buckling / stability", "–", "pending")
    with s4:
        st.metric("Fatigue", "–", "pending")

    st.markdown("---")

    small_title("5.1 Main loads snapshot (from Excel logic)")

    if res is None:
        st.info("No results yet. Go to **Loads & combinations → Run crane load calculations**.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Dead load D94**")
        st.write(f"{res['dead_total_kN']:.2f} kN")
        st.write("**Dead load per m D95**")
        st.write(f"{res['dead_per_m_kN']:.2f} kN/m")
    with c2:
        st.write("**Max dyn wheel D99**")
        st.write(f"{res['max_dyn_wheel_kN']:.2f} kN")
        st.write("**Companion dyn wheel D100**")
        st.write(f"{res['max_comp_dyn_wheel_kN']:.2f} kN")
    with c3:
        st.write("**Buffer force D132**")
        st.write(f"{res['buffer_force_kN']:.2f} kN")
        st.write("**Large test load D135**")
        st.write(f"{res['large_test_load_kN']:.2f} kN")

    st.markdown("---")
    st.info(
        "Next step: add girder cross-section, internal force calculation and "
        "stress/buckling checks so we can match the lower part of the Excel sheet."
    )

# =========================================================
# TAB 6 – REPORT
# =========================================================
def render_tab_report():
    st.subheader("Crane report (skeleton)")

    st.markdown("### 1. Project information")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.text_input("Project name", st.session_state["proj_name"], disabled=True)
        st.text_input("Client", st.session_state["proj_client"], disabled=True)
    with c2:
        st.text_input("Document title", st.session_state["proj_title"], disabled=True)
        st.text_input("Location", st.session_state["proj_location"], disabled=True)
    with c3:
        st.text_input("Revision", st.session_state["proj_revision"], disabled=True)
        st.text_input("Date", str(st.session_state["proj_date"]), disabled=True)

    st.text_area(
        "Notes / comments",
        st.session_state["proj_notes"],
        disabled=True,
        height=80,
    )

    st.markdown("---")
    st.markdown("### 2. Crane definition")

    c4, c5, c6 = st.columns(3)
    with c4:
        st.text_input("Crane type", st.session_state["crane_type"], disabled=True)
        st.text_input("Duty class", st.session_state["crane_duty_class"], disabled=True)
    with c5:
        st.text_input("Span L [m]", f"{st.session_state['span_L']:.2f}", disabled=True)
        st.text_input("Runway gauge [m]", f"{st.session_state['runway_gauge']:.2f}", disabled=True)
    with c6:
        st.text_input("SWL [t]", f"{st.session_state['crane_SWL']:.1f}", disabled=True)
        st.text_input("Structure class", st.session_state["structure_class"], disabled=True)

    st.markdown("---")
    st.markdown("### 3. Main load results (Excel 97-05-24 block)")

    res = st.session_state.get("crane_main_loads", None)
    if res is None:
        st.info("No results yet. Run calculations from the **Loads & combinations** tab.")
    else:
        st.json(
            {
                "D94 dead load [kN]": round(res["dead_total_kN"], 3),
                "D95 dead load per m [kN/m]": round(res["dead_per_m_kN"], 3),
                "D96 trolley load [kN]": round(res["trolley_kN"], 3),
                "D97 hoist load [kN]": round(res["hoist_load_kN"], 3),
                "D99 max dyn wheel [kN]": round(res["max_dyn_wheel_kN"], 3),
                "D100 comp dyn wheel [kN]": round(res["max_comp_dyn_wheel_kN"], 3),
                "D104 max static wheel [kN]": round(res["max_stat_wheel_kN"], 3),
                "D105 comp static wheel [kN]": round(res["max_comp_stat_wheel_kN"], 3),
                "D132 buffer force [kN]": round(res["buffer_force_kN"], 3),
                "D134 φ5": round(res["phi5"], 3),
                "D135 large test load [kN]": round(res["large_test_load_kN"], 3),
            }
        )

    st.button("💾 Export PDF (coming later)", disabled=True)

# =========================================================
# MAIN TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "1) Project info",
        "2) Geometry & classes",
        "3) Hoist & end carriages",
        "4) Loads & combinations",
        "5) Checks & results",
        "6) Report",
    ]
)

with tab1:
    render_tab_project_info()
with tab2:
    render_tab_geometry_classes()
with tab3:
    render_tab_hoist_endcarriages()
with tab4:
    render_tab_loads_combinations()
with tab5:
    render_tab_checks_results()
with tab6:
    render_tab_report()
