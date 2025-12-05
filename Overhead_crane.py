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

    # --- General crane data ---
    ss.setdefault("crane_type", "Double-girder overhead crane")
    ss.setdefault("span_L", 25.0)          # m
    ss.setdefault("runway_gauge", 15.0)    # m

    ss.setdefault("crane_SWL", 65.0)       # t
    ss.setdefault("bridge_mass_t", 24.0)   # bridge w/o hoist/trolley
    ss.setdefault("endcar_mass_t", 3.3)    # both end carriages total

    ss.setdefault("num_wheels_per_side", 2)
    ss.setdefault("wheel_spacing_end_carriage", 5.9)   # m
    ss.setdefault("wheel_spacing_between_endcarriages", 24.0)

    ss.setdefault("crane_duty_class", "FEM 2m / ISO M5")
    ss.setdefault("class_utilization", "U5")
    ss.setdefault("class_load_spectrum", "Q3")
    ss.setdefault("structure_class", "HC2")

    # Hoisting group for φ2
    ss.setdefault("hoisting_group", "H2")  # H1/H2/H3/H4-like

    # --- Hoist, trolley, LT ---
    ss.setdefault("hoist_mass", 5.3)       # t
    ss.setdefault("trolley_mass", 2.0)     # t
    ss.setdefault("hoist_speed", 4.0)      # m/min
    ss.setdefault("trolley_speed", 16.0)   # m/min
    ss.setdefault("crab_min_pos", 1.0)
    ss.setdefault("crab_max_pos", 24.0)

    ss.setdefault("lt_speed", 20.0)        # m/min
    ss.setdefault("lt_accel", 0.5)         # m/s²

    ss.setdefault("wheel_diameter", 400.0)
    ss.setdefault("wheel_material", "Steel")

    # Hoist wheel arrangement
    ss.setdefault("hoist_axles", 2)

    # Approaches
    ss.setdefault("left_approach_mm", 1000.0)
    ss.setdefault("right_approach_mm", 1200.0)

    # --- Buffer & special loads ---
    ss.setdefault("buffer_type", "CELLULAR ELASTO")   # ELASTIC SPRING / CELLULAR ELASTO / Other
    ss.setdefault("buffer_k", 9.0)                    # kN/m

    # --- Loads & factors ---
    ss.setdefault("gamma_Q", 1.35)
    ss.setdefault("gamma_G", 1.10)
    ss.setdefault("phi_hoist", 1.1)
    ss.setdefault("phi_LT", 1.1)
    ss.setdefault("phi_CT", 1.1)
    ss.setdefault("fatigue_safety_gamma_Ff", 1.0)
    ss.setdefault("phi1_dynamic", 1.1)

    ss.setdefault("selected_load_case", "LC1 – Hoisting, crab at midspan")
    ss.setdefault("serviceability_defl_limit", "L/700")

    # --- Detailed girder & other inputs (Excel-style) ---
    ss.setdefault("electrical_panel_kg", 500.0)
    ss.setdefault("other_weight_kg", 0.0)

    ss.setdefault("web_height_mm", 1480.0)
    ss.setdefault("web_thickness_rail_mm", 8.0)
    ss.setdefault("web_thickness_mm", 8.0)

    ss.setdefault("top_flange_width_mm", 490.0)
    ss.setdefault("top_flange_thickness_mm", 20.0)
    ss.setdefault("a_mm", 50.0)

    ss.setdefault("bottom_flange_width_mm", 490.0)
    ss.setdefault("bottom_flange_thickness_mm", 16.0)
    ss.setdefault("b_mm", 50.0)

    ss.setdefault("end_girder_web_height_mm", 1480.0)
    ss.setdefault("stiffener_type", "EA 80X8")
    ss.setdefault("stiffener_no", 4.0)
    ss.setdefault("stiffener_1_from_top_mm", 300.0)
    ss.setdefault("stiffener_2_from_top_mm", 900.0)
    ss.setdefault("stiffener_3_from_top_mm", 1150.0)

    ss.setdefault("rail_type", "50X30")
    ss.setdefault("diaph_thickness_mm", 6.0)
    ss.setdefault("diaph_distance_mm", 2000.0)

    ss.setdefault("girder_material", "S355")
    ss.setdefault("yield_strength_Nmm2", 355.0)
    ss.setdefault("ultimate_strength_Nmm2", 490.0)
    ss.setdefault("E_modulus_Nmm2", 210000.0)

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
2. **General**  
3. **Girder cross-section**  
4. **Hoist & end carriages**  
5. **Loads & combinations**  
6. **Checks & results**  
7. **Report**
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
    dead load, wheel loads, buffer and test loads.
    Bridge self-weight is taken from bridge_mass_t + endcar_mass_t (+ extra weights).
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

    # Other weights in kg (panel, etc.)
    elec_kg = float(ss.get("electrical_panel_kg", 0.0))
    other_kg = float(ss.get("other_weight_kg", 0.0))

    def mass_t_to_kN(m_t: float) -> float:
        return 9.81 * m_t

    def mass_kg_to_kN(m_kg: float) -> float:
        return 9.81 * m_kg / 1000.0

    # Dead loads
    bridge_kN = mass_t_to_kN(bridge_mass_t)
    endcar_kN = mass_t_to_kN(endcar_mass_t)
    elec_kN = mass_kg_to_kN(elec_kg)
    other_kN = mass_kg_to_kN(other_kg)

    dead_total_kN = bridge_kN + endcar_kN + elec_kN + other_kN
    dead_per_m_kN = (1000.0 * dead_total_kN / span_mm) if span_mm > 0 else 0.0

    # Live loads: trolley + hoist load
    trolley_kN = mass_t_to_kN(trolley_mass_t)
    hoist_load_kN = mass_t_to_kN(SWL_t)

    # --- Dynamic factors ---
    phi1 = float(ss.get("phi1_dynamic", 1.1))

    hoist_speed = float(ss["hoist_speed"])     # m/min
    hoisting_group = ss.get("hoisting_group", "H2")

    # Excel-like dynamic factor φ₂ depending on group and speed
    v_ratio = hoist_speed / 60.0
    if hoisting_group == "H1":
        phi2 = 1.1 + 0.13 * v_ratio
    elif hoisting_group == "H2":
        phi2 = 1.2 + 0.27 * v_ratio
    elif hoisting_group == "H3":
        phi2 = 1.3 + 0.40 * v_ratio
    else:
        phi2 = 1.4 + 0.53 * v_ratio

    # --- Wheel loads ---
    total_axles = int(ss["num_wheels_per_side"]) * 2
    hoist_axles = int(ss.get("hoist_axles", 2))

    left_app = float(ss.get("left_approach_mm", 1000.0))
    right_app = float(ss.get("right_approach_mm", 1000.0))
    min_app = min(left_app, right_app)

    live_sum = trolley_kN + hoist_load_kN

    if span_mm <= 0:
        span_mm = 1.0  # avoid div/0

    def safe_div(value, n_axles):
        return value / n_axles if n_axles > 0 else 0.0

    # Max dynamic wheel load
    max_dyn_wheel = safe_div(
        dead_total_kN / 2.0
        + phi2 * ((span_mm - min_app) / span_mm) * live_sum,
        total_axles,
    )

    # Companion dynamic wheel (trolley side)
    max_comp_dyn_wheel = safe_div(
        dead_total_kN / 2.0
        + phi2 * (min_app / span_mm) * live_sum,
        hoist_axles,
    )

    # Static versions (φ₂ = 1.0)
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

    # --- Buffer force ---
    buffer_type = ss.get("buffer_type", "CELLULAR ELASTO")
    if buffer_type == "ELASTIC SPRING":
        phi_buffer = 1.25
    elif buffer_type == "CELLULAR ELASTO":
        phi_buffer = 1.25
    else:
        phi_buffer = 1.6

    buffer_k = float(ss.get("buffer_k", 9.0))  # kN/m
    lt_speed = float(ss["lt_speed"])           # m/min

    if lt_speed > 0 and buffer_k > 0:
        buffer_force = (
            (phi_buffer / 1000.0)
            * (0.7 * lt_speed)
            * (1000.0 * (hoist_load_kN + trolley_kN + dead_total_kN) * buffer_k / 9.81)
            ** 0.5
        )
    else:
        buffer_force = 0.0

    # --- Test loads ---
    phi5 = 0.5 * (1.0 + phi2)
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
        st.text_input(
            "Document title",
            key="proj_title",
            value=st.session_state["proj_title"],
        )
        st.text_input(
            "Project name",
            key="proj_name",
            value=st.session_state["proj_name"],
        )
    with c2:
        st.text_input(
            "Client / End user",
            key="proj_client",
            value=st.session_state["proj_client"],
        )
        st.text_input(
            "Plant / Location",
            key="proj_location",
            value=st.session_state["proj_location"],
        )
    with c3:
        st.text_input(
            "Revision",
            key="proj_revision",
            value=st.session_state["proj_revision"],
        )
        st.date_input(
            "Date",
            key="proj_date",
            value=st.session_state["proj_date"],
        )

    st.text_area(
        "Notes / comments",
        key="proj_notes",
        value=st.session_state["proj_notes"],
        height=100,
    )

    st.markdown("---")
    st.caption("This meta will later show up in the report / PDF header.")

# =========================================================
# TAB 2 – GENERAL (GLOBAL GEOMETRY & CLASSES)
# =========================================================
def render_tab_general():
    st.subheader("General crane data (geometry & classes)")

    # --- 2.1 Crane type & geometry ---
    small_title("2.1 Crane type & basic geometry")

    g1, g2, g3 = st.columns(3)
    with g1:
        st.selectbox(
            "Crane type",
            [
                "Double-girder overhead crane",
                "Single-girder overhead crane",
                "Underslung crane",
                "Gantry crane",
            ],
            key="crane_type",
            index=[
                "Double-girder overhead crane",
                "Single-girder overhead crane",
                "Underslung crane",
                "Gantry crane",
            ].index(st.session_state["crane_type"]),
        )
        st.number_input(
            "Span L (m) – centre to centre of runway",
            min_value=0.0,
            key="span_L",
            value=float(st.session_state["span_L"]),
            step=0.1,
        )
    with g2:
        st.number_input(
            "Runway gauge (m) – rail to rail",
            min_value=0.0,
            key="runway_gauge",
            value=float(st.session_state["runway_gauge"]),
            step=0.1,
        )
        st.number_input(
            "Rated lifting capacity SWL (t)",
            min_value=0.0,
            key="crane_SWL",
            value=float(st.session_state["crane_SWL"]),
            step=0.5,
        )
    with g3:
        st.number_input(
            "Bridge mass (without hoist/trolley) (t)",
            min_value=0.0,
            key="bridge_mass_t",
            value=float(st.session_state["bridge_mass_t"]),
            step=0.5,
        )

    st.markdown("---")

    # --- 2.2 Wheel arrangement & end carriages ---
    small_title("2.2 Wheel arrangement & end carriages")

    w1, w2, w3 = st.columns(3)
    with w1:
        st.number_input(
            "Number of wheels per side",
            min_value=1,
            max_value=8,
            key="num_wheels_per_side",
            value=int(st.session_state["num_wheels_per_side"]),
            step=1,
        )
    with w2:
        st.number_input(
            "Wheel spacing in each end carriage (m)",
            min_value=0.0,
            key="wheel_spacing_end_carriage",
            value=float(st.session_state["wheel_spacing_end_carriage"]),
            step=0.1,
        )
    with w3:
        st.number_input(
            "Total end carriage mass (both) (t)",
            min_value=0.0,
            key="endcar_mass_t",
            value=float(st.session_state["endcar_mass_t"]),
            step=0.1,
        )

    st.caption("Wheel arrangement is used to distribute calculated wheel loads.")

    st.markdown("---")

    # --- 2.3 Classes & hoisting group ---
    small_title("2.3 Duty classes & hoisting group")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.selectbox(
            "Duty class (FEM / ISO)",
            ["FEM 1Bm / ISO M3", "FEM 2m / ISO M5", "FEM 3m / ISO M6", "FEM 4m / ISO M8"],
            key="crane_duty_class",
            index=["FEM 1Bm / ISO M3", "FEM 2m / ISO M5", "FEM 3m / ISO M6", "FEM 4m / ISO M8"].index(
                st.session_state["crane_duty_class"]
            ),
        )
    with c2:
        st.selectbox(
            "Class of utilization (U)",
            [f"U{i}" for i in range(1, 9)],
            key="class_utilization",
            index=[f"U{i}" for i in range(1, 9)].index(st.session_state["class_utilization"]),
        )
    with c3:
        st.selectbox(
            "Load spectrum class (Q)",
            ["Q1", "Q2", "Q3", "Q4"],
            key="class_load_spectrum",
            index=["Q1", "Q2", "Q3", "Q4"].index(st.session_state["class_load_spectrum"]),
        )
    with c4:
        st.selectbox(
            "Structure class (EN 13001)",
            ["HC1", "HC2", "HC3"],
            key="structure_class",
            index=["HC1", "HC2", "HC3"].index(st.session_state["structure_class"]),
        )

    c5, c6, c7 = st.columns(3)
    with c5:
        st.selectbox(
            "Hoisting group (for φ₂)",
            ["H1", "H2", "H3", "H4"],
            key="hoisting_group",
            index=["H1", "H2", "H3", "H4"].index(st.session_state["hoisting_group"]),
        )
    with c6:
        st.number_input(
            "φ₁ (dynamic factor for dead load)",
            min_value=1.0,
            key="phi1_dynamic",
            value=float(st.session_state["phi1_dynamic"]),
            step=0.05,
        )
    with c7:
        st.number_input(
            "Number of hoist axles",
            min_value=1,
            max_value=4,
            key="hoist_axles",
            value=int(st.session_state["hoist_axles"]),
            step=1,
        )

# =========================================================
# TAB 3 – GIRDER CROSS-SECTION
# =========================================================
def render_tab_girder_section():
    st.subheader("Girder cross-section (Excel-style inputs)")

    st.caption("These match the inputs in sheet 97-05-24 for plate sizes, stiffeners, rail and material.")

    # -----------------------
    # 1. PLATE DIMENSIONS
    # -----------------------
    with st.expander("1) Plate dimensions", expanded=True):

        # Extra weights (panel etc.)
        ow1, ow2 = st.columns(2)
        with ow1:
            st.number_input(
                "Electrical panel weight [kg]",
                key="electrical_panel_kg",
                value=float(st.session_state["electrical_panel_kg"]),
            )
        with ow2:
            st.number_input(
                "Other weight [kg]",
                key="other_weight_kg",
                value=float(st.session_state["other_weight_kg"]),
            )

        g1, g2, g3 = st.columns(3)

        with g1:
            st.number_input(
                "Web height [mm]",
                key="web_height_mm",
                value=float(st.session_state["web_height_mm"]),
            )
            st.number_input(
                "Web thickness at rail [mm]",
                key="web_thickness_rail_mm",
                value=float(st.session_state["web_thickness_rail_mm"]),
            )
            st.number_input(
                "Web thickness (other) [mm]",
                key="web_thickness_mm",
                value=float(st.session_state["web_thickness_mm"]),
            )

        with g2:
            st.number_input(
                "Top flange width [mm]",
                key="top_flange_width_mm",
                value=float(st.session_state["top_flange_width_mm"]),
            )
            st.number_input(
                "Top flange thickness [mm]",
                key="top_flange_thickness_mm",
                value=float(st.session_state["top_flange_thickness_mm"]),
            )
            st.number_input(
                "a [mm] (detail)",
                key="a_mm",
                value=float(st.session_state["a_mm"]),
            )

        with g3:
            st.number_input(
                "Bottom flange width [mm]",
                key="bottom_flange_width_mm",
                value=float(st.session_state["bottom_flange_width_mm"]),
            )
            st.number_input(
                "Bottom flange thickness [mm]",
                key="bottom_flange_thickness_mm",
                value=float(st.session_state["bottom_flange_thickness_mm"]),
            )
            st.number_input(
                "b [mm] (detail)",
                key="b_mm",
                value=float(st.session_state["b_mm"]),
            )

    # -----------------------
    # 2. STIFFENERS
    # -----------------------
    with st.expander("2) Stiffeners", expanded=False):

        s1, s2, s3 = st.columns(3)

        with s1:
            st.text_input(
                "Stiffener type",
                key="stiffener_type",
                value=st.session_state["stiffener_type"],
            )
            st.number_input(
                "Number of stiffeners",
                key="stiffener_no",
                value=float(st.session_state["stiffener_no"]),
                step=1.0,
            )

        with s2:
            st.number_input(
                "Stiffener #1 distance from top [mm]",
                key="stiffener_1_from_top_mm",
                value=float(st.session_state["stiffener_1_from_top_mm"]),
            )
            st.number_input(
                "Stiffener #2 distance from top [mm]",
                key="stiffener_2_from_top_mm",
                value=float(st.session_state["stiffener_2_from_top_mm"]),
            )

        with s3:
            st.number_input(
                "Stiffener #3 distance from top [mm]",
                key="stiffener_3_from_top_mm",
                value=float(st.session_state["stiffener_3_from_top_mm"]),
            )
            st.number_input(
                "End girder web height [mm]",
                key="end_girder_web_height_mm",
                value=float(st.session_state["end_girder_web_height_mm"]),
            )

    # -----------------------
    # 3. RAIL & DIAPHRAGMS
    # -----------------------
    with st.expander("3) Rail & diaphragms", expanded=False):

        r1, r2, r3 = st.columns(3)

        with r1:
            st.text_input(
                "Rail type",
                key="rail_type",
                value=st.session_state["rail_type"],
            )

        with r2:
            st.number_input(
                "Diaphragm thickness [mm]",
                key="diaph_thickness_mm",
                value=float(st.session_state["diaph_thickness_mm"]),
            )

        with r3:
            st.number_input(
                "Diaphragm spacing [mm]",
                key="diaph_distance_mm",
                value=float(st.session_state["diaph_distance_mm"]),
            )

    # -----------------------
    # 4. MATERIAL
    # -----------------------
    with st.expander("4) Material", expanded=False):

        m1, m2, m3 = st.columns(3)

        with m1:
            st.text_input(
                "Material",
                key="girder_material",
                value=st.session_state["girder_material"],
            )

        with m2:
            st.number_input(
                "Yield strength fy [N/mm²]",
                key="yield_strength_Nmm2",
                value=float(st.session_state["yield_strength_Nmm2"]),
            )

        with m3:
            st.number_input(
                "Ultimate strength fu [N/mm²]",
                key="ultimate_strength_Nmm2",
                value=float(st.session_state["ultimate_strength_Nmm2"]),
            )

        st.number_input(
            "Elastic modulus E [N/mm²]",
            key="E_modulus_Nmm2",
            value=float(st.session_state["E_modulus_Nmm2"]),
        )

# =========================================================
# TAB 4 – HOIST & END CARRIAGES
# =========================================================
def render_tab_hoist_endcarriages():
    st.subheader("Hoist, trolley & end carriages")

    # --- 4.1 Hoist & trolley ---
    small_title("4.1 Hoist & trolley")

    h1, h2, h3 = st.columns(3)
    with h1:
        st.number_input(
            "Hoist mass (t)",
            min_value=0.0,
            key="hoist_mass",
            value=float(st.session_state["hoist_mass"]),
            step=0.1,
        )
    with h2:
        st.number_input(
            "Trolley / crab mass (t)",
            min_value=0.0,
            key="trolley_mass",
            value=float(st.session_state["trolley_mass"]),
            step=0.1,
        )
    with h3:
        st.number_input(
            "Hoisting speed (m/min)",
            min_value=0.0,
            key="hoist_speed",
            value=float(st.session_state["hoist_speed"]),
            step=0.5,
        )

    h4, h5, h6 = st.columns(3)
    with h4:
        st.number_input(
            "Cross travel speed (m/min)",
            min_value=0.0,
            key="trolley_speed",
            value=float(st.session_state["trolley_speed"]),
            step=0.5,
        )
    with h5:
        st.number_input(
            "Crab min position from left end carriage (m)",
            min_value=0.0,
            key="crab_min_pos",
            value=float(st.session_state["crab_min_pos"]),
            step=0.1,
        )
    with h6:
        st.number_input(
            "Crab max position from left end carriage (m)",
            min_value=0.0,
            key="crab_max_pos",
            value=float(st.session_state["crab_max_pos"]),
            step=0.1,
        )

    st.markdown("---")

    # --- 4.2 Long travel, approaches & buffer ---
    small_title("4.2 Long travel, approaches & buffer")

    e1, e2, e3 = st.columns(3)
    with e1:
        st.number_input(
            "Long travel speed (m/min)",
            min_value=0.0,
            key="lt_speed",
            value=float(st.session_state["lt_speed"]),
            step=0.5,
        )
    with e2:
        st.number_input(
            "Long travel nominal acceleration (m/s²)",
            min_value=0.0,
            key="lt_accel",
            value=float(st.session_state["lt_accel"]),
            step=0.05,
        )
    with e3:
        st.number_input(
            "Buffer spring stiffness k (kN/m)",
            min_value=0.0,
            key="buffer_k",
            value=float(st.session_state["buffer_k"]),
            step=0.5,
        )

    e4, e5, e6 = st.columns(3)
    with e4:
        st.selectbox(
            "Buffer type",
            ["ELASTIC SPRING", "CELLULAR ELASTO", "Other"],
            key="buffer_type",
            index=["ELASTIC SPRING", "CELLULAR ELASTO", "Other"].index(
                st.session_state["buffer_type"]
            ),
        )
    with e5:
        st.number_input(
            "Left approach (mm)",
            min_value=0.0,
            key="left_approach_mm",
            value=float(st.session_state["left_approach_mm"]),
            step=10.0,
        )
    with e6:
        st.number_input(
            "Right approach (mm)",
            min_value=0.0,
            key="right_approach_mm",
            value=float(st.session_state["right_approach_mm"]),
            step=10.0,
        )

# =========================================================
# TAB 5 – LOADS & COMBINATIONS
# =========================================================
def render_tab_loads_combinations():
    st.subheader("Loads & combinations")

    # --- 5.1 Partial safety & dynamic factors (global EN-like factors) ---
    small_title("5.1 Partial safety & dynamic factors")

    l1, l2, l3, l4 = st.columns(4)
    with l1:
        st.number_input(
            "γ_Q (variable / live loads)",
            min_value=1.0,
            key="gamma_Q",
            value=float(st.session_state["gamma_Q"]),
            step=0.05,
        )
    with l2:
        st.number_input(
            "γ_G (permanent / dead loads)",
            min_value=1.0,
            key="gamma_G",
            value=float(st.session_state["gamma_G"]),
            step=0.05,
        )
    with l3:
        st.number_input(
            "φ (hoisting, global)",
            min_value=1.0,
            key="phi_hoist",
            value=float(st.session_state["phi_hoist"]),
            step=0.05,
        )
    with l4:
        st.number_input(
            "γ_Ff (fatigue load factor)",
            min_value=0.8,
            key="fatigue_safety_gamma_Ff",
            value=float(st.session_state["fatigue_safety_gamma_Ff"]),
            step=0.05,
        )

    l5, l6, l7 = st.columns(3)
    with l5:
        st.number_input(
            "φ (long travel)",
            min_value=1.0,
            key="phi_LT",
            value=float(st.session_state["phi_LT"]),
            step=0.05,
        )
    with l6:
        st.number_input(
            "φ (cross travel)",
            min_value=1.0,
            key="phi_CT",
            value=float(st.session_state["phi_CT"]),
            step=0.05,
        )
    with l7:
        st.selectbox(
            "Serviceability deflection limit (crane girder)",
            ["L/500", "L/600", "L/700", "L/800"],
            key="serviceability_defl_limit",
            index=["L/500", "L/600", "L/700", "L/800"].index(
                st.session_state["serviceability_defl_limit"]
            ),
        )

    st.markdown("---")

    # --- 5.2 Conceptual load cases ---
    small_title("5.2 Crane load cases (conceptual)")

    st.selectbox(
        "Select conceptual load case (for reporting later)",
        [
            "LC1 – Hoisting, crab at midspan",
            "LC2 – Hoisting + LT travel, braking at end",
            "LC3 – Hoisting + CT travel, crab near end carriage",
            "LC4 – Crane parked with max crab offset",
            "LC5 – Fatigue spectrum (repeated cycles)",
        ],
        key="selected_load_case",
    )

    st.markdown("---")

    # --- 5.3 Run Excel-style main load calculations ---
    small_title("5.3 Excel-style main load calculations")

    if st.button("Run crane load calculations (from current inputs)", key="run_calc_btn"):
        st.session_state["crane_main_loads"] = compute_crane_main_loads(st.session_state)

    res = st.session_state.get("crane_main_loads", None)

    if res is None:
        st.info("Press the button above to compute dead loads, wheel loads, buffer and test load.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total dead load [kN]", f"{res['dead_total_kN']:.2f}")
            st.metric("Dead load per meter [kN/m]", f"{res['dead_per_m_kN']:.2f}")
        with c2:
            st.metric("Trolley load [kN]", f"{res['trolley_kN']:.2f}")
            st.metric("Hoisting load [kN]", f"{res['hoist_load_kN']:.2f}")
        with c3:
            st.metric("Dynamic factor φ₂", f"{res['phi2']:.3f}")
            st.metric("Test factor φ₅", f"{res['phi5']:.3f}")

        st.markdown("#### Wheel loads (per wheel)")
        w1, w2, w3, w4 = st.columns(4)
        with w1:
            st.write("**Max dynamic wheel**")
            st.write(f"{res['max_dyn_wheel_kN']:.2f} kN")
        with w2:
            st.write("**Companion dynamic wheel**")
            st.write(f"{res['max_comp_dyn_wheel_kN']:.2f} kN")
        with w3:
            st.write("**Max static wheel**")
            st.write(f"{res['max_stat_wheel_kN']:.2f} kN")
        with w4:
            st.write("**Companion static wheel**")
            st.write(f"{res['max_comp_stat_wheel_kN']:.2f} kN")

        st.markdown("#### Special loads")
        s1, s2 = st.columns(2)
        with s1:
            st.write("**Buffer force**")
            st.write(f"{res['buffer_force_kN']:.2f} kN")
        with s2:
            st.write("**Large test load**")
            st.write(f"{res['large_test_load_kN']:.2f} kN")

# =========================================================
# TAB 6 – CHECKS & RESULTS
# =========================================================
def render_tab_checks_results():
    st.subheader("Checks & results")

    res = st.session_state.get("crane_main_loads", None)

    # Summary metrics placeholders
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

    small_title("6.1 Main loads snapshot (from Excel logic)")

    if res is None:
        st.info("No results yet. Go to **Loads & combinations → Run crane load calculations**.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Dead load**")
        st.write(f"{res['dead_total_kN']:.2f} kN")
        st.write("**Dead load per m**")
        st.write(f"{res['dead_per_m_kN']:.2f} kN/m")
    with c2:
        st.write("**Max dyn wheel**")
        st.write(f"{res['max_dyn_wheel_kN']:.2f} kN")
        st.write("**Companion dyn wheel**")
        st.write(f"{res['max_comp_dyn_wheel_kN']:.2f} kN")
    with c3:
        st.write("**Buffer force**")
        st.write(f"{res['buffer_force_kN']:.2f} kN")
        st.write("**Large test load**")
        st.write(f"{res['large_test_load_kN']:.2f} kN")

    st.markdown("---")
    st.info(
        "Next step: add girder cross-section properties and stress/buckling checks "
        "to match the lower part of your Excel sheet."
    )

# =========================================================
# TAB 7 – REPORT
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
    st.markdown("### 3. Main load results (snapshot)")

    res = st.session_state.get("crane_main_loads", None)
    if res is None:
        st.info("No results yet. Run calculations from the **Loads & combinations** tab.")
    else:
        st.json(
            {
                "Dead load [kN]": round(res["dead_total_kN"], 3),
                "Dead load per m [kN/m]": round(res["dead_per_m_kN"], 3),
                "Trolley load [kN]": round(res["trolley_kN"], 3),
                "Hoist load [kN]": round(res["hoist_load_kN"], 3),
                "Max dynamic wheel [kN]": round(res["max_dyn_wheel_kN"], 3),
                "Companion dynamic wheel [kN]": round(res["max_comp_dyn_wheel_kN"], 3),
                "Max static wheel [kN]": round(res["max_stat_wheel_kN"], 3),
                "Companion static wheel [kN]": round(res["max_comp_stat_wheel_kN"], 3),
                "Buffer force [kN]": round(res["buffer_force_kN"], 3),
                "φ₂": round(res["phi2"], 3),
                "φ₅": round(res["phi5"], 3),
                "Large test load [kN]": round(res["large_test_load_kN"], 3),
            }
        )

    st.button("💾 Export PDF (coming later)", disabled=True)

# =========================================================
# MAIN TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "1) Project info",
        "2) General",
        "3) Girder cross-section",
        "4) Hoist & end carriages",
        "5) Loads & combinations",
        "6) Checks & results",
        "7) Report",
    ]
)

with tab1:
    render_tab_project_info()
with tab2:
    render_tab_general()
with tab3:
    render_tab_girder_section()
with tab4:
    render_tab_hoist_endcarriages()
with tab5:
    render_tab_loads_combinations()
with tab6:
    render_tab_checks_results()
with tab7:
    render_tab_report()
