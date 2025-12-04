# crane_design_app.py
# EngiSnap — Overhead crane calculator (layout only, no real calcs yet)

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

/* Headings */
h1 {font-size: 1.6rem !important; font-weight: 650 !important;}
h2 {font-size: 1.25rem !important; font-weight: 600 !important;}
h3 {font-size: 1.05rem !important; font-weight: 600 !important;}

/* Main container */
div.block-container {
    padding-top: 1.6rem;
    max-width: 1250px;
}

/* Expander look */
.stExpander {
    border-radius: 8px !important;
    border: 1px solid #e0e0e0 !important;
}

/* Labels a bit smaller & bolder */
.stNumberInput > label,
.stTextInput > label,
.stSelectbox > label {
    font-size: 0.85rem;
    font-weight: 500;
}

/* Hide Streamlit default menu & footer */
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
                Layout prototype for static, buckling, serviceability and fatigue checks
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
    # Only set defaults if not already set
    ss = st.session_state

    ss.setdefault("proj_title", "Crane calculation")
    ss.setdefault("proj_name", "")
    ss.setdefault("proj_client", "")
    ss.setdefault("proj_location", "")
    ss.setdefault("proj_revision", "A")
    ss.setdefault("proj_date", date.today())
    ss.setdefault("proj_notes", "")

    # Geometry & classes
    ss.setdefault("crane_type", "Single-girder overhead crane")
    ss.setdefault("span_L", 20.0)
    ss.setdefault("runway_gauge", 15.0)
    ss.setdefault("crane_SWL", 10.0)
    ss.setdefault("crane_self_weight", 12.0)
    ss.setdefault("num_wheels_per_side", 2)
    ss.setdefault("wheel_spacing_end_carriage", 2.5)
    ss.setdefault("wheel_spacing_between_endcarriages", 18.0)

    ss.setdefault("class_utilization", "U5")
    ss.setdefault("class_load_spectrum", "Q3")
    ss.setdefault("crane_duty_class", "FEM 2m / ISO M5")
    ss.setdefault("structure_class", "EN 13001 – HC2")

    # Hoist & end carriages
    ss.setdefault("hoist_mass", 3.0)
    ss.setdefault("trolley_mass", 2.0)
    ss.setdefault("hoist_speed", 8.0)
    ss.setdefault("trolley_speed", 40.0)
    ss.setdefault("crab_min_pos", 0.5)
    ss.setdefault("crab_max_pos", 19.5)
    ss.setdefault("lt_speed", 80.0)
    ss.setdefault("lt_accel", 0.3)
    ss.setdefault("wheel_diameter", 400.0)
    ss.setdefault("wheel_material", "Steel")

    # Loads & combinations
    ss.setdefault("gamma_Q", 1.35)
    ss.setdefault("gamma_G", 1.10)
    ss.setdefault("phi_hoist", 1.1)
    ss.setdefault("phi_LT", 1.1)
    ss.setdefault("phi_CT", 1.1)
    ss.setdefault("fatigue_safety_gamma_Ff", 1.0)
    ss.setdefault("selected_load_case", "LC1 – Hoisting, crab midspan")
    ss.setdefault("serviceability_defl_limit", "L/700")

    # Dummy results placeholders
    ss.setdefault("result_serviceability", None)
    ss.setdefault("result_static", None)
    ss.setdefault("result_buckling", None)
    ss.setdefault("result_fatigue", None)
    ss.setdefault("crane_overall_status", None)

init_default_state()

# -------------------------
# SIDEBAR – WORKFLOW & SNAPSHOT
# -------------------------
with st.sidebar:
    st.markdown("## Workflow")
    st.markdown(
        """
1. **Project info** – basic data  
2. **Geometry & classes** – span, wheels, duty  
3. **Hoist & end carriages** – masses & speeds  
4. **Loads & combinations** – γ, φ, load cases  
5. **Checks & results** – SLS / ULS / fatigue  
6. **Report** – summary & export (later)
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
    st.caption("All checks are placeholders in this version. No real code is hurting itself with EN 13001 yet 😄.")

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
    st.caption("This meta will later show up on the PDF and in the report header.")

# =========================================================
# TAB 2 – GEOMETRY & CLASSES
# =========================================================
def render_tab_geometry_classes():
    st.subheader("Crane geometry & classification")

    # --- 2.1 Crane type & basic geometry ---
    small_title("2.1 Crane type & basic geometry")

    g1, g2, g3 = st.columns(3)
    with g1:
        st.session_state["crane_type"] = st.selectbox(
            "Crane type",
            [
                "Single-girder overhead crane",
                "Double-girder overhead crane",
                "Underslung crane",
                "Gantry crane",
            ],
            index=["Single-girder overhead crane",
                   "Double-girder overhead crane",
                   "Underslung crane",
                   "Gantry crane"].index(st.session_state["crane_type"])
            if st.session_state["crane_type"] in [
                "Single-girder overhead crane",
                "Double-girder overhead crane",
                "Underslung crane",
                "Gantry crane"
            ] else 0,
            key="crane_type_sel",
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
        st.session_state["crane_self_weight"] = st.number_input(
            "Crane self weight (t) (girder + end carriages + trolley)",
            min_value=0.0,
            value=float(st.session_state["crane_self_weight"]),
            step=0.5,
            key="crane_self_weight_in",
        )

    st.markdown("---")

    # --- 2.2 Wheel arrangement ---
    small_title("2.2 Wheel arrangement")

    w1, w2, w3 = st.columns(3)
    with w1:
        st.session_state["num_wheels_per_side"] = st.number_input(
            "Number of wheels per side",
            min_value=1, max_value=8,
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
        st.session_state["wheel_spacing_between_endcarriages"] = st.number_input(
            "Distance between end carriage centres (m)",
            min_value=0.0,
            value=float(st.session_state["wheel_spacing_between_endcarriages"]),
            step=0.1,
            key="wheel_spacing_between_ec_in",
        )

    st.caption("Later we’ll use these to compute wheel loads and runway girder reactions.")

    st.markdown("---")

    # --- 2.3 Classes (EN 13001 / FEM) ---
    small_title("2.3 Crane duty & structure class")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.session_state["crane_duty_class"] = st.selectbox(
            "Duty class (FEM / ISO)",
            ["FEM 1Bm / ISO M3",
             "FEM 2m / ISO M5",
             "FEM 3m / ISO M6",
             "FEM 4m / ISO M8"],
            index=1,
            key="crane_duty_class_sel",
        )
    with c2:
        st.session_state["class_utilization"] = st.selectbox(
            "Class of utilization (U)",
            [f"U{i}" for i in range(1, 9)],
            index=4,
            key="class_utilization_sel",
        )
    with c3:
        st.session_state["class_load_spectrum"] = st.selectbox(
            "Load spectrum class (Q)",
            ["Q1", "Q2", "Q3", "Q4"],
            index=2,
            key="class_load_spectrum_sel",
        )
    with c4:
        st.session_state["structure_class"] = st.selectbox(
            "Structure class (EN 13001)",
            ["HC1", "HC2", "HC3"],
            index=1,
            key="structure_class_sel",
        )

    st.caption(
        "These classes will later drive fatigue load spectra and partial factors "
        "(EN 13001 / FEM)."
    )

# =========================================================
# TAB 3 – HOIST & END CARRIAGES
# =========================================================
def render_tab_hoist_endcarriages():
    st.subheader("Hoist, trolley & end carriages")

    # --- 3.1 Hoist & trolley ---
    small_title("3.1 Hoist & trolley data")

    h1, h2, h3 = st.columns(3)
    with h1:
        st.session_state["hoist_mass"] = st.number_input(
            "Hoist mechanism mass (t)",
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
            step=1.0,
            key="hoist_speed_in",
        )

    h4, h5, h6 = st.columns(3)
    with h4:
        st.session_state["trolley_speed"] = st.number_input(
            "Cross travel speed (m/min)",
            min_value=0.0,
            value=float(st.session_state["trolley_speed"]),
            step=1.0,
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

    # --- 3.2 Long travel (LT) & end carriages ---
    small_title("3.2 Long travel (LT) & end carriage data")

    e1, e2, e3 = st.columns(3)
    with e1:
        st.session_state["lt_speed"] = st.number_input(
            "Long travel speed (m/min)",
            min_value=0.0,
            value=float(st.session_state["lt_speed"]),
            step=1.0,
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
        st.session_state["wheel_diameter"] = st.number_input(
            "Wheel diameter (mm)",
            min_value=0.0,
            value=float(st.session_state["wheel_diameter"]),
            step=10.0,
            key="wheel_diameter_in",
        )

    e4, e5 = st.columns(2)
    with e4:
        st.session_state["wheel_material"] = st.selectbox(
            "Wheel material",
            ["Steel", "Spheroidal graphite iron", "Polyurethane"],
            index=0,
            key="wheel_material_sel",
        )
    with e5:
        st.text_input(
            "Drive arrangement (e.g. 4-wheel drive, one side only)",
            value="",
            key="drive_arrangement_in",
        )

    st.caption("Later we’ll derive dynamic horizontal actions (surge, skew, braking) from these.")

# =========================================================
# TAB 4 – LOADS & COMBINATIONS
# =========================================================
def render_tab_loads_combinations():
    st.subheader("Loads & combinations (input only)")

    # --- 4.1 Partial safety factors & dynamic factors ---
    small_title("4.1 Partial safety factors & dynamic factors")

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
            "φ_hoist (dynamic factor hoisting)",
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
            "φ_LT (dynamic factor long travel)",
            min_value=1.0,
            value=float(st.session_state["phi_LT"]),
            step=0.05,
            key="phi_LT_in",
        )
    with l6:
        st.session_state["phi_CT"] = st.number_input(
            "φ_CT (dynamic factor cross travel)",
            min_value=1.0,
            value=float(st.session_state["phi_CT"]),
            step=0.05,
            key="phi_CT_in",
        )
    with l7:
        st.session_state["serviceability_defl_limit"] = st.selectbox(
            "Serviceability deflection limit for crane girders",
            ["L/500", "L/600", "L/700", "L/800"],
            index=["L/500", "L/600", "L/700", "L/800"].index(
                st.session_state["serviceability_defl_limit"]
            ) if st.session_state["serviceability_defl_limit"] in
                 ["L/500", "L/600", "L/700", "L/800"] else 2,
            key="serviceability_defl_limit_sel",
        )

    st.markdown("---")

    # --- 4.2 Load cases (conceptual) ---
    small_title("4.2 Crane load cases")

    st.session_state["selected_load_case"] = st.selectbox(
        "Select a conceptual load case",
        [
            "LC1 – Hoisting, crab at midspan, no travel",
            "LC2 – Hoisting + LT travel, braking at end stop",
            "LC3 – Hoisting + CT travel, crab near end carriage",
            "LC4 – Parking crane with max crab offset",
            "LC5 – Fatigue spectrum (repeated cycles)",
        ],
        key="selected_load_case_sel",
    )

    st.caption(
        "Later each LC will generate wheel loads, horizontal actions and internal forces "
        "for serviceability, static, buckling and fatigue."
    )

    st.markdown("---")

    # --- 4.3 Manual wheel loads (optional) ---
    with st.expander("4.3 Manual wheel loads (for quick checks)", expanded=False):
        w1, w2, w3, w4 = st.columns(4)
        with w1:
            st.number_input("Max wheel load left rail (kN)", value=0.0, key="W_left_max_in")
        with w2:
            st.number_input("Max wheel load right rail (kN)", value=0.0, key="W_right_max_in")
        with w3:
            st.number_input("Horizontal wheel force (surge) (kN)", value=0.0, key="H_surge_in")
        with w4:
            st.number_input("Horizontal skew force (kN)", value=0.0, key="H_skew_in")

        st.caption("For now this is just storage – no calculation is done with these inputs.")

# =========================================================
# TAB 5 – CHECKS & RESULTS
# =========================================================
def render_tab_checks_results():
    st.subheader("Checks & results (placeholder)")

    st.caption("Here we will later plug in the actual calculation engine.")

    # --- 5.1 Summary cards ---
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric(
            "Serviceability (deflection)",
            value="–",
            delta="pending",
        )
    with s2:
        st.metric(
            "Static resistance",
            value="–",
            delta="pending",
        )
    with s3:
        st.metric(
            "Buckling / stability",
            value="–",
            delta="pending",
        )
    with s4:
        st.metric(
            "Fatigue",
            value="–",
            delta="pending",
        )

    st.markdown("---")

    # --- 5.2 Detailed placeholders ---
    col_sls, col_uls = st.columns(2)

    with col_sls:
        small_title("Serviceability (SLS)")
        st.info(
            "To be implemented: girder deflections, rail misalignment, oscillations, "
            "deflection criteria (e.g. L/700)."
        )

        small_title("Fatigue")
        st.info(
            "To be implemented: stress ranges at critical details, damage sums for "
            "U/Q classes, comparison to allowable damage D ≤ 1.0."
        )

    with col_uls:
        small_title("Static strength (ULS)")
        st.info(
            "To be implemented: member stresses (bending + axial + shear), "
            "local wheel load effects, verification vs resistances."
        )

        small_title("Buckling & global stability")
        st.info(
            "To be implemented: flexural / lateral buckling of runway / crane girders, "
            "interaction with axial force and bending."
        )

    st.markdown("---")
    st.caption("Once the math is ready, this tab will show utilisation tables similar to your beam app.")

# =========================================================
# TAB 6 – REPORT
# =========================================================
def render_tab_report():
    st.subheader("Crane report (skeleton)")

    st.markdown("### 1. Project information")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.text_input(
            "Project name",
            value=st.session_state["proj_name"],
            disabled=True,
            key="rpt_proj_name",
        )
        st.text_input(
            "Client",
            value=st.session_state["proj_client"],
            disabled=True,
            key="rpt_client",
        )
    with c2:
        st.text_input(
            "Document title",
            value=st.session_state["proj_title"],
            disabled=True,
            key="rpt_doc_title",
        )
        st.text_input(
            "Location",
            value=st.session_state["proj_location"],
            disabled=True,
            key="rpt_location",
        )
    with c3:
        st.text_input(
            "Revision",
            value=st.session_state["proj_revision"],
            disabled=True,
            key="rpt_revision",
        )
        st.text_input(
            "Date",
            value=str(st.session_state["proj_date"]),
            disabled=True,
            key="rpt_date",
        )

    st.text_area(
        "Notes / comments",
        value=st.session_state["proj_notes"],
        disabled=True,
        key="rpt_notes",
        height=80,
    )

    st.markdown("---")

    st.markdown("### 2. Crane definition")
    c4, c5, c6 = st.columns(3)
    with c4:
        st.text_input(
            "Crane type",
            value=st.session_state["crane_type"],
            disabled=True,
            key="rpt_crane_type",
        )
        st.text_input(
            "Duty class",
            value=st.session_state["crane_duty_class"],
            disabled=True,
            key="rpt_duty_class",
        )
    with c5:
        st.text_input(
            "Span L [m]",
            value=f"{st.session_state['span_L']:.2f}",
            disabled=True,
            key="rpt_span_L",
        )
        st.text_input(
            "Runway gauge [m]",
            value=f"{st.session_state['runway_gauge']:.2f}",
            disabled=True,
            key="rpt_runway_gauge",
        )
    with c6:
        st.text_input(
            "SWL [t]",
            value=f"{st.session_state['crane_SWL']:.1f}",
            disabled=True,
            key="rpt_SWL",
        )
        st.text_input(
            "Structure class",
            value=st.session_state["structure_class"],
            disabled=True,
            key="rpt_structure_class",
        )

    st.markdown("---")
    st.markdown("### 3. Results overview")

    st.info(
        "In the full version this tab will contain: \n"
        "- SLS / ULS / buckling / fatigue utilisation tables\n"
        "- Governing load case and detail\n"
        "- EN 13001, EN 1993 clause references\n"
        "- PDF export button"
    )

    st.button("💾 Export PDF (coming soon)", disabled=True, key="pdf_dummy_btn")


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
