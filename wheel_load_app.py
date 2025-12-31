# crane_wheel_load_app.py
# EngiSnap — Crane wheel load calculator (EN 1991-3 / BS EN 1991-3:2006)
# Built to match the layout style of the Beam Eurocode Checker app.

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import streamlit as st
from datetime import date

# -----------------------------
# Helpers (same style as beam app)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

def asset_path(rel: str) -> Path:
    return (BASE_DIR / rel).resolve()

def safe_image(path_like, **kwargs) -> bool:
    try:
        pp = Path(path_like)
        if not pp.is_absolute():
            pp = asset_path(str(pp))
        if pp.exists():
            st.image(str(pp), **kwargs)
            return True
    except Exception:
        pass
    return False

def kN_from_kg(m_kg: float) -> float:
    return float(m_kg) * 9.81 / 1000.0

def ms_from_mmin(v_m_per_min: float) -> float:
    return float(v_m_per_min) / 60.0

def mm_to_m(x_mm: float) -> float:
    return float(x_mm) / 1000.0

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def fmt(x: float, unit: str = "", nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{nd}f}{(' ' + unit) if unit else ''}"

def badge(ok: bool) -> str:
    return "✅ OK" if ok else "❌ NOT OK"

# -----------------------------
# Core calculations (from provided report)
# -----------------------------
@dataclass
class WheelLoadInputs:
    # Geometry
    L_mm: float = 14900.0
    e_min_mm: float = 1600.0
    n_wheel_pairs: int = 4               # "n" in the report (wheel pairs)
    n_runway_beams: int = 2              # number of runway beams (rails)
    wheel_spacing_a_mm: float = 6500.0   # a
    rail_clearance_z_mm: float = 22.0    # z (rail/wheel flange clearance)
    rail_width_br_mm: float = 100.0      # b_r (used to compute y = 0.1*b_r)

    # Loads (kg)
    QBr_kg: float = 33700.0  # bridge
    QCr_kg: float = 55900.0  # crane (not directly used in the report equations shown)
    QTr_kg: float = 23000.0  # crab/trolley
    Qh_kg: float = 150000.0  # hoist load
    Qhb_kg: float = 3000.0   # hook block & rope

    # Speeds (m/min)
    v_h_mmin: float = 6.3
    v_br_mmin: float = 20.0
    v_cr_mmin: float = 20.0

    # Drive / friction
    mu: float = 0.20
    m_w: int = 2  # number of single wheel drives

    # Dynamic factors / choices
    phi1_dead: float = 1.10               # dead load factor
    hoisting_class: str = "HC2"           # for phi2_min and beta2
    beta3: float = 0.50                   # 0.5 (slow-release) or 1.0 (rapid-release)
    phi4_rail_tolerance: float = 1.0      # class 1 -> 1.0
    phi5_T: float = 1.5                   # transverse drive factor
    phi5_L: float = 2.0                   # longitudinal drive factor

    # Tests
    eta_dyn: float = 1.10                 # dynamic test coefficient
    eta_stat: float = 1.25                # static test coefficient

    # Buffer
    xi_b: float = 1.0                     # buffer characteristic
    S_B_kN_per_m: float = 300.0           # buffer spring constant

    # Guidance distances e_i (mm) of wheel pairs from guidance means
    # Example from report: 0, 2200, 4300, a
    e_list_mm: Tuple[float, ...] = (0.0, 2200.0, 4300.0, 6500.0)


def phi2_from_hoisting_class(hoisting_class: str, v_h_ms: float) -> Tuple[float, float, float]:
    # From the table shown in the report (page 7)
    # HC1: beta2=0.17, phi2_min=1.05
    # HC2: beta2=0.34, phi2_min=1.10
    # HC3: beta2=0.51, phi2_min=1.15
    # HC4: beta2=0.68, phi2_min=1.20
    table = {
        "HC1": (0.17, 1.05),
        "HC2": (0.34, 1.10),
        "HC3": (0.51, 1.15),
        "HC4": (0.68, 1.20),
    }
    beta2, phi2_min = table.get(hoisting_class.upper(), table["HC2"])
    phi2_raw = phi2_min + beta2 * v_h_ms
    # The report uses: phi2 = max(1.15, phi2_raw)
    phi2 = max(1.15, phi2_raw)
    return phi2, beta2, phi2_min

def phi6_dynamic_test(phi2: float) -> float:
    # Report: phi6,d = 0.5 * (1 + phi2)
    return 0.5 * (1.0 + phi2)

def phi7_buffer(xi_b: float) -> float:
    # Report: if 0<=xi_b<=0.5 -> 1.6? (special), else 1.25 + 0.7*(xi_b-0.5)
    # The figure in the report uses:
    #   if 0<=xi_b<=0.5 => 1.6 (for xi_b=0.5 gives 1.6) then else => 1.25 + 0.7*(xi_b-0.5)
    # To match the shown equation box:
    if xi_b <= 0.5:
        return 1.6
    return 1.25 + 0.7 * (xi_b - 0.5)

def compute_all(inp: WheelLoadInputs) -> Dict[str, Any]:
    # Convert units
    L = mm_to_m(inp.L_mm)
    e = mm_to_m(inp.e_min_mm)
    a = mm_to_m(inp.wheel_spacing_a_mm)

    QBr = kN_from_kg(inp.QBr_kg)
    QTr = kN_from_kg(inp.QTr_kg)
    Qh  = kN_from_kg(inp.Qh_kg)
    Qhb = kN_from_kg(inp.Qhb_kg)

    v_h = ms_from_mmin(inp.v_h_mmin)
    v_br = ms_from_mmin(inp.v_br_mmin)

    n = max(1, int(inp.n_wheel_pairs))
    nr = max(1, int(inp.n_runway_beams))

    # ---- Static wheel loads (report page 6) ----
    Qr_max = (1/n) * (QBr/2 + (L - e)/L * (QTr + Qh))
    Qra_max = (1/n) * (QBr/2 + (e/L) * (QTr + Qh))
    Qr_min = (1/n) * (QBr/2 + (e/L) * (QTr))
    Qra_min = (1/n) * (QBr/2 + (L - e)/L * (QTr))

    # ---- Dynamic factors (report pages 6–9) ----
    phi1 = float(inp.phi1_dead)
    phi2, beta2, phi2_min = phi2_from_hoisting_class(inp.hoisting_class, v_h)

    # rapid load release phi3 (report page 8)
    # Δm: released hoisting mass (here taken as hoist load), m: total hoisting mass (hoist + hook block)
    delta_m = inp.Qh_kg
    m_total = inp.Qh_kg + inp.Qhb_kg
    phi3 = 1.0 - (delta_m / m_total) * (1.0 + float(inp.beta3))

    phi4 = float(inp.phi4_rail_tolerance)
    phi5_T = float(inp.phi5_T)
    phi5_L = float(inp.phi5_L)

    phi6_d = phi6_dynamic_test(phi2)
    phi7 = phi7_buffer(float(inp.xi_b))

    # ---- Vertical load groups (report pages 10–12) ----
    # Group 1 (φ1 on self-weight, φ2 on (QTr+Qh))
    Qr1_max = (1/n) * (phi1*QBr/2 + phi2*(L - e)/L * (QTr + Qh))
    Qr1a_max = (1/n) * (phi1*QBr/2 + phi2*(e/L) * (QTr + Qh))

    # Group 2 (rapid load release on hook block)
    Qr2_max = (1/n) * (phi3*(L - e)/L * Qhb)
    Qr2a_max = (1/n) * (phi3*(e/L) * Qhb)

    # Group 3 (self-weight / unloaded case in report)
    Qr3_max = Qra_min
    Qr3a_max = Qr_min

    # Groups 4–7 (rail tolerance factor)
    Qr4_max = (1/n) * (phi4*QBr/2 + (L - e)/L * (phi4*(QTr + Qh)))
    Qr4a_max = (1/n) * (phi4*QBr/2 + (e/L) * (phi4*(QTr + Qh)))

    # Group 8 (test load)
    Qr8_max_d = (1/n) * (phi1*QBr/2 + (L - e)/L * (phi1*QTr + phi6_d*inp.eta_dyn*Qh))
    Qr8a_max_d = (1/n) * (phi1*QBr/2 + (e/L) * (phi1*QTr + phi6_d*inp.eta_dyn*Qh))

    Qr8_max_s = (1/n) * (phi1*QBr/2 + (L - e)/L * (phi1*QTr + inp.eta_stat*Qh))
    Qr8a_max_s = (1/n) * (phi1*QBr/2 + (e/L) * (phi1*QTr + inp.eta_stat*Qh))

    # Group 9–10 (shown as same as static max/min in report)
    Qr9_max = Qr_max
    Qr9a_max = Qra_max

    # ---- Horizontal forces (report pages 13–16) ----
    # Crane accel/decel
    K = inp.mu * inp.m_w * Qr_min
    H_L = phi5_L * K / nr

    # Skewing (uses static max wheel loads)
    xi1 = Qr_max / (Qr_max + Qra_max) if (Qr_max + Qra_max) > 0 else 0.0
    xi2 = 1.0 - xi1
    Ls = (xi1 - 0.5) * L
    M = K * Ls
    HT1 = phi5_T * xi2 * M / a if a > 0 else 0.0
    HT2 = phi5_T * xi1 * M / a if a > 0 else 0.0

    # Guide force due to skewing
    z = mm_to_m(inp.rail_clearance_z_mm)
    br = mm_to_m(inp.rail_width_br_mm)
    y = 0.1 * br
    alpha_x = max(0.75 * z / a if a > 0 else 0.0, (0.010 / a if a > 0 else 0.0))  # 10 mm / a
    alpha_y = y / a if a > 0 else 0.0
    alpha0 = 0.001
    alpha = min(alpha_x + alpha_y + alpha0, 0.015)
    f = 0.3 * (1.0 - math.exp(-250.0 * alpha))

    e_list = [mm_to_m(v) for v in inp.e_list_mm]
    if len(e_list) == 0:
        e_list = [0.0, a]
    se = sum(e_list)
    se2 = sum([ei**2 for ei in e_list])
    h_a = (se2 / se) if se > 0 else 0.0
    lam = 1.0 - (se / (n * h_a)) if (n * h_a) > 0 else 0.0

    S = f * lam * (QBr + QTr + Qh)

    # Distribution coefficients and guide forces per rail (arrays)
    lam1 = []
    lam2 = []
    HS1 = []
    HS2 = []
    for ei in e_list:
        t = (1.0 - ei / h_a) if h_a > 0 else 0.0
        lam1_i = (xi2 / n) * t
        lam2_i = (xi1 / n) * t
        lam1.append(lam1_i)
        lam2.append(lam2_i)
        HS1.append(f * lam1_i * (QBr + QTr + Qh))
        HS2.append(f * lam2_i * (QBr + QTr + Qh))

    # Trolley accel/decel (10% of (QTr+Qh) shared by trolley wheels)
    n_trolley_wheels = 6  # from report example
    HT3 = 0.1 * (QTr + Qh) / n_trolley_wheels

    # Buffer collision force (report page 16)
    v1 = 0.7 * v_br
    mc_kg = (inp.QBr_kg + inp.QTr_kg + inp.Qh_kg)  # kN -> use masses in kg directly
    # Convert S_B (kN/m) -> N/m
    SB_N_per_m = inp.S_B_kN_per_m * 1000.0
    HB1 = phi7 * v1 * math.sqrt(mc_kg * SB_N_per_m) / 1000.0  # -> kN

    results = {
        "inputs": asdict(inp),
        "units": {"force": "kN", "length": "m", "speed": "m/s"},
        "loads_kN": {"QBr": QBr, "QTr": QTr, "Qh": Qh, "Qhb": Qhb},
        "static": {"Qr_max": Qr_max, "Qra_max": Qra_max, "Qr_min": Qr_min, "Qra_min": Qra_min},
        "factors": {
            "phi1": phi1, "phi2": phi2, "beta2": beta2, "phi2_min": phi2_min,
            "phi3": phi3, "phi4": phi4, "phi5_T": phi5_T, "phi5_L": phi5_L,
            "phi6_d": phi6_d, "phi7": phi7, "v_h_ms": v_h, "v_br_ms": v_br,
        },
        "groups_vertical": {
            "G1_max": Qr1_max, "G1a_max": Qr1a_max,
            "G2_max": Qr2_max, "G2a_max": Qr2a_max,
            "G3_max": Qr3_max, "G3a_max": Qr3a_max,
            "G4_max": Qr4_max, "G4a_max": Qr4a_max,
            "G8_dyn_max": Qr8_max_d, "G8_dyn_a_max": Qr8a_max_d,
            "G8_stat_max": Qr8_max_s, "G8_stat_a_max": Qr8a_max_s,
            "G9_max": Qr9_max, "G9a_max": Qr9a_max,
        },
        "horizontal": {
            "K": K, "H_L": H_L,
            "xi1": xi1, "xi2": xi2, "Ls": Ls, "M": M,
            "HT1": HT1, "HT2": HT2,
            "alpha_x": alpha_x, "alpha_y": alpha_y, "alpha": alpha, "f": f,
            "h_a": h_a, "lambda": lam, "S": S,
            "lambda1": lam1, "lambda2": lam2,
            "HS1": HS1, "HS2": HS2,
            "HT3": HT3,
            "HB1": HB1,
            "HB2": HT3,
        }
    }
    return results

# -----------------------------
# UI
# -----------------------------
def render_sidebar():
    st.sidebar.markdown("### ⚙️ Calculator")
    st.sidebar.caption("EN 1991-3 wheel loads & horizontal actions (based on your attached calculation report).")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Notes**")
    st.sidebar.markdown("- Units: inputs in **mm**, **kg**, **m/min**. Results in **kN**, **m**, **m/s**.")
    st.sidebar.markdown("- This tool reproduces the same equation flow as the report.")
    st.sidebar.markdown("---")

def render_header():
    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

    header_col1, header_col2 = st.columns([1, 4])
    with header_col1:
        safe_image("EngiSnap-Logo.png", width=140) or st.write("")
    with header_col2:
        st.markdown(
            """
            <div style="padding-top:10px;">
                <div style="font-size:1.5rem;font-weight:650;margin-bottom:0.1rem;">
                    EngiSnap — Crane wheel load calculator
                </div>
                <div style="color:#555;font-size:0.9rem;">
                    Actions induced by cranes and machinery (BS EN 1991-3:2006)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def df_from_results(res: Dict[str, Any]) -> pd.DataFrame:
    r = res["static"]
    g = res["groups_vertical"]
    h = res["horizontal"]

    rows = []
    def add(name, value, unit="kN", group="—"):
        rows.append({"Group": group, "Result": name, "Value": value, "Unit": unit})

    # Static
    add("Qr,max (static)", r["Qr_max"], group="Static")
    add("Qr,a,max (static)", r["Qra_max"], group="Static")
    add("Qr,min (static)", r["Qr_min"], group="Static")
    add("Qr,a,min (static)", r["Qra_min"], group="Static")

    # Vertical groups
    add("G1 max", g["G1_max"], group="Vertical")
    add("G1 accompanying", g["G1a_max"], group="Vertical")
    add("G2 max", g["G2_max"], group="Vertical")
    add("G2 accompanying", g["G2a_max"], group="Vertical")
    add("G3 max", g["G3_max"], group="Vertical")
    add("G3 accompanying", g["G3a_max"], group="Vertical")
    add("G4 max", g["G4_max"], group="Vertical")
    add("G4 accompanying", g["G4a_max"], group="Vertical")
    add("G8 dynamic test max", g["G8_dyn_max"], group="Vertical")
    add("G8 dynamic test accompanying", g["G8_dyn_a_max"], group="Vertical")
    add("G8 static test max", g["G8_stat_max"], group="Vertical")
    add("G8 static test accompanying", g["G8_stat_a_max"], group="Vertical")
    add("G9 max", g["G9_max"], group="Vertical")
    add("G9 accompanying", g["G9a_max"], group="Vertical")

    # Horizontal
    add("K (drive force base)", h["K"], group="Horizontal")
    add("H_L (bridge accel/decel per rail)", h["H_L"], group="Horizontal")
    add("H_T1 (skewing, side 1)", h["HT1"], group="Horizontal")
    add("H_T2 (skewing, side 2)", h["HT2"], group="Horizontal")
    add("S (guide force)", h["S"], group="Horizontal")
    add("H_T3 (trolley accel/decel per wheel)", h["HT3"], group="Horizontal")
    add("H_B1 (buffer collision)", h["HB1"], group="Horizontal")
    add("H_B2 (trolley buffer)", h["HB2"], group="Horizontal")

    df = pd.DataFrame(rows)
    df["Value"] = df["Value"].astype(float)
    return df


def render_report_inputs(inp: WheelLoadInputs):
    """Show all user inputs in the same tab-style layout, but read-only (disabled)."""
    st.markdown("## Inputs (read-only)")
    st.caption("These are the exact values used for the calculations below.")

    # --- Project data (mirrors General tab) ---
    st.markdown("### Project data")
    meta_col1, meta_col2, meta_col3 = st.columns([1, 1, 1])

    with meta_col1:
        st.text_input("Document title", value=st.session_state.get("doc_title_in", "Crane wheel load check"), key="r_doc_title_in", disabled=True)
        st.text_input("Project name", value=st.session_state.get("project_name_in", ""), key="r_project_name_in", disabled=True)

    with meta_col2:
        st.text_input("Position / Location (Crane ID)", value=st.session_state.get("position_in", ""), key="r_position_in", disabled=True)
        st.text_input("Requested by", value=st.session_state.get("requested_by_in", ""), key="r_requested_by_in", disabled=True)

    with meta_col3:
        st.text_input("Revision", value=st.session_state.get("revision_in", "A"), key="r_revision_in", disabled=True)
        st.date_input("Date", value=st.session_state.get("run_date_in", date.today()), key="r_run_date_in", disabled=True)

    st.text_area("Notes / comments", value=st.session_state.get("notes_in", ""), key="r_notes_in", disabled=True)

    st.markdown("---")

    # --- Geometry (mirrors Geometry tab) ---
    st.markdown("### Geometry of crane")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Crane span L (mm)", min_value=1.0, value=float(inp.L_mm), step=100.0, key="r_L_mm", disabled=True)
        st.number_input("Minimum hook approach e_min (mm)", min_value=0.0, value=float(inp.e_min_mm), step=50.0, key="r_e_min_mm", disabled=True)
    with c2:
        st.number_input("Number of wheel pairs n", min_value=1, value=int(inp.n_wheel_pairs), step=1, key="r_n_wheel_pairs", disabled=True)
        st.number_input("Number of runway beams (rails) n_r", min_value=1, value=int(inp.n_runway_beams), step=1, key="r_n_runway_beams", disabled=True)
    with c3:
        st.number_input("Wheel spacing a (mm)", min_value=1.0, value=float(inp.wheel_spacing_a_mm), step=50.0, key="r_a_mm", disabled=True)
        st.number_input("Rail/wheel flange clearance z (mm)", min_value=0.0, value=float(inp.rail_clearance_z_mm), step=1.0, key="r_z_mm", disabled=True)
        st.number_input("Rail width b_r (mm)", min_value=1.0, value=float(inp.rail_width_br_mm), step=1.0, key="r_br_mm", disabled=True)

    st.markdown("### Guidance distances (wheel pair to guidance means)")
    st.caption("e₁…e₄ (mm)")
    e_cols = st.columns(4)
    e_list = list(inp.e_list_mm)
    while len(e_list) < 4:
        e_list.append(0.0)
    for i in range(4):
        e_cols[i].number_input(f"e{i+1} (mm)", value=float(e_list[i]), step=100.0, key=f"r_e{i+1}_mm", disabled=True)

    st.markdown("---")

    # --- Loads (mirrors Loads tab) ---
    st.markdown("### Loads (masses)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Bridge mass QBr (kg)", min_value=0.0, value=float(inp.QBr_kg), step=100.0, key="r_QBr_kg", disabled=True)
        st.number_input("Trolley/Crab mass QTr (kg)", min_value=0.0, value=float(inp.QTr_kg), step=100.0, key="r_QTr_kg", disabled=True)
    with c2:
        st.number_input("Hoist load Qh (kg)", min_value=0.0, value=float(inp.Qh_kg), step=500.0, key="r_Qh_kg", disabled=True)
        st.number_input("Hook block & rope mass Qhb (kg)", min_value=0.0, value=float(inp.Qhb_kg), step=50.0, key="r_Qhb_kg", disabled=True)
    with c3:
        st.number_input("Crane mass QCr (kg) (optional)", min_value=0.0, value=float(inp.QCr_kg), step=100.0, key="r_QCr_kg", disabled=True)

    st.markdown("---")

    # --- Speeds (mirrors Speeds tab) ---
    st.markdown("### Speeds")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Hoisting speed v_h (m/min)", min_value=0.0, value=float(inp.v_h_mmin), step=0.1, key="r_v_h", disabled=True)
    with c2:
        st.number_input("Bridge speed v_br (m/min)", min_value=0.0, value=float(inp.v_br_mmin), step=0.5, key="r_v_br", disabled=True)
    with c3:
        st.number_input("Crab speed v_cr (m/min)", min_value=0.0, value=float(inp.v_cr_mmin), step=0.5, key="r_v_cr", disabled=True)

    st.markdown("### Hoisting class (for φ2)")
    st.selectbox("Hoisting class", ["HC1", "HC2", "HC3", "HC4"],
                 index=["HC1", "HC2", "HC3", "HC4"].index(inp.hoisting_class),
                 key="r_hoisting_class", disabled=True)

    st.markdown("---")

    # --- Drives (mirrors Drives tab) ---
    st.markdown("### Drive parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Friction coefficient μ", min_value=0.0, max_value=1.0, value=float(inp.mu), step=0.01, key="r_mu", disabled=True)
        st.number_input("Number of single wheel drives m_w", min_value=0, value=int(inp.m_w), step=1, key="r_m_w", disabled=True)
    with c2:
        st.number_input("Dead load dynamic factor φ1", min_value=0.5, max_value=2.0, value=float(inp.phi1_dead), step=0.05, key="r_phi1", disabled=True)
        st.selectbox("β3 (rapid load release)", [0.5, 1.0], index=0 if float(inp.beta3) == 0.5 else 1, key="r_beta3", disabled=True)
    with c3:
        st.number_input("Rail tolerance factor φ4", min_value=0.5, max_value=2.0, value=float(inp.phi4_rail_tolerance), step=0.05, key="r_phi4", disabled=True)
        st.number_input("Drive factor φ5,T (transverse)", min_value=1.0, max_value=3.0, value=float(inp.phi5_T), step=0.1, key="r_phi5T", disabled=True)
        st.number_input("Drive factor φ5,L (longitudinal)", min_value=1.0, max_value=3.0, value=float(inp.phi5_L), step=0.1, key="r_phi5L", disabled=True)

    st.markdown("### Test & buffer")
    c4, c5, c6 = st.columns(3)
    with c4:
        st.number_input("Dynamic test coefficient η_d", min_value=1.0, max_value=2.0, value=float(inp.eta_dyn), step=0.05, key="r_eta_dyn", disabled=True)
        st.number_input("Static test coefficient η_s", min_value=1.0, max_value=2.0, value=float(inp.eta_stat), step=0.05, key="r_eta_stat", disabled=True)
    with c5:
        st.number_input("Buffer characteristic ξ_b", min_value=0.0, max_value=2.0, value=float(inp.xi_b), step=0.05, key="r_xi_b", disabled=True)
    with c6:
        st.number_input("Buffer spring constant S_B (kN/m)", min_value=1.0, value=float(inp.S_B_kN_per_m), step=10.0, key="r_SB", disabled=True)

    with st.expander("Reference standard (what this tool follows)", expanded=False):
        st.markdown(
            """This calculator follows **BS EN 1991-3:2006 (EN 1991-3)** — *Actions induced by cranes and machinery*.

It covers the typical actions used for runway beam / crane girder design:
- Vertical wheel loads (static + dynamic factors)
- Longitudinal forces (acceleration / braking)
- Transverse / skewing forces + guide forces
- Buffer collision forces (where relevant)"""
        )

def render_report(res: Dict[str, Any]):
    inp = res["inputs"]
    loads = res["loads_kN"]
    fac = res["factors"]
    r = res["static"]
    g = res["groups_vertical"]
    h = res["horizontal"]

    st.markdown("## Detailed calculations")
    st.markdown("### (1) Inputs summary")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Geometry**")
        st.write(f"L = {fmt(inp['L_mm'], 'mm', 0)}")
        st.write(f"e_min = {fmt(inp['e_min_mm'], 'mm', 0)}")
        st.write(f"Wheel pairs n = {inp['n_wheel_pairs']}")
        st.write(f"Runway beams n_r = {inp['n_runway_beams']}")
    with c2:
        st.markdown("**Masses**")
        st.write(f"QBr = {fmt(inp['QBr_kg'], 'kg', 0)} → {fmt(loads['QBr'], 'kN')}")
        st.write(f"QTr = {fmt(inp['QTr_kg'], 'kg', 0)} → {fmt(loads['QTr'], 'kN')}")
        st.write(f"Qh  = {fmt(inp['Qh_kg'], 'kg', 0)} → {fmt(loads['Qh'], 'kN')}")
        st.write(f"Qhb = {fmt(inp['Qhb_kg'], 'kg', 0)} → {fmt(loads['Qhb'], 'kN')}")
    with c3:
        st.markdown("**Speeds & dynamics**")
        st.write(f"v_h = {fmt(inp['v_h_mmin'], 'm/min', 1)} → {fmt(fac['v_h_ms'], 'm/s')}")
        st.write(f"v_br = {fmt(inp['v_br_mmin'], 'm/min', 1)} → {fmt(fac['v_br_ms'], 'm/s')}")
        st.write(f"φ1 = {fac['phi1']:.3f}, φ2 = {fac['phi2']:.3f}, φ3 = {fac['phi3']:.3f}")

    st.markdown("### (2) Static vertical wheel loads")
    st.latex(r"Q_{r,\max}=\frac{1}{n}\left(\frac{Q_{Br}}{2}+\frac{L-e_{\min}}{L}(Q_{Tr}+Q_h)\right)")
    st.write(f"Q_r,max = {fmt(r['Qr_max'], 'kN')}")
    st.latex(r"Q_{r,a,\max}=\frac{1}{n}\left(\frac{Q_{Br}}{2}+\frac{e_{\min}}{L}(Q_{Tr}+Q_h)\right)")
    st.write(f"Q_r,a,max = {fmt(r['Qra_max'], 'kN')}")
    st.latex(r"Q_{r,\min}=\frac{1}{n}\left(\frac{Q_{Br}}{2}+\frac{e_{\min}}{L}(Q_{Tr})\right)")
    st.write(f"Q_r,min = {fmt(r['Qr_min'], 'kN')}")
    st.latex(r"Q_{r,a,\min}=\frac{1}{n}\left(\frac{Q_{Br}}{2}+\frac{L-e_{\min}}{L}(Q_{Tr})\right)")
    st.write(f"Q_r,a,min = {fmt(r['Qra_min'], 'kN')}")

    st.markdown("### (3) Dynamic factors")
    st.write(f"φ1 (dead load) = {fac['phi1']:.3f}")
    st.write(f"Hoisting class = {inp['hoisting_class']} → β₂ = {fac['beta2']:.3f}, φ₂,min = {fac['phi2_min']:.3f}")
    st.latex(r"\varphi_2=\max\left(1.15,\ \varphi_{2,\min}+\beta_2 v_h\right)")
    st.write(f"φ2 = {fac['phi2']:.3f}")
    st.latex(r"\varphi_3=1-\frac{\Delta m}{m}(1+\beta_3)")
    st.write(f"φ3 = {fac['phi3']:.3f}")
    st.latex(r"\varphi_{6,d}=\frac12(1+\varphi_2)")
    st.write(f"φ6,d = {fac['phi6_d']:.3f}")
    st.latex(r"\varphi_7=\begin{cases}1.6 & \xi_b\le 0.5\\ 1.25+0.7(\xi_b-0.5) & \xi_b>0.5\end{cases}")
    st.write(f"φ7 = {fac['phi7']:.3f}")

    st.markdown("### (4) Vertical load groups")
    st.write(f"Group 1 max = {fmt(g['G1_max'], 'kN')}, accompanying = {fmt(g['G1a_max'], 'kN')}")
    st.write(f"Group 2 max = {fmt(g['G2_max'], 'kN')}, accompanying = {fmt(g['G2a_max'], 'kN')}")
    st.write(f"Group 3 max = {fmt(g['G3_max'], 'kN')}, accompanying = {fmt(g['G3a_max'], 'kN')}")
    st.write(f"Group 4 max = {fmt(g['G4_max'], 'kN')}, accompanying = {fmt(g['G4a_max'], 'kN')}")
    st.write(f"Group 8 (dynamic test) max = {fmt(g['G8_dyn_max'], 'kN')}, accompanying = {fmt(g['G8_dyn_a_max'], 'kN')}")
    st.write(f"Group 8 (static test) max = {fmt(g['G8_stat_max'], 'kN')}, accompanying = {fmt(g['G8_stat_a_max'], 'kN')}")

    st.markdown("### (5) Horizontal forces")
    st.latex(r"K=\mu\,m_w\,Q_{r,\min}")
    st.write(f"K = {fmt(h['K'], 'kN')}")
    st.latex(r"H_L=\frac{\varphi_{5,L}\,K}{n_r}")
    st.write(f"H_L = {fmt(h['H_L'], 'kN')} per rail")

    st.latex(r"\xi_1=\frac{Q_{r,\max}}{Q_{r,\max}+Q_{r,a,\max}},\ \xi_2=1-\xi_1")
    st.write(f"ξ1 = {h['xi1']:.3f}, ξ2 = {h['xi2']:.3f}")
    st.latex(r"L_s=(\xi_1-0.5)\,L,\quad M=K\,L_s")
    st.write(f"L_s = {fmt(h['Ls'], 'm')}, M = {fmt(h['M'], 'kN·m')}")
    st.latex(r"H_{T1}=\varphi_{5,T}\,\xi_2\,\frac{M}{a},\quad H_{T2}=\varphi_{5,T}\,\xi_1\,\frac{M}{a}")
    st.write(f"H_T1 = {fmt(h['HT1'], 'kN')}, H_T2 = {fmt(h['HT2'], 'kN')}")

    st.latex(r"f=0.3\left(1-e^{-250\,\alpha}\right),\quad S=f\,\lambda\,(Q_{Br}+Q_{Tr}+Q_h)")
    st.write(f"α = {h['alpha']:.4f}, f = {h['f']:.3f}, λ = {h['lambda']:.3f}")
    st.write(f"Guide force S = {fmt(h['S'], 'kN')}")

    st.markdown("**Guide force distribution (per wheel pair)**")
    df = pd.DataFrame({
        "Wheel pair i": list(range(1, len(h["HS1"]) + 1)),
        "H_S1,i (Rail 1) [kN]": h["HS1"],
        "H_S2,i (Rail 2) [kN]": h["HS2"],
    })
    st.dataframe(df, use_container_width=True)

    st.latex(r"H_{T3}=0.1\frac{(Q_{Tr}+Q_h)}{n_{trolley}}")
    st.write(f"H_T3 = {fmt(h['HT3'], 'kN')} per trolley wheel")

    st.latex(r"H_{B1}=\varphi_7\,v_1\,\sqrt{m_c\,S_B}")
    st.write(f"H_B1 = {fmt(h['HB1'], 'kN')}")

def main():
    st.set_page_config(
        page_title="EngiSnap Crane Wheel Loads (EN 1991-3)",
        page_icon=str(asset_path("EngiSnap-Logo.png")) if asset_path("EngiSnap-Logo.png").exists() else "🧰",
        layout="wide",
    )

    render_header()
    render_sidebar()

    if "wheel_inp" not in st.session_state:
        st.session_state["wheel_inp"] = WheelLoadInputs()

    inp: WheelLoadInputs = st.session_state["wheel_inp"]

    tab_general, tab_geometry, tab_loads, tab_speeds, tab_drives, tab_results, tab_report = st.tabs(
        ["General", "Geometry", "Loads", "Speeds", "Drives", "Results", "Report"]
    )

    with tab_general:
        st.markdown("### Project data")

        meta_col1, meta_col2, meta_col3 = st.columns([1, 1, 1])

        with meta_col1:
            st.text_input("Document title", value=st.session_state.get("doc_title_in", "Crane wheel load check"), key="doc_title_in")
            st.text_input("Project name", value=st.session_state.get("project_name_in", ""), key="project_name_in")

        with meta_col2:
            st.text_input("Position / Location (Crane ID)", value=st.session_state.get("position_in", ""), key="position_in")
            st.text_input("Requested by", value=st.session_state.get("requested_by_in", ""), key="requested_by_in")

        with meta_col3:
            st.text_input("Revision", value=st.session_state.get("revision_in", "A"), key="revision_in")
            st.date_input("Date", value=st.session_state.get("run_date_in", date.today()), key="run_date_in")

        st.text_area("Notes / comments", value=st.session_state.get("notes_in", ""), key="notes_in")

        st.markdown("---")

        with st.expander("Reference standard (what this tool follows)", expanded=False):
            st.markdown(
                """This calculator follows **BS EN 1991-3:2006 (EN 1991-3)** — *Actions induced by cranes and machinery*.

It covers the typical actions used for runway beam / crane girder design:
- Vertical wheel loads (static + dynamic factors)
- Longitudinal forces (acceleration / braking)
- Transverse / skewing forces + guide forces
- Buffer collision forces (where relevant)

If your project uses a different national annex or internal standard, keep the same workflow and swap the factors accordingly."""
            )

        st.info("Fill inputs in the tabs, then go to **Results** or **Report**.")


    with tab_geometry:
        st.markdown("### Geometry of crane")
        c1, c2, c3 = st.columns(3)
        with c1:
            inp.L_mm = st.number_input("Crane span L (mm)", min_value=1.0, value=float(inp.L_mm), step=100.0)
            inp.e_min_mm = st.number_input("Minimum hook approach e_min (mm)", min_value=0.0, value=float(inp.e_min_mm), step=50.0)
        with c2:
            inp.n_wheel_pairs = int(st.number_input("Number of wheel pairs n", min_value=1, value=int(inp.n_wheel_pairs), step=1))
            inp.n_runway_beams = int(st.number_input("Number of runway beams (rails) n_r", min_value=1, value=int(inp.n_runway_beams), step=1))
        with c3:
            inp.wheel_spacing_a_mm = st.number_input("Wheel spacing a (mm)", min_value=1.0, value=float(inp.wheel_spacing_a_mm), step=50.0)
            inp.rail_clearance_z_mm = st.number_input("Rail/wheel flange clearance z (mm)", min_value=0.0, value=float(inp.rail_clearance_z_mm), step=1.0)
            inp.rail_width_br_mm = st.number_input("Rail width b_r (mm)", min_value=1.0, value=float(inp.rail_width_br_mm), step=1.0)

        st.markdown("### Guidance distances (wheel pair to guidance means)")
        st.caption("Use 4 values by default: e₁, e₂, e₃, e₄ (mm).")
        e_cols = st.columns(4)
        e_list = list(inp.e_list_mm)
        while len(e_list) < 4:
            e_list.append(0.0)
        for i in range(4):
            e_list[i] = e_cols[i].number_input(f"e{i+1} (mm)", value=float(e_list[i]), step=100.0)
        inp.e_list_mm = tuple(e_list)

    with tab_loads:
        st.markdown("### Loads (masses)")
        c1, c2, c3 = st.columns(3)
        with c1:
            inp.QBr_kg = st.number_input("Bridge mass QBr (kg)", min_value=0.0, value=float(inp.QBr_kg), step=100.0)
            inp.QTr_kg = st.number_input("Trolley/Crab mass QTr (kg)", min_value=0.0, value=float(inp.QTr_kg), step=100.0)
        with c2:
            inp.Qh_kg = st.number_input("Hoist load Qh (kg)", min_value=0.0, value=float(inp.Qh_kg), step=500.0)
            inp.Qhb_kg = st.number_input("Hook block & rope mass Qhb (kg)", min_value=0.0, value=float(inp.Qhb_kg), step=50.0)
        with c3:
            inp.QCr_kg = st.number_input("Crane mass QCr (kg) (optional)", min_value=0.0, value=float(inp.QCr_kg), step=100.0)
            st.caption("QCr is kept for completeness but not used in the shown report equations.")

    with tab_speeds:
        st.markdown("### Speeds")
        c1, c2, c3 = st.columns(3)
        with c1:
            inp.v_h_mmin = st.number_input("Hoisting speed v_h (m/min)", min_value=0.0, value=float(inp.v_h_mmin), step=0.1)
        with c2:
            inp.v_br_mmin = st.number_input("Bridge speed v_br (m/min)", min_value=0.0, value=float(inp.v_br_mmin), step=0.5)
        with c3:
            inp.v_cr_mmin = st.number_input("Crab speed v_cr (m/min)", min_value=0.0, value=float(inp.v_cr_mmin), step=0.5)

        st.markdown("### Hoisting class (for φ2)")
        inp.hoisting_class = st.selectbox("Hoisting class", ["HC1", "HC2", "HC3", "HC4"], index=["HC1","HC2","HC3","HC4"].index(inp.hoisting_class))

    with tab_drives:
        st.markdown("### Drive parameters")
        c1, c2, c3 = st.columns(3)
        with c1:
            inp.mu = st.number_input("Friction coefficient μ", min_value=0.0, max_value=1.0, value=float(inp.mu), step=0.01)
            inp.m_w = int(st.number_input("Number of single wheel drives m_w", min_value=0, value=int(inp.m_w), step=1))
        with c2:
            inp.phi1_dead = st.number_input("Dead load dynamic factor φ1", min_value=0.5, max_value=2.0, value=float(inp.phi1_dead), step=0.05)
            inp.beta3 = st.selectbox("β3 (rapid load release)", [0.5, 1.0], index=0 if float(inp.beta3)==0.5 else 1)
        with c3:
            inp.phi4_rail_tolerance = st.number_input("Rail tolerance factor φ4", min_value=0.5, max_value=2.0, value=float(inp.phi4_rail_tolerance), step=0.05)
            inp.phi5_T = st.number_input("Drive factor φ5,T (transverse)", min_value=1.0, max_value=3.0, value=float(inp.phi5_T), step=0.1)
            inp.phi5_L = st.number_input("Drive factor φ5,L (longitudinal)", min_value=1.0, max_value=3.0, value=float(inp.phi5_L), step=0.1)

        st.markdown("### Test & buffer")
        c4, c5, c6 = st.columns(3)
        with c4:
            inp.eta_dyn = st.number_input("Dynamic test coefficient η_d", min_value=1.0, max_value=2.0, value=float(inp.eta_dyn), step=0.05)
            inp.eta_stat = st.number_input("Static test coefficient η_s", min_value=1.0, max_value=2.0, value=float(inp.eta_stat), step=0.05)
        with c5:
            inp.xi_b = st.number_input("Buffer characteristic ξ_b", min_value=0.0, max_value=2.0, value=float(inp.xi_b), step=0.05)
        with c6:
            inp.S_B_kN_per_m = st.number_input("Buffer spring constant S_B (kN/m)", min_value=1.0, value=float(inp.S_B_kN_per_m), step=10.0)

    with tab_results:
        st.markdown("### Results")
        res = compute_all(inp)
        df = df_from_results(res)

        # friendly summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Max wheel load (static)", fmt(res["static"]["Qr_max"], "kN"))
        with col2:
            st.metric("Accompanying (static)", fmt(res["static"]["Qra_max"], "kN"))
        with col3:
            st.metric("Guide force S", fmt(res["horizontal"]["S"], "kN"))
        with col4:
            st.metric("Buffer collision HB1", fmt(res["horizontal"]["HB1"], "kN"))

        st.dataframe(df, use_container_width=True)

        st.markdown("### Export (CSV)")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results as CSV", data=csv, file_name="crane_wheel_load_results.csv", mime="text/csv")

    with tab_report:
        res = compute_all(inp)
        render_report_inputs(inp)
        st.markdown('---')
        render_report(res)

    st.session_state["wheel_inp"] = inp

if __name__ == "__main__":
    main()
