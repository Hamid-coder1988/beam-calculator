# beam_design_app.py
# EngiSnap — Beam Code 6 (DB-backed, wizard UI, gallery ready cases, pro report + PDF)
# + V(x) & M(x) diagrams integrated (Beam-only for now)

import streamlit as st
import pandas as pd
import math
from io import BytesIO
from datetime import datetime, date
import traceback
import re
import numbers
import numpy as np
import matplotlib.pyplot as plt  # NEW

# -------------------------
# Optional Postgres driver
# -------------------------
try:
    import psycopg2
    HAS_PG = True
except Exception:
    psycopg2 = None
    HAS_PG = False

# -------------------------
# Optional PDF engine (reportlab)
# -------------------------
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
    HAS_RL = True
except Exception:
    HAS_RL = False


# =========================================================
# DB CONNECTION (same as Beam Code 3, uses st.secrets)
# =========================================================
def get_conn():
    if not HAS_PG:
        raise RuntimeError("psycopg2 not installed. Install with: pip install psycopg2-binary")
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
        return psycopg2.connect(
            host=host, port=port, database=database,
            user=user, password=password, sslmode=sslmode
        )
    except Exception:
        try:
            from urllib.parse import urlparse
            dburl = st.secrets["DATABASE_URL"]
            parsed = urlparse(dburl)
            return psycopg2.connect(
                dbname=parsed.path.lstrip("/"),
                user=parsed.username,
                password=parsed.password,
                host=parsed.hostname,
                port=parsed.port,
                sslmode="require"
            )
        except Exception as e:
            raise RuntimeError(
                "Database credentials not found in st.secrets. "
                "Add [postgres] or DATABASE_URL to secrets."
            ) from e


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
# Fallback sample rows (if DB not available)
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


# =========================================================
# DB INSPECTION HELPERS
# =========================================================
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

    types, sizes_map = [], {}
    for _, row in rows_df.iterrows():
        t = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
        s = str(row.iloc[1]) if pd.notna(row.iloc[1]) else ""
        if not t:
            continue
        if t not in types:
            types.append(t); sizes_map[t] = []
        if s and s not in sizes_map[t]:
            sizes_map[t].append(s)

    return types, sizes_map, tbl


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
            (f'SELECT * FROM "{table_name}" WHERE "Type"=%s AND "Size"=%s LIMIT 1;', (type_value, size_value)),
            (f"SELECT * FROM {table_name} WHERE type=%s AND size=%s LIMIT 1;", (type_value, size_value)),
            (f'SELECT * FROM "{table_name}" WHERE type=%s AND size=%s LIMIT 1;', (type_value, size_value)),
        ]
    else:
        q_variants = [
            ('SELECT * FROM "Beam" WHERE "Type"=%s AND "Size"=%s LIMIT 1;', (type_value, size_value)),
            ('SELECT * FROM beam WHERE type=%s AND size=%s LIMIT 1;', (type_value, size_value)),
        ]
    for q, p in q_variants:
        df_row, err = run_sql(q, params=p)
        if err:
            continue
        if df_row is not None and not df_row.empty:
            row = df_row.iloc[0].to_dict()
            break
    return row


# =========================================================
# ROBUST NUMERIC PARSER & PICK
# =========================================================
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
            return d[k]
    kl = {str(k).lower(): k for k in d.keys()}
    for k in keys:
        lk = str(k).lower()
        if lk in kl and d.get(kl[lk]) is not None:
            return d.get(kl[lk])
    for k in keys:
        sk = str(k).lower()
        for col in d.keys():
            if sk in str(col).lower():
                v = d.get(col)
                if v is not None:
                    return v
    return default


# =========================================================
# GENERAL HELPERS
# =========================================================
def material_to_fy(mat: str) -> float:
    return {"S235": 235.0, "S275": 275.0, "S355": 355.0}.get(mat, 355.0)


def supports_torsion_and_warping(family: str) -> bool:
    if not family:
        return False
    f = family.upper()
    hollow = any(x in f for x in ["CHS", "RHS", "SHS", "HSS", "BOX", "TUBE"])
    if hollow:
        return False
    return any(x in f for x in ["IPE", "IPN", "HEA", "HEB", "HEM", "UB", "UC", "UNP", "UPN", "I ", "H "])


# =========================================================
# READY CASES GALLERY SYSTEM (77 placeholders)
# =========================================================
# ============================
# CASE 1: SSB-UDL
# ============================
def ss_udl_case(L, w):
    """
    Simply supported beam with full-span UDL.
    Inputs: L (m), w (kN/m)
    Returns (N, My, Mz, Vy, Vz) maxima for prefill.
    """
    Mmax = w * L**2 / 8.0   # kN·m
    Vmax = w * L / 2.0      # kN
    return (0.0, float(Mmax), 0.0, float(Vmax), 0.0)


def ss_udl_diagram(L, w, E=None, I=None, n=200):
    """
    Returns x (m), V (kN), M (kN·m), delta (m) for SSB-UDL.

    V(x) = w(L/2 - x)
    M(x) = w x (L - x) / 2
    δ(x) = w x / (24 E I) * (L^3 - 2Lx^2 + x^3)
    """
    x = np.linspace(0.0, L, n)

    V = w * (L/2.0 - x)                 # kN
    M = w * x * (L - x) / 2.0           # kN·m

    delta = None
    if E and I and I > 0:
        w_Nm = w * 1000.0               # N/m
        delta = (w_Nm * x / (24.0 * E * I)) * (L**3 - 2*L*x**2 + x**3)  # m

    return x, V, M, delta


def ss_udl_deflection_max(L, w, E, I):
    """
    δ_max = 5 w L^4 / (384 E I)
    Inputs: L (m), w (kN/m), E (Pa), I (m^4)
    Returns δ_max in meters.
    """
    w_Nm = w * 1000.0
    return 5.0 * w_Nm * L**4 / (384.0 * E * I)


# ============================
# CASE 2: SSB-CLAC (central point load)
# ============================
def ss_central_point_case(L, P):
    """
    Simply supported beam with a point load at midspan.
    Inputs: L (m), P (kN)
    Returns (N, My, Mz, Vy, Vz) maxima for prefill.
    """
    Mmax = P * L / 4.0      # kN·m
    Vmax = P / 2.0          # kN
    return (0.0, float(Mmax), 0.0, float(Vmax), 0.0)


def ss_central_point_diagram(L, P, E=None, I=None, n=200):
    """
    Returns x (m), V (kN), M (kN·m), delta (m) for midspan point load.

    Reactions: R = P/2
    V(x) = +P/2 for x<L/2, -P/2 for x>=L/2
    M(x) = P x /2 for x<L/2,  P(L-x)/2 for x>=L/2

    Deflection:
      δ(x) = P x / (48 E I) * (3L^2 - 4x^2) for x<=L/2
      symmetrical about midspan
    """
    x = np.linspace(0.0, L, n)

    V = np.where(x < L/2.0, P/2.0, -P/2.0)  # kN
    M = np.where(x < L/2.0, P*x/2.0, P*(L-x)/2.0)  # kN·m

    delta = None
    if E and I and I > 0:
        P_N = P * 1000.0  # N
        # use left-half formula, mirror automatically by x definition
        delta = (P_N * x / (48.0 * E * I)) * (3*L**2 - 4*x**2)  # m
        # for x>L/2, formula gives negative; take symmetric absolute
        # so we mirror shape:
        delta = np.where(x <= L/2.0, delta, (P_N * (L-x) / (48.0 * E * I)) * (3*L**2 - 4*(L-x)**2))

    return x, V, M, delta


def ss_central_point_deflection_max(L, P, E, I):
    """
    δ_max = P L^3 / (48 E I)
    Inputs: L (m), P (kN), E (Pa), I (m^4)
    Returns δ_max in meters.
    """
    P_N = P * 1000.0
    return P_N * L**3 / (48.0 * E * I)


def dummy_case_func(*args, **kwargs):
    return (0.0, 0.0, 0.0, 0.0, 0.0)


def make_cases(prefix, n, default_inputs):
    cases = []
    for i in range(1, n + 1):
        cases.append({
            "key": f"{prefix}-{i:02d}",
            "label": f"Case {i}",
            "img_path": None,   # placeholder
            "inputs": default_inputs.copy(),
            "func": dummy_case_func
        })
    return cases


READY_CATALOG = {
    "Beam": {
        "Simply Supported Beams (13 cases)": make_cases("SS", 13, {"L": 6.0, "w": 10.0}),
        "Beams Fixed at one end (3 cases)": make_cases("FE", 3, {"L": 6.0, "w": 10.0}),
        "Beams Fixed at both ends (3 cases)": make_cases("FB", 3, {"L": 6.0, "w": 10.0}),
        "Cantilever Beams (6 cases)": make_cases("C", 6, {"L": 3.0, "w": 10.0}),
        "Beams with Overhang (6 cases)": make_cases("OH", 6, {"L": 6.0, "a": 1.5, "w": 10.0}),
        "Continuous Beams — Two Spans / Three Supports (7 cases)": make_cases("CS2", 7, {"L1": 4.0, "L2": 4.0, "w": 10.0}),
        "Continuous Beams — Three Spans / Four Supports (3 cases)": make_cases("CS3", 3, {"L1": 4.0, "L2": 4.0, "L3": 4.0, "w": 10.0}),
        "Continuous Beams — Four Spans / Five Supports (3 cases)": make_cases("CS4", 3, {"L1": 4.0, "L2": 4.0, "L3": 4.0, "L4": 4.0, "w": 10.0}),
    },
    "Frame": {
        "Three Member Frames (Pin / Roller) (8 cases)": make_cases("FR3PR", 8, {"L": 4.0, "H": 3.0, "P": 10.0}),
        "Three Member Frames (Pin / Pin) (5 cases)": make_cases("FR3PP", 5, {"L": 4.0, "H": 3.0, "P": 10.0}),
        "Three Member Frames (Fixed / Fixed) (3 cases)": make_cases("FR3FF", 3, {"L": 4.0, "H": 3.0, "P": 10.0}),
        "Three Member Frames (Fixed / Free) (5 cases)": make_cases("FR3FFr", 5, {"L": 4.0, "H": 3.0, "P": 10.0}),
        "Two Member Frames (Pin / Pin) (2 cases)": make_cases("FR2PP", 2, {"L": 4.0, "H": 3.0, "P": 10.0}),
        "Two Member Frames (Fixed / Fixed) (2 cases)": make_cases("FR2FF", 2, {"L": 4.0, "H": 3.0, "P": 10.0}),
        "Two Member Frames (Fixed / Pin) (4 cases)": make_cases("FR2FP", 4, {"L": 4.0, "H": 3.0, "P": 10.0}),
        "Two Member Frames (Fixed / Free) (4 cases)": make_cases("FR2FFr", 4, {"L": 4.0, "H": 3.0, "P": 10.0}),
    }
}

# ---- Patch Case 1 of Simply Supported Beams to real UDL formulas ----
READY_CATALOG["Beam"]["Simply Supported Beams (13 cases)"][0]["label"] = "SSB -  UDL"
READY_CATALOG["Beam"]["Simply Supported Beams (13 cases)"][0]["inputs"] = {"L": 6.0, "w": 10.0}
READY_CATALOG["Beam"]["Simply Supported Beams (13 cases)"][0]["func"] = ss_udl_case
READY_CATALOG["Beam"]["Simply Supported Beams (13 cases)"][0]["diagram_func"] = ss_udl_diagram

# ---- Patch Case 2 of Simply Supported Beams: central point load ----
READY_CATALOG["Beam"]["Simply Supported Beams (13 cases)"][1]["label"] = "SSB-CLAC"
READY_CATALOG["Beam"]["Simply Supported Beams (13 cases)"][1]["inputs"] = {"L": 6.0, "P": 20.0}
READY_CATALOG["Beam"]["Simply Supported Beams (13 cases)"][1]["func"] = ss_central_point_case
READY_CATALOG["Beam"]["Simply Supported Beams (13 cases)"][1]["diagram_func"] = ss_central_point_diagram

def render_case_gallery(chosen_type, chosen_cat, n_per_row=5):
    cases = READY_CATALOG[chosen_type][chosen_cat]
    cols = st.columns(n_per_row)
    clicked = None

    for i, case in enumerate(cases):
        col = cols[i % n_per_row]
        with col:
            if case.get("img_path"):
                col.image(case["img_path"], use_container_width=True)
            else:
                col.markdown(
                    "<div style='height:90px;border:1px dashed #bbb;"
                    "border-radius:8px;display:flex;align-items:center;"
                    "justify-content:center;color:#888;font-size:12px;'>"
                    "image placeholder</div>",
                    unsafe_allow_html=True
                )
            col.caption(case["label"])

            if col.button("Select", key=f"case_select_{chosen_type}_{chosen_cat}_{case['key']}"):
                clicked = case["key"]

    return clicked


def render_ready_cases_panel():
    # Make it open by default so diagrams are visible immediately
    with st.expander("Ready design cases (Beam & Frame) — Gallery", expanded=True):
        st.write("Pick a case visually, then enter its parameters. "
                 "Diagrams update instantly for Beam cases where implemented.")

        chosen_type = st.radio(
            "Step 1 — Structural type",
            ["Beam", "Frame"],
            horizontal=True,
            key="ready_type_gallery"
        )

        categories = list(READY_CATALOG[chosen_type].keys())
        chosen_cat = st.selectbox(
            "Step 2 — Category",
            categories,
            key="ready_cat_gallery"
        )

        # Reset selection if type/category changes
        last_type = st.session_state.get("_ready_last_type")
        last_cat = st.session_state.get("_ready_last_cat")
        if (last_type != chosen_type) or (last_cat != chosen_cat):
            st.session_state["ready_case_key"] = None
            st.session_state["_ready_last_type"] = chosen_type
            st.session_state["_ready_last_cat"] = chosen_cat

        st.markdown("Step 3 — Choose a case:")
        clicked_key = render_case_gallery(chosen_type, chosen_cat, n_per_row=5)

        if clicked_key:
            st.session_state["ready_case_key"] = clicked_key

        case_key = st.session_state.get("ready_case_key")

        if not case_key:
            st.info("Select a case above to enter its parameters and see diagrams.")
            # Clear diagram state
            st.session_state["ready_selected_case"] = None
            st.session_state["ready_input_vals"] = None
            return

        current_cases = READY_CATALOG[chosen_type][chosen_cat]
        current_keys = {c["key"] for c in current_cases}
        if case_key not in current_keys:
            st.session_state["ready_case_key"] = None
            st.info("Selected case was from another category. Please pick a case again.")
            st.session_state["ready_selected_case"] = None
            st.session_state["ready_input_vals"] = None
            return

        selected_case = next(c for c in current_cases if c["key"] == case_key)

        st.markdown(f"**Selected:** {selected_case['key']} — {selected_case['label']}")

        # NEW: which bending axis this case represents
        axis_choice = st.radio(
            "Bending axis for this case",
            ["Strong axis (y)", "Weak axis (z)"],
            horizontal=True,
            key=f"axis_choice_{case_key}"
        )


        
        input_vals = {}
        for k, v in selected_case.get("inputs", {}).items():
            input_vals[k] = st.number_input(
                k,
                value=float(v),
                key=f"ready_param_{case_key}_{k}"
            )

        # Store selection + inputs for diagrams
        st.session_state["ready_selected_case"] = selected_case
        st.session_state["ready_input_vals"] = input_vals

        # >>> SHOW DIAGRAMS RIGHT HERE <<<
        render_beam_diagrams_panel()

if st.button("Apply case to Loads", key=f"apply_case_{case_key}"):

    # Get the function for this case
    func = selected_case.get("func", dummy_case_func)

    try:
        args = [input_vals[k] for k in selected_case["inputs"].keys()]
        N_case, M_generic, _, V_generic, _ = func(*args)
    except Exception:
        N_case, M_generic, V_generic = 0.0, 0.0, 0.0

    # -----------------------------------------
    # NEW — get axis choice (stored by radio button)
    # -----------------------------------------
    axis_choice = st.session_state.get(f"axis_choice_{case_key}", "Strong axis (y)")
    st.session_state["bending_axis"] = (
        "y" if axis_choice.startswith("Strong") else "z"
    )

    # -----------------------------------------
    # NEW — Map M and V to My/Mz and Vy/Vz
    # -----------------------------------------
    if st.session_state["bending_axis"] == "y":
        My_case = M_generic
        Mz_case = 0.0
        Vy_case = V_generic
        Vz_case = 0.0
    else:
        My_case = 0.0
        Mz_case = M_generic
        Vy_case = 0.0
        Vz_case = V_generic

    # -----------------------------------------
    # Save values to session_state
    # -----------------------------------------
    st.session_state["prefill_from_case"] = True
    st.session_state["prefill_N_kN"] = float(N_case)
    st.session_state["prefill_My_kNm"] = float(My_case)
    st.session_state["prefill_Mz_kNm"] = float(Mz_case)
    st.session_state["prefill_Vy_kN"] = float(Vy_case)
    st.session_state["prefill_Vz_kN"] = float(Vz_case)

    if "L" in input_vals:
        st.session_state["case_L"] = float(input_vals["L"])
    elif "L1" in input_vals:
        st.session_state["case_L"] = float(input_vals["L1"])

    st.success("Case applied. Scroll down to Loads tab and run check.")


# =========================================================
# UI RENDERERS
# =========================================================
def render_sidebar_guidelines():
    st.sidebar.title("Guidelines")
    st.sidebar.markdown("""
1) Member & Section  
2) Loads (or Ready Case)  
3) Run check  
4) Results  
5) Report  

DB sections are read-only.
""")


def render_project_data():
    with st.expander("Project data", expanded=False):
        meta_col1, meta_col2, meta_col3 = st.columns([1, 1, 1])
        with meta_col1:
            doc_name = st.text_input("Document title", value="Beam check", key="doc_title_in")
            project_name = st.text_input("Project name", value="", key="project_name_in")
        with meta_col2:
            position = st.text_input("Position / Location (Beam ID)", value="", key="position_in")
            requested_by = st.text_input("Requested by", value="", key="requested_by_in")
        with meta_col3:
            revision = st.text_input("Revision", value="0.1", key="revision_in")
            run_date = st.date_input("Date", value=date.today(), key="run_date_in")
    st.markdown("---")
    return doc_name, project_name, position, requested_by, revision, run_date


def render_section_selection():
    st.subheader("Section selection")
    types, sizes_map, detected_table = fetch_types_and_sizes()

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        material = st.selectbox("Material", ["S235", "S275", "S355"], index=2, key="mat_sel")
    with c2:
        family = st.selectbox(
            "Section family / Type (DB)",
            ["-- choose --"] + types if types else ["-- choose --"],
            key="fam_sel"
        )
    with c3:
        selected_name = None
        selected_row = None
        if family and family != "-- choose --":
            names = sizes_map.get(family, [])
            selected_name = st.selectbox(
                "Section size (DB)",
                ["-- choose --"] + names if names else ["-- choose --"],
                key="size_sel"
            )
            if selected_name and selected_name != "-- choose --":
                selected_row = get_section_row_db(
                    family, selected_name,
                    detected_table if detected_table != "sample" else None
                )

    return material, family, selected_name, selected_row, detected_table


def build_section_display(selected_row):
    alpha_default_val = 0.49
    sr = selected_row
    bad_fields = []

    def get_num_from_row(*keys, default=0.0, fieldname=None):
        raw = pick(sr, *keys, default=None)
        val = safe_float(raw, default=default)
        if (raw is not None) and (val == default) and (raw != default):
            bad_fields.append((fieldname or keys[0], raw))
        return val

    A_cm2 = get_num_from_row("A_cm2", "area_cm2", "area", "A", default=0.0, fieldname="A_cm2")
    S_y_cm3 = get_num_from_row("S_y_cm3", "Sy_cm3", "S_y", "Wel_y", "Wy_cm3", default=0.0, fieldname="S_y_cm3")
    S_z_cm3 = get_num_from_row("S_z_cm3", "Sz_cm3", "S_z", "Wel_z", "Wz_cm3", default=0.0, fieldname="S_z_cm3")
    I_y_cm4 = get_num_from_row("I_y_cm4", "Iy_cm4", "I_y", "Iy", "Iyy", default=0.0, fieldname="I_y_cm4")
    I_z_cm4 = get_num_from_row("I_z_cm4", "Iz_cm4", "I_z", "Iz", "Izz", default=0.0, fieldname="I_z_cm4")
    J_cm4 = get_num_from_row("J_cm4", "J", "torsion_const", default=0.0, fieldname="J_cm4")
    c_max_mm = get_num_from_row("c_max_mm", "c_mm", "c", "cmax", default=0.0, fieldname="c_max_mm")
    Wpl_y_cm3 = get_num_from_row("Wpl_y_cm3", "Wpl_y", "W_pl_y", default=0.0, fieldname="Wpl_y_cm3")
    Wpl_z_cm3 = get_num_from_row("Wpl_z_cm3", "Wpl_z", "W_pl_z", default=0.0, fieldname="Wpl_z_cm3")
    Iw_cm6 = get_num_from_row("Iw_cm6", "Iw", default=0.0, fieldname="Iw_cm6")
    It_cm4 = get_num_from_row("It_cm4", "It", default=0.0, fieldname="It_cm4")
    alpha_curve = get_num_from_row("alpha_curve", "alpha", default=alpha_default_val, fieldname="alpha_curve")

    flange_class_db = str(pick(sr, "flange_class_db", "flange_class", default="n/a"))
    web_class_bending_db = str(pick(sr, "web_class_bending_db", "web_class_bending", default="n/a"))
    web_class_compression_db = str(pick(sr, "web_class_compression_db", "web_class_compression", default="n/a"))

    sr_display = {
        "family": pick(sr, "Type", "family", "type", default="DB"),
        "name": pick(sr, "Size", "name", "designation", default="DB"),
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
    return sr_display, bad_fields


def render_section_summary_like_props(material, sr_display, key_prefix="sum"):
    fy = material_to_fy(material)
    st.markdown("### Selected section summary")

    s1, s2, s3 = st.columns(3)
    s1.text_input("Material", value=material, disabled=True, key=f"{key_prefix}_mat")
    s2.text_input("fy (MPa)", value=f"{fy:.0f}", disabled=True, key=f"{key_prefix}_fy")
    s3.text_input("Family / Type", value=str(sr_display.get("family", "")), disabled=True, key=f"{key_prefix}_fam")

    s4, s5, s6 = st.columns(3)
    s4.text_input("Size", value=str(sr_display.get("name", "")), disabled=True, key=f"{key_prefix}_size")
    s5.number_input("A (cm²)", value=float(sr_display.get("A_cm2", 0.0)), disabled=True, key=f"{key_prefix}_A")
    s6.number_input("c_max (mm)", value=float(sr_display.get("c_max_mm", 0.0)), disabled=True, key=f"{key_prefix}_c")


def render_section_properties_readonly(sr_display, key_prefix="db"):
    c1, c2, c3 = st.columns(3)
    c1.number_input("A (cm²)", value=float(sr_display.get('A_cm2', 0.0)), disabled=True, key=f"{key_prefix}_A")
    c2.number_input("S_y (cm³) about y", value=float(sr_display.get('S_y_cm3', 0.0)), disabled=True, key=f"{key_prefix}_Sy")
    c3.number_input("S_z (cm³) about z", value=float(sr_display.get('S_z_cm3', 0.0)), disabled=True, key=f"{key_prefix}_Sz")

    c4, c5, c6 = st.columns(3)
    c4.number_input("I_y (cm⁴) about y", value=float(sr_display.get('I_y_cm4', 0.0)), disabled=True, key=f"{key_prefix}_Iy")
    c5.number_input("I_z (cm⁴) about z", value=float(sr_display.get('I_z_cm4', 0.0)), disabled=True, key=f"{key_prefix}_Iz")
    c6.number_input("J (cm⁴) (torsion const)", value=float(sr_display.get('J_cm4', 0.0)), disabled=True, key=f"{key_prefix}_J")

    c7, c8, c9 = st.columns(3)
    c7.number_input("c_max (mm)", value=float(sr_display.get('c_max_mm', 0.0)), disabled=True, key=f"{key_prefix}_c")
    c8.number_input("Wpl_y (cm³)", value=float(sr_display.get('Wpl_y_cm3', 0.0)), disabled=True, key=f"{key_prefix}_Wply")
    c9.number_input("Wpl_z (cm³)", value=float(sr_display.get('Wpl_z_cm3', 0.0)), disabled=True, key=f"{key_prefix}_Wplz")

    c10, c11, c12 = st.columns(3)
    c10.number_input("Iw (cm⁶) (warping)", value=float(sr_display.get("Iw_cm6", 0.0)), disabled=True, key=f"{key_prefix}_Iw")
    c11.number_input("It (cm⁴) (torsion inertia)", value=float(sr_display.get("It_cm4", 0.0)), disabled=True, key=f"{key_prefix}_It")
    c12.empty()

    cls1, cls2, cls3 = st.columns(3)
    cls1.text_input("Flange class (DB)", value=str(sr_display.get('flange_class_db', "n/a")), disabled=True, key=f"{key_prefix}_fc")
    cls2.text_input("Web class (bending, DB)", value=str(sr_display.get('web_class_bending_db', "n/a")), disabled=True, key=f"{key_prefix}_wc_b")
    cls3.text_input("Web class (compression, DB)", value=str(sr_display.get('web_class_compression_db', "n/a")), disabled=True, key=f"{key_prefix}_wc_c")

    a1, a2, a3 = st.columns(3)
    alpha_db_val = sr_display.get('alpha_curve', 0.49)
    alpha_label_db = next((lbl for lbl, val in [
        ("0.13 (a)", 0.13), ("0.21 (b)", 0.21), ("0.34 (c)", 0.34), ("0.49 (d)", 0.49), ("0.76 (e)", 0.76)
    ] if abs(val - float(alpha_db_val)) < 1e-8), f"{alpha_db_val}")
    a1.text_input("Buckling α (DB)", value=str(alpha_label_db), disabled=True, key=f"{key_prefix}_alpha")
    a2.empty(); a3.empty()


def render_loads_form(family_for_torsion: str):
    prefill = st.session_state.get("prefill_from_case", False)
    defval = lambda key, fallback: float(st.session_state.get(key, fallback)) if prefill else fallback
    torsion_supported = supports_torsion_and_warping(family_for_torsion)

    with st.form("loads_form", clear_on_submit=False):
        st.subheader("Design forces and moments (ULS) — INPUT")
        st.caption("Positive N = compression.")

        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            L = st.number_input("Element length L (m)", value=defval("case_L", 6.0), min_value=0.0, key="L_in")
        with r1c2:
            N_kN = st.number_input("Axial force N (kN)", value=defval("prefill_N_kN", 0.0), key="N_in")
        with r1c3:
            Vy_kN = st.number_input("Shear V_y (kN)", value=defval("prefill_Vy_kN", 0.0), key="Vy_in")

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            Vz_kN = st.number_input("Shear V_z (kN)", value=defval("prefill_Vz_kN", 0.0), key="Vz_in")
        with r2c2:
            My_kNm = st.number_input("Bending M_y (kN·m) about y", value=defval("prefill_My_kNm", 0.0), key="My_in")
        with r2c3:
            Mz_kNm = st.number_input("Bending M_z (kN·m) about z", value=defval("prefill_Mz_kNm", 0.0), key="Mz_in")

        st.markdown("### Buckling effective length factors (K)")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            K_y = st.number_input("K_y", value=1.0, min_value=0.1, step=0.05, key="Ky_in")
        with k2:
            K_z = st.number_input("K_z", value=1.0, min_value=0.1, step=0.05, key="Kz_in")
        with k3:
            K_LT = st.number_input("K_LT", value=1.0, min_value=0.1, step=0.05, key="KLT_in")
        with k4:
            K_T = st.number_input("K_T", value=1.0, min_value=0.1, step=0.05, key="KT_in")

        Tx_kNm = 0.0
        if torsion_supported:
            st.markdown("### Torsion (only for open I/H/U sections)")
            Tx_kNm = st.number_input("Torsion T_x (kN·m)", value=0.0, key="Tx_in")

        run_btn = st.form_submit_button("Run check")
        if run_btn:
            st.session_state["run_clicked"] = True
            st.session_state["inputs"] = dict(
                L=L, N_kN=N_kN, Vy_kN=Vy_kN, Vz_kN=Vz_kN,
                My_kNm=My_kNm, Mz_kNm=Mz_kNm, Tx_kNm=Tx_kNm,
                K_y=K_y, K_z=K_z, K_LT=K_LT, K_T=K_T
            )

    return torsion_supported


# =========================================================
# COMPUTATION CORE
# =========================================================
def compute_checks(use_props, fy, inputs, torsion_supported):
    gamma_M0 = 1.0
    gamma_M1 = 1.0

    L = inputs["L"]
    N_kN = inputs["N_kN"]
    Vy_kN = inputs["Vy_kN"]
    Vz_kN = inputs["Vz_kN"]
    My_kNm = inputs["My_kNm"]
    Mz_kNm = inputs["Mz_kNm"]
    Tx_kNm = inputs.get("Tx_kNm", 0.0)
    K_y = inputs["K_y"]; K_z = inputs["K_z"]

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
    alpha_curve_db = use_props.get("alpha_curve", 0.49)

    if A_m2 <= 0:
        raise ValueError("Section area not provided (A <= 0).")

    N_Rd_N = A_m2 * fy * 1e6 / gamma_M0
    T_Rd_N = N_Rd_N
    M_Rd_y_Nm = Wpl_y_m3 * fy * 1e6 / gamma_M0
    Av_m2 = 0.6 * A_m2
    V_Rd_N = Av_m2 * fy * 1e6 / (math.sqrt(3) * gamma_M0)

    sigma_allow_MPa = 0.6 * fy

    sigma_axial_Pa = N_N / A_m2
    sigma_by_Pa = My_Nm / S_y_m3 if S_y_m3 > 0 else 0.0
    sigma_bz_Pa = Mz_Nm / S_z_m3 if S_z_m3 > 0 else 0.0

    tau_y_Pa = Vz_N / Av_m2 if Av_m2 > 0 else 0.0
    tau_z_Pa = Vy_N / Av_m2 if Av_m2 > 0 else 0.0

    tau_torsion_Pa = 0.0
    if torsion_supported and J_m4 > 0 and c_max_m > 0:
        tau_torsion_Pa = T_Nm * c_max_m / J_m4

    tau_total_Pa = math.sqrt(tau_y_Pa**2 + tau_z_Pa**2 + tau_torsion_Pa**2)
    tau_total_MPa = tau_total_Pa / 1e6
    sigma_total_MPa = (sigma_axial_Pa + sigma_by_Pa + sigma_bz_Pa) / 1e6
    sigma_eq_MPa = math.sqrt((abs(sigma_total_MPa))**2 + 3.0 * (tau_total_MPa**2))

    # Buckling simplified
    E = 210e9
    buck_results = []
    for axis_label, I_axis, K_axis in [("y", I_y_m4, K_y), ("z", I_z_m4, K_z)]:
        if I_axis <= 0:
            buck_results.append((axis_label, None, None, None, None, "No I"))
            continue
        Leff_axis = K_axis * L
        Ncr = (math.pi**2 * E * I_axis) / (Leff_axis**2)
        lambda_bar = math.sqrt((A_m2 * fy * 1e6) / Ncr) if Ncr > 0 else float('inf')
        alpha_use = alpha_curve_db if alpha_curve_db is not None else 0.49
        phi = 0.5 * (1.0 + alpha_use * (lambda_bar**2))
        sqrt_term = max(phi**2 - lambda_bar**2, 0.0)
        chi = 1.0 / (phi + math.sqrt(sqrt_term)) if (phi + math.sqrt(sqrt_term)) > 0 else 0.0
        N_b_Rd_N = chi * A_m2 * fy * 1e6 / gamma_M1
        status = "OK" if abs(N_N) <= N_b_Rd_N else "EXCEEDS"
        buck_results.append((axis_label, Ncr, lambda_bar, chi, N_b_Rd_N, status))

    N_b_Rd_min_N = min([r[4] for r in buck_results if r[4]], default=None)
    compression_resistance_N = N_b_Rd_min_N if N_b_Rd_min_N else N_Rd_N

    def status_and_util(applied, resistance):
        if not resistance:
            return ("n/a", None)
        util = abs(applied) / resistance
        return ("OK" if util <= 1.0 else "EXCEEDS", util)

    rows = []

    applied_N = N_N if N_N >= 0 else 0.0
    status_comp, util_comp = status_and_util(applied_N, compression_resistance_N)
    rows.append({"Check":"Compression (N≥0)","Applied":f"{applied_N/1e3:.3f} kN",
                 "Resistance":f"{compression_resistance_N/1e3:.3f} kN",
                 "Utilization":f"{util_comp:.3f}" if util_comp else "n/a",
                 "Status":status_comp})

    applied_tension_N = -N_N if N_N < 0 else 0.0
    status_ten, util_ten = status_and_util(applied_tension_N, T_Rd_N)
    rows.append({"Check":"Tension (N<0)","Applied":f"{applied_tension_N/1e3:.3f} kN",
                 "Resistance":f"{T_Rd_N/1e3:.3f} kN",
                 "Utilization":f"{util_ten:.3f}" if util_ten else "n/a",
                 "Status":status_ten})

    applied_shear_N = math.sqrt(Vy_N**2 + Vz_N**2)
    status_shear, util_shear = status_and_util(applied_shear_N, V_Rd_N)
    rows.append({"Check":"Shear (resultant Vy & Vz)","Applied":f"{applied_shear_N/1e3:.3f} kN",
                 "Resistance":f"{V_Rd_N/1e3:.3f} kN",
                 "Utilization":f"{util_shear:.3f}" if util_shear else "n/a",
                 "Status":status_shear})

    rows.append({"Check":"Bending y-y (σ_by)","Applied":f"{sigma_by_Pa/1e6:.3f} MPa",
                 "Resistance":f"{sigma_allow_MPa:.3f} MPa",
                 "Utilization":f"{(abs(sigma_by_Pa/1e6)/sigma_allow_MPa):.3f}" if sigma_allow_MPa>0 else "n/a",
                 "Status":"OK" if abs(sigma_by_Pa/1e6)/sigma_allow_MPa<=1.0 else "EXCEEDS"})

    rows.append({"Check":"Bending z-z (σ_bz)","Applied":f"{sigma_bz_Pa/1e6:.3f} MPa",
                 "Resistance":f"{sigma_allow_MPa:.3f} MPa",
                 "Utilization":f"{(abs(sigma_bz_Pa/1e6)/sigma_allow_MPa):.3f}" if sigma_allow_MPa>0 else "n/a",
                 "Status":"OK" if abs(sigma_bz_Pa/1e6)/sigma_allow_MPa<=1.0 else "EXCEEDS"})

    rows.append({"Check":"Biaxial bending + axial + shear (indicative)",
                 "Applied":f"σ_eq={sigma_eq_MPa:.3f} MPa",
                 "Resistance":f"{sigma_allow_MPa:.3f} MPa",
                 "Utilization":f"{(sigma_eq_MPa/sigma_allow_MPa):.3f}" if sigma_allow_MPa>0 else "n/a",
                 "Status":"OK" if sigma_eq_MPa/sigma_allow_MPa<=1.0 else "EXCEEDS"})

    for axis_label, Ncr, lambda_bar, chi, N_b_Rd_N, status in buck_results:
        if N_b_Rd_N:
            util_buck = abs(N_N) / N_b_Rd_N
            rows.append({"Check":f"Flexural buckling {axis_label}",
                         "Applied":f"{abs(N_N)/1e3:.3f} kN",
                         "Resistance":f"{N_b_Rd_N/1e3:.3f} kN",
                         "Utilization":f"{util_buck:.3f}",
                         "Status":"OK" if util_buck<=1.0 else "EXCEEDS"})

    df_rows = pd.DataFrame(rows).set_index("Check")
    overall_ok = not any(df_rows["Status"] == "EXCEEDS")

    util_values = []
    for chk, r in df_rows.iterrows():
        try:
            util_values.append((chk, float(r["Utilization"])))
        except Exception:
            pass
    governing = max(util_values, key=lambda x: x[1]) if util_values else (None, None)

    extras = dict(
        sigma_allow_MPa=sigma_allow_MPa,
        sigma_eq_MPa=sigma_eq_MPa,
        buck_results=buck_results,
        N_Rd_N=N_Rd_N, M_Rd_y_Nm=M_Rd_y_Nm, V_Rd_N=V_Rd_N
    )
    return df_rows, overall_ok, governing, extras


def render_results(df_rows, overall_ok, governing):
    st.markdown("### Result summary")
    gov_check, gov_util = governing
    status_txt = "OK" if overall_ok else "NOT OK"

    st.caption(
        f"Overall status: **{status_txt}**  |  Governing check: **{gov_check or 'n/a'}**  |  Max util: **{gov_util:.3f}**"
        if gov_util else f"Overall status: **{status_txt}**"
    )

    st.markdown("---")
    st.markdown("### Detailed checks")

    def highlight_row(row):
        s = row["Status"]
        if s == "OK":
            color = "background-color: #e6f7e6"
        elif s == "EXCEEDS":
            color = "background-color: #fde6e6"
        else:
            color = "background-color: #f0f0f0"
        return [color] * len(row)

    st.write(df_rows.style.apply(highlight_row, axis=1))


# =========================================================
# DIAGRAM GENERATION (Beam only for now)
# =========================================================
def diagrams_simply_supported_udl(L, w):
    x = np.linspace(0, L, 401)
    V = w * (L/2 - x)            # kN
    M = w * x * (L - x) / 2      # kN·m
    return x, V, M

def diagrams_simply_supported_point(L, P, a=None):
    if a is None:
        a = L/2
    b = L - a
    R1 = P * b / L
    x = np.linspace(0, L, 801)
    V = np.where(x < a, R1, R1 - P)     # kN
    M = np.where(x < a, R1*x, R1*x - P*(x-a))  # kN·m
    return x, V, M

def diagrams_cantilever_udl(L, w):
    x = np.linspace(0, L, 401)
    V = -w * (L - x)                 # kN
    M = -w * (L - x)**2 / 2          # kN·m
    return x, V, M

def diagrams_cantilever_point(L, P, a=None):
    if a is None:
        a = L
    x = np.linspace(0, L, 801)
    V = np.where(x < a, -P, 0.0)           # kN
    M = np.where(x < a, -P*(a - x), 0.0)   # kN·m
    return x, V, M

def get_beam_diagrams_for_case(case_key: str, input_vals: dict):
    key = (case_key or "").upper()
    inputs = {k: float(v) for k, v in (input_vals or {}).items()}

    L = inputs.get("L", None)
    w = inputs.get("w", None)
    P = inputs.get("P", None)
    a = inputs.get("a", None)

    if not L or L <= 0:
        return None, None, None, "No span length L found for diagram."

    if key.startswith("SS"):
        if w is not None:
            x, V, M = diagrams_simply_supported_udl(L, w)
            return x, V, M, "Simply supported UDL diagram."
        if P is not None:
            x, V, M = diagrams_simply_supported_point(L, P, a=a)
            return x, V, M, "Simply supported point-load diagram."
        return None, None, None, "SS case recognized but missing w or P."

    if key.startswith("C"):
        if w is not None:
            x, V, M = diagrams_cantilever_udl(L, w)
            return x, V, M, "Cantilever UDL diagram."
        if P is not None:
            x, V, M = diagrams_cantilever_point(L, P, a=a)
            return x, V, M, "Cantilever point-load diagram."
        return None, None, None, "Cantilever case recognized but missing w or P."

    return None, None, None, "Diagram not implemented yet for this category."

def render_beam_diagrams_panel():
    """
    Draw V(x) and M(x) side-by-side.
    Show δ_max above diagrams.
    Deflection δ(x) is optional via checkbox.
    """
    selected_case = st.session_state.get("ready_selected_case")
    input_vals = st.session_state.get("ready_input_vals")
    sr_display = st.session_state.get("sr_display")
    chosen_type = st.session_state.get("ready_type_gallery", "Beam")

    if chosen_type != "Beam":
        st.info("Diagrams for frames will be added later.")
        return

    if not selected_case or not input_vals:
        return

    diag_func = selected_case.get("diagram_func")
    if not diag_func:
        st.info("No diagrams yet for this case.")
        return

    # Section stiffness for deflection
    # Section stiffness for deflection
    E = 210e9  # Pa

    I_y_m4 = float(sr_display.get("I_y_cm4", 0.0)) * 1e-8 if sr_display else 0.0
    I_z_m4 = float(sr_display.get("I_z_cm4", 0.0)) * 1e-8 if sr_display else 0.0

    bending_axis = st.session_state.get("bending_axis", "y")
    if bending_axis == "z":
        I_m4 = I_z_m4
    else:
        I_m4 = I_y_m4

    if I_m4 <= 0:
        I_m4 = None  # allow V/M but block deflection

    # Run diagram function with I depending on strong/weak axis
    args = [input_vals[k] for k in selected_case["inputs"].keys()]
    x, V, M, delta = diag_func(*args, E=E, I=I_m4)


    # Run diagram function
    args = [input_vals[k] for k in selected_case["inputs"].keys()]
    x, V, M, delta = diag_func(*args, E=E, I=Iy_m4)

    # ---- Show max deflection ABOVE diagrams ----
    defl_max_func = selected_case.get("defl_max_func")
    if defl_max_func and Iy_m4 is not None:
        try:
            dmax_m = defl_max_func(*args, E, Iy_m4)
            st.info(f"Maximum deflection δ_max ≈ **{dmax_m*1000.0:.3f} mm**")
        except Exception:
            pass
    else:
        st.caption("δ_max not available for this case (missing Iy or formula).")

    # ---- V(x) and M(x) side-by-side ----
    colV, colM = st.columns(2)

    with colV:
        st.markdown("#### Shear force diagram V(x)")
        fig1, ax1 = plt.subplots()
        ax1.plot(x, V)
        ax1.axhline(0, linewidth=1)
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("V (kN)")
        ax1.grid(True)
        st.pyplot(fig1)

    with colM:
        st.markdown("#### Bending moment diagram M(x)")
        fig2, ax2 = plt.subplots()
        ax2.plot(x, M)
        ax2.axhline(0, linewidth=1)
        ax2.set_xlabel("x (m)")
        ax2.set_ylabel("M (kN·m)")
        ax2.grid(True)
        st.pyplot(fig2)

    # ---- Optional deflection diagram ----
    show_defl = st.checkbox("Show deflection diagram δ(x)", value=False, key="show_defl_diag")

    if show_defl:
        if delta is None or Iy_m4 is None:
            st.warning("Deflection diagram not available: missing Iy or case deflection formula.")
            return

        # same layout as V/M: left column plot, right column empty
        colD, colEmpty = st.columns(2)

        with colD:
            st.markdown("#### Deflection diagram δ(x)")
            fig3, ax3 = plt.subplots(figsize=(6, 3.5))  # SAME SIZE as V/M
            ax3.plot(x, delta * 1000.0)  # mm
            ax3.axhline(0, linewidth=1)
            ax3.set_xlabel("x (m)")
            ax3.set_ylabel("δ (mm)")
            ax3.grid(True)
            st.pyplot(fig3)

        with colEmpty:
            st.empty()


# =========================================================
# REPORT TAB UPGRADES + PDF
# =========================================================
def render_project_data_readonly(meta, key_prefix="rpt_proj"):
    doc_name, project_name, position, requested_by, revision, run_date = meta
    st.markdown("### 1. Project data")

    c1, c2, c3 = st.columns(3)
    c1.text_input("Document title", value=doc_name, disabled=True, key=f"{key_prefix}_doc")
    c2.text_input("Project name", value=project_name, disabled=True, key=f"{key_prefix}_proj")
    c3.text_input("Position / Beam ID", value=position, disabled=True, key=f"{key_prefix}_pos")

    c4, c5, c6 = st.columns(3)
    c4.text_input("Requested by", value=requested_by, disabled=True, key=f"{key_prefix}_req")
    c5.text_input("Revision", value=revision, disabled=True, key=f"{key_prefix}_rev")
    c6.text_input("Date", value=run_date.isoformat(), disabled=True, key=f"{key_prefix}_date")


def render_material_properties_readonly(material, key_prefix="rpt_mat"):
    fy = material_to_fy(material)
    st.markdown("### 2. Material properties")

    m1, m2, m3 = st.columns(3)
    m1.text_input("Steel grade", value=material, disabled=True, key=f"{key_prefix}_mat")
    m2.text_input("Yield strength fy (MPa)", value=f"{fy:.0f}", disabled=True, key=f"{key_prefix}_fy")
    m3.text_input("Elastic modulus E (MPa)", value="210000", disabled=True, key=f"{key_prefix}_E")

    m4, m5, m6 = st.columns(3)
    m4.text_input("Shear modulus G (MPa)", value="80769", disabled=True, key=f"{key_prefix}_G")
    m5.text_input("Safety factors", value="Included in DB (γ=1.0)", disabled=True, key=f"{key_prefix}_gamma")
    m6.empty()


def render_loads_readonly(inputs, torsion_supported, key_prefix="rpt_load"):
    st.markdown("### 4. Load inputs & buckling data (ULS)")

    r1, r2, r3 = st.columns(3)
    r1.number_input("Element length L (m)", value=float(inputs["L"]), disabled=True, key=f"{key_prefix}_L")
    r2.number_input("Axial force N (kN)", value=float(inputs["N_kN"]), disabled=True, key=f"{key_prefix}_N")
    r3.number_input("Shear Vy (kN)", value=float(inputs["Vy_kN"]), disabled=True, key=f"{key_prefix}_Vy")

    r4, r5, r6 = st.columns(3)
    r4.number_input("Shear Vz (kN)", value=float(inputs["Vz_kN"]), disabled=True, key=f"{key_prefix}_Vz")
    r5.number_input("Moment My (kN·m)", value=float(inputs["My_kNm"]), disabled=True, key=f"{key_prefix}_My")
    r6.number_input("Moment Mz (kN·m)", value=float(inputs["Mz_kNm"]), disabled=True, key=f"{key_prefix}_Mz")

    st.markdown("#### Buckling effective length factors (K)")
    k1, k2, k3 = st.columns(3)
    k1.number_input("K_y", value=float(inputs["K_y"]), disabled=True, key=f"{key_prefix}_Ky")
    k2.number_input("K_z", value=float(inputs["K_z"]), disabled=True, key=f"{key_prefix}_Kz")
    k3.number_input("K_LT", value=float(inputs["K_LT"]), disabled=True, key=f"{key_prefix}_KLT")

    if torsion_supported:
        t1, t2, t3 = st.columns(3)
        t1.number_input("Torsion Tx (kN·m)", value=float(inputs.get("Tx_kNm", 0.0)), disabled=True, key=f"{key_prefix}_Tx")
        t2.empty(); t3.empty()


def build_report_pdf(meta, material, use_props, inputs, df_rows, overall_ok, governing, full=False):
    if not HAS_RL:
        return None

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    x0, y = 20 * mm, height - 20 * mm
    lh = 6 * mm

    def line(txt, bold=False):
        nonlocal y
        if y < 20 * mm:
            c.showPage()
            y = height - 20 * mm
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 10)
        c.drawString(x0, y, str(txt))
        y -= lh

    doc_name, project_name, position, requested_by, revision, run_date = meta
    fy = material_to_fy(material)

    line(doc_name, bold=True)
    line(f"Project: {project_name}")
    line(f"Position / ID: {position}")
    line(f"Requested by: {requested_by}")
    line(f"Revision: {revision}   Date: {run_date.isoformat()}")
    line("")

    line("Material properties", bold=True)
    line(f"Steel grade: {material}")
    line(f"fy: {fy:.0f} MPa, E: 210000 MPa, G: 80769 MPa")
    line("")

    line("Section properties (DB)", bold=True)
    for k in ["family","name","A_cm2","S_y_cm3","S_z_cm3","I_y_cm4","I_z_cm4","J_cm4",
              "c_max_mm","Wpl_y_cm3","Wpl_z_cm3","Iw_cm6","It_cm4","alpha_curve",
              "flange_class_db","web_class_bending_db","web_class_compression_db"]:
        if k in use_props:
            line(f"{k}: {use_props[k]}")
    line("")

    line("Loads & buckling (ULS)", bold=True)
    for k in ["L","N_kN","Vy_kN","Vz_kN","My_kNm","Mz_kNm","Tx_kNm","K_y","K_z","K_LT","K_T"]:
        if k in inputs:
            line(f"{k}: {inputs[k]}")
    line("")

    line("Results summary", bold=True)
    gov_check, gov_util = governing
    line(f"Overall status: {'OK' if overall_ok else 'NOT OK'}")
    if gov_check:
        line(f"Governing check: {gov_check}  (util={gov_util:.3f})")
    line("")

    line("Detailed checks", bold=True)
    for chk, r in df_rows.iterrows():
        line(f"{chk}: util={r.get('Utilization','n/a')}  status={r.get('Status','n/a')}")

    if full:
        line("")
        line("Full formulas included in app (Full report mode).", bold=True)

    c.save()
    pdf = buf.getvalue()
    buf.close()
    return pdf


def render_report_tab(meta, material, use_props, inputs, df_rows, overall_ok, governing, extras):
    st.markdown("## Engineering report")

    report_mode = st.radio(
        "Report mode",
        ["Light report (professional summary)", "Full report (with formulas & details)"],
        horizontal=True,
        key="report_mode"
    )
    full_mode = report_mode.startswith("Full")

    # PDF button
    if HAS_RL:
        pdf_bytes = build_report_pdf(
            meta, material, use_props, inputs, df_rows, overall_ok, governing, full=full_mode
        )
        st.download_button(
            "Download PDF report",
            data=pdf_bytes,
            file_name=f"{meta[0].replace(' ', '_')}_{meta[4]}.pdf",
            mime="application/pdf",
            key="dl_pdf_report"
        )
    else:
        st.warning("PDF export requires reportlab. Add to requirements.txt: reportlab")

    st.markdown("---")

    render_project_data_readonly(meta, key_prefix="rpt_proj_v2")
    st.markdown("---")

    render_material_properties_readonly(material, key_prefix="rpt_mat_v2")
    st.markdown("---")

    st.markdown("### 3. Section properties (from DB)")
    render_section_properties_readonly(use_props, key_prefix="rpt_sec_v2")
    st.markdown("---")

    torsion_supported = supports_torsion_and_warping(use_props.get("family", ""))
    render_loads_readonly(inputs, torsion_supported, key_prefix="rpt_load_v2")
    st.markdown("---")

    st.markdown("### 5. Results summary")
    gov_check, gov_util = governing
    status_txt = "OK" if overall_ok else "NOT OK"

    s1, s2, s3 = st.columns(3)
    s1.text_input("Overall status", value=status_txt, disabled=True, key="rpt_res_status")
    s2.text_input("Governing check", value=gov_check or "n/a", disabled=True, key="rpt_res_gov")
    s3.text_input("Max utilization", value=f"{gov_util:.3f}" if gov_util is not None else "n/a",
                  disabled=True, key="rpt_res_util")

    st.markdown("### 5.1 Detailed checks")
    def highlight_row(row):
        s = row["Status"]
        if s == "OK":
            color = "background-color: #e6f7e6"
        elif s == "EXCEEDS":
            color = "background-color: #fde6e6"
        else:
            color = "background-color: #f0f0f0"
        return [color] * len(row)

    st.write(df_rows.style.apply(highlight_row, axis=1))

    if full_mode:
        st.markdown("---")
        st.markdown("## 6. Full formulas & intermediate steps")

        with st.expander("6.1 Axial resistance (EN1993-1-1 §6.2.3)", expanded=False):
            st.latex(r"N_{Rd} = A \cdot f_y")
            st.write(f"N_Rd = {extras['N_Rd_N']/1e3:.3f} kN")

        with st.expander("6.2 Bending resistance (EN1993-1-1 §6.2.5)", expanded=False):
            st.latex(r"M_{Rd,y} = W_{pl,y} \cdot f_y")
            st.write(f"M_Rd,y = {extras['M_Rd_y_Nm']/1e3:.3f} kN·m")

        with st.expander("6.3 Shear resistance (EN1993-1-1 §6.2.6)", expanded=False):
            st.latex(r"V_{Rd} = \dfrac{A_v f_y}{\sqrt{3}}")
            st.write(f"V_Rd = {extras['V_Rd_N']/1e3:.3f} kN")

        with st.expander("6.4 Flexural buckling (EN1993-1-1 §6.3.1)", expanded=False):
            st.latex(r"N_{cr} = \dfrac{\pi^2 E I}{(K L)^2}")
            for axis_label, Ncr, lambda_bar, chi, N_b_Rd_N, status in extras["buck_results"]:
                if N_b_Rd_N:
                    st.write(
                        f"Axis {axis_label}: "
                        f"Ncr={Ncr/1e3:.2f} kN, "
                        f"λ̄={lambda_bar:.3f}, χ={chi:.3f}, "
                        f"Nb,Rd={N_b_Rd_N/1e3:.2f} kN → {status}"
                    )

    st.caption("Preliminary report only — verify final design per EN1993.")


# =========================================================
# APP ENTRY
# =========================================================
st.set_page_config(page_title="EngiSnap Beam Design Eurocode Checker", layout="wide")

st.markdown("""
<style>
h1 {font-size: 1.6rem !important;}
h2 {font-size: 1.25rem !important;}
h3 {font-size: 1.05rem !important;}
div.block-container {padding-top: 1.2rem;}
</style>
""", unsafe_allow_html=True)

st.title("EngiSnap — Standard steel beam checks (Eurocode prototype)")
st.caption("Simplified screening checks — not a full EN1993 implementation.")


def render_section_preview_placeholder(title="Cross-section preview", key_prefix="prev"):
    st.markdown("### Cross-section preview")

    center_cols = st.columns([1, 2, 1])
    center_cols[1].markdown(
        f"""
<div style="
    border: 2px dashed #bbb;
    border-radius: 10px;
    width: 100%;
    max-width: 460px;
    height: 240px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #fafafa;
    color: #777;
    font-weight: 600;
    margin: 6px auto 8px auto;
    text-align:center;
">
    {title}<br/>
    <span style="font-size:12px;font-weight:400;color:#999;">
        (image placeholder — will be loaded from DB later)
    </span>
</div>
""",
        unsafe_allow_html=True,
    )


render_sidebar_guidelines()

tab1, tab2, tab3, tab4 = st.tabs(["1) Member & Section", "2) Loads", "3) Results", "4) Report"])

with tab1:
    meta = render_project_data()
    material, family, selected_name, selected_row, detected_table = render_section_selection()
    st.session_state["material"] = material
    st.session_state["meta"] = meta

    if selected_row is not None:
        sr_display, bad_fields = build_section_display(selected_row)
        st.session_state["sr_display"] = sr_display

        render_section_summary_like_props(material, sr_display, key_prefix="sum_tab1")

        render_section_preview_placeholder(
            title=f"{sr_display.get('family','')}  {sr_display.get('name','')}",
            key_prefix="tab1_prev"
        )

        with st.expander("Section properties (from DB — read only)", expanded=False):
            render_section_properties_readonly(sr_display, key_prefix="tab1_db")

        if bad_fields:
            st.warning("Some DB numeric fields were not parsed cleanly. See debug in Results tab.")
    else:
        st.info("Select a DB section to continue.")

with tab2:
    sr_display = st.session_state.get("sr_display", None)
    if sr_display is None:
        st.warning("Go to Member & Section tab first and select a section.")
    else:
        render_ready_cases_panel()  # diagrams are inside now
        render_loads_form(sr_display.get("family", ""))


with tab3:
    sr_display = st.session_state.get("sr_display", None)
    material = st.session_state.get("material", "S355")
    fy = material_to_fy(material)

    if not st.session_state.get("run_clicked", False):
        st.info("Run the Loads form first, then come back here.")
    elif sr_display is None:
        st.warning("No section data found. Select a section first.")
    else:
        inputs = st.session_state.get("inputs", {})
        torsion_supported = supports_torsion_and_warping(sr_display.get("family", ""))

        try:
            df_rows, overall_ok, governing, extras = compute_checks(sr_display, fy, inputs, torsion_supported)
            st.session_state["df_rows"] = df_rows
            st.session_state["overall_ok"] = overall_ok
            st.session_state["governing"] = governing
            st.session_state["extras"] = extras

            render_results(df_rows, overall_ok, governing)

            with st.expander("Engineer debug (raw DB row & intermediate values)", expanded=False):
                st.json(sr_display)
                st.write(extras)

        except Exception as e:
            st.error(f"Computation error: {e}")

with tab4:
    sr_display = st.session_state.get("sr_display", None)
    inputs = st.session_state.get("inputs", {})
    overall_ok = st.session_state.get("overall_ok", None)
    df_rows = st.session_state.get("df_rows", None)
    governing = st.session_state.get("governing", (None, None))
    extras = st.session_state.get("extras", {})
    material = st.session_state.get("material", "S355")
    meta = st.session_state.get("meta", None)

    if sr_display is None or not inputs or overall_ok is None or df_rows is None or meta is None:
        st.info("Select section and run checks first.")
    else:
        render_report_tab(meta, material, sr_display, inputs, df_rows, overall_ok, governing, extras)
















