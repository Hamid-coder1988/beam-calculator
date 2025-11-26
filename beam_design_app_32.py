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
import io

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

from io import BytesIO
from reportlab.lib.utils import ImageReader  # for PDF images

def fig_to_png_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


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

# --------------------------------------------------------
# SECTION IMAGE LOADER (CUSTOM FOR YOUR DB TYPES)
# --------------------------------------------------------
def get_section_image(family: str):
    if not family:
        return None

    family = family.upper().strip()

    # Group 1: All use IPE.png
    GROUP_IPE = ["IPE", "HEA", "HEB", "HEM", "UB", "UC", "UBP"]

    if family in GROUP_IPE:
        return "sections_img/IPE.png"   # <-- FIXED PATH

    # Group 2: own image
    GROUP_OWN = ["SHS", "RHS", "CHS", "UPE", "UPN", "PFC", "L"]

    if family in GROUP_OWN:
        return f"sections_img/{family}.png"   # <-- FIXED PATH

    return None
    

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

# ============================
# CASE 3: SSB-C1
# Simply supported beam, 2 unequal point loads + partial UDL
# ============================

def ssb_c1_diagram(L, P1, a1, P2, a2, w, a_udl, b_udl, E=None, I=None, n=801):
    """
    Simply supported beam with:
      - Point load P1 at x = a1
      - Point load P2 at x = a2
      - Partial UDL of intensity w from x = a_udl to x = a_udl + b_udl

    Units:
      L, a1, a2, a_udl, b_udl in m
      P1, P2 in kN
      w in kN/m
      E in Pa, I in m^4

    Returns:
      x (m), V (kN), M (kN·m), delta (m or None if E/I not given)
    """
    L = float(L)
    P1 = float(P1)
    P2 = float(P2)
    w = float(w)
    a1 = float(a1)
    a2 = float(a2)
    a_udl = float(a_udl)
    b_udl = float(b_udl)

    # clamp UDL to the span
    udl_start = max(0.0, a_udl)
    udl_end = min(L, a_udl + b_udl)
    b_eff = max(0.0, udl_end - udl_start)

    # x grid
    x = np.linspace(0.0, L, n)

    # ------------------
    # Reactions R1, R2
    # ------------------
    W_udl = w * b_eff  # kN
    if b_eff > 0.0:
        x_udl_c = udl_start + b_eff / 2.0
    else:
        x_udl_c = 0.0

    M_total = P1 * a1 + P2 * a2 + W_udl * x_udl_c
    R1 = M_total / L
    R2 = P1 + P2 + W_udl - R1

    # ------------------
    # Shear V(x)
    # ------------------
    V = np.full_like(x, R1, dtype=float)

    # subtract point loads when x >= their positions
    V = V - P1 * (x >= a1) - P2 * (x >= a2)

    # UDL contribution
    if b_eff > 0.0:
        mask1 = (x >= udl_start) & (x <= udl_end)
        mask2 = (x > udl_end)

        V[mask1] -= w * (x[mask1] - udl_start)
        V[mask2] -= w * b_eff

    # ------------------
    # Moment M(x)
    # ------------------
    M = R1 * x
    M -= P1 * np.clip(x - a1, 0.0, None)
    M -= P2 * np.clip(x - a2, 0.0, None)

    if b_eff > 0.0:
        M_udl = np.zeros_like(x)
        # within UDL region
        mask1 = (x >= udl_start) & (x <= udl_end)
        M_udl[mask1] = w * (x[mask1] - udl_start)**2 / 2.0
        # to the right of UDL
        mask2 = (x > udl_end)
        M_udl[mask2] = w * b_eff * (x[mask2] - (udl_start + b_eff / 2.0))
        M -= M_udl

    # ------------------
    # Deflection δ(x) via numeric double integration of M/EI
    # ------------------
    delta = None
    if E and I and I > 0 and L > 0:
        M_Nm = M * 1000.0  # kN·m -> N·m
        curvature = M_Nm / (E * I)  # 1/m

        dx = np.diff(x)
        theta = np.zeros_like(x)
        delta_raw = np.zeros_like(x)

        # integrate curvature -> slope
        for i in range(len(x) - 1):
            theta[i+1] = theta[i] + 0.5 * (curvature[i] + curvature[i+1]) * dx[i]

        # integrate slope -> deflection
        for i in range(len(x) - 1):
            delta_raw[i+1] = delta_raw[i] + 0.5 * (theta[i] + theta[i+1]) * dx[i]

        # enforce simply supported: δ(0) = δ(L) = 0
        delta = delta_raw - (x / L) * delta_raw[-1]

    return x, V, M, delta


def ssb_c1_case(L, P1, a1, P2, a2, w, a_udl, b_udl):
    """
    Case function for SSB-C1 used to prefill Loads tab.
    Returns (N, My, Mz, Vy, Vz) maxima (strong axis).
    """
    x, V, M, _ = ssb_c1_diagram(L, P1, a1, P2, a2, w, a_udl, b_udl, E=None, I=None)
    Vmax = float(np.nanmax(np.abs(V))) if V is not None else 0.0
    Mmax = float(np.nanmax(np.abs(M))) if M is not None else 0.0
    return (0.0, Mmax, 0.0, Vmax, 0.0)


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
        # Category 1: 5 cases
        "Simply Supported Beams (5 cases)": make_cases("SS", 5, {"L": 6.0, "w": 10.0}),
        # Category 2: 1 case
        "Beams Fixed at one end (1 case)": make_cases("FE", 1, {"L": 6.0, "w": 10.0}),
        # Category 3: 1 case
        "Beams Fixed at both ends (1 case)": make_cases("FB", 1, {"L": 6.0, "w": 10.0}),
        # Category 4: 2 cases
        "Cantilever Beams (2 cases)": make_cases("C", 2, {"L": 3.0, "w": 10.0}),
        # Category 5: 3 cases
        "Beams with Overhang (3 cases)": make_cases("OH", 3, {"L": 6.0, "a": 1.5, "w": 10.0}),
        # Category 6: 2 cases
        "Continuous Beams — Two Spans / Three Supports (2 cases)": make_cases("CS2", 2, {"L1": 4.0, "L2": 4.0, "w": 10.0}),
        # Category 7: 1 case
        "Continuous Beams — Three Spans / Four Supports (1 case)": make_cases("CS3", 1, {"L1": 4.0, "L2": 4.0, "L3": 4.0, "w": 10.0}),
        # Category 8: 1 case
        "Continuous Beams — Four Spans / Five Supports (1 case)": make_cases("CS4", 1, {"L1": 4.0, "L2": 4.0, "L3": 4.0, "L4": 4.0, "w": 10.0}),
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
READY_CATALOG["Beam"]["Simply Supported Beams (5 cases)"][0]["label"] = "SSB -  C1"
READY_CATALOG["Beam"]["Simply Supported Beams (5 cases)"][0]["inputs"] = {"L": 6.0, "w": 10.0}
READY_CATALOG["Beam"]["Simply Supported Beams (5 cases)"][0]["func"] = ss_udl_case
READY_CATALOG["Beam"]["Simply Supported Beams (5 cases)"][0]["diagram_func"] = ss_udl_diagram

# ---- Patch Case 2 of Simply Supported Beams: central point load ----
READY_CATALOG["Beam"]["Simply Supported Beams (5 cases)"][1]["label"] = "SSB -  C2"
READY_CATALOG["Beam"]["Simply Supported Beams (5 cases)"][1]["inputs"] = {"L": 6.0, "P": 20.0}
READY_CATALOG["Beam"]["Simply Supported Beams (5 cases)"][1]["func"] = ss_central_point_case
READY_CATALOG["Beam"]["Simply Supported Beams (5 cases)"][1]["diagram_func"] = ss_central_point_diagram

# ---- Patch Case 3 of Simply Supported Beams: SSB-C1 (2P + partial UDL) ----
READY_CATALOG["Beam"]["Simply Supported Beams (5 cases)"][2]["label"] = "SSB - C3"
READY_CATALOG["Beam"]["Simply Supported Beams (5 cases)"][2]["inputs"] = {
    "L": 6.0,
    "P1": 50.0,
    "a1": 2.0,
    "P2": 30.0,
    "a2": 4.0,
    "w": 10.0,
    "a_udl": 1.5,
    "b_udl": 3.0,
}
READY_CATALOG["Beam"]["Simply Supported Beams (5 cases)"][2]["func"] = ssb_c1_case
READY_CATALOG["Beam"]["Simply Supported Beams (5 cases)"][2]["diagram_func"] = ssb_c1_diagram


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
        
        # Choose bending axis for this beam case
        if chosen_type == "Beam":
            axis_choice = st.radio(
                "Step 4 : Bending axis for this case",
                ["Strong axis (y)", "Weak axis (z)"],
                horizontal=True,
                key=f"axis_choice_{case_key}"
            )
        else:
            # Frames: just keep strong axis convention
            axis_choice = "Strong axis (y)"

        # Arrange case inputs in rows of 3 per line
        input_vals = {}
        inputs_dict = selected_case.get("inputs", {})
        keys = list(inputs_dict.keys())

        for i in range(0, len(keys), 3):
            row_keys = keys[i:i+3]
            cols = st.columns(3)
            for col, k in zip(cols, row_keys):
                with col:
                    v = inputs_dict[k]
                    input_vals[k] = st.number_input(
                        k,
                        value=float(float(v)),
                        key=f"ready_param_{case_key}_{k}"
                    )


        # Store selection + inputs for diagrams
        st.session_state["ready_selected_case"] = selected_case
        st.session_state["ready_input_vals"] = input_vals

        # >>> SHOW DIAGRAMS RIGHT HERE <<<
        render_beam_diagrams_panel()

        if st.button("Apply case to Loads", key=f"apply_case_{case_key}"):

            func = selected_case.get("func", dummy_case_func)
            try:
                args = [input_vals[k] for k in selected_case["inputs"].keys()]
                # Your case functions currently return: N, My, Mz, Vy, Vz
                N_case, My_raw, Mz_raw, Vy_raw, Vz_raw = func(*args)
            except Exception:
                N_case = My_raw = Mz_raw = Vy_raw = Vz_raw = 0.0

            # Decide which axis this case is using
            bending_axis = "y" if axis_choice.startswith("Strong") else "z"
            st.session_state["bending_axis"] = bending_axis

            # Map bending and shear to that axis
            if bending_axis == "y":
                # Use My/Vy as-is (strong axis). If those are zero, fall back to z.
                My_case = My_raw if My_raw != 0.0 else Mz_raw
                Mz_case = 0.0
                Vy_case = Vy_raw if Vy_raw != 0.0 else Vz_raw
                Vz_case = 0.0
            else:
                # Weak axis: move My→Mz, Vy→Vz
                Mz_case = My_raw if My_raw != 0.0 else Mz_raw
                My_case = 0.0
                Vz_case = Vy_raw if Vy_raw != 0.0 else Vz_raw
                Vy_case = 0.0

            # Prefill Loads tab
            st.session_state["prefill_from_case"] = True
            st.session_state["prefill_N_kN"] = float(N_case)
            st.session_state["prefill_My_kNm"] = float(My_case)
            st.session_state["prefill_Mz_kNm"] = float(Mz_case)
            st.session_state["prefill_Vy_kN"] = float(Vy_case)
            st.session_state["prefill_Vz_kN"] = float(Vz_case)

            # Use L or L1 as element length if present
            if "L" in input_vals:
                st.session_state["case_L"] = float(input_vals["L"])
            elif "L1" in input_vals:
                st.session_state["case_L"] = float(input_vals["L1"])

            # Keep these for diagrams (you already had something like this)
            st.session_state["ready_selected_case"] = selected_case
            st.session_state["ready_input_vals"] = input_vals

            st.success("Case applied. Now go to Loads tab and click Run check.")

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
            revision = st.text_input("Revision", value="A", key="revision_in")
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
    """
    Map DB row -> a clean sr_display dict with:
    - all UI fields the user wants
    - plus the "old" A_cm2, S_y_cm3, etc. for calculations.
    DB column names assumed:
      h, b, tw, tf, m, A, Av,z, Av,y, Iy, iy, Wel,y, Wpl,y,
      Iz, iz, Wel,z, Wpl,z, IT, WT, Iw, Ww,
      Npl,Rd, Vpl,Rd,z, Vpl,Rd,y, Mel,Rd,y, Mpl,Rd,y, Mel,Rd,z, Mpl,Rd,z,
      a y, a z, CL Wb, CL Wc, CL Fb
    """
    alpha_default_val = 0.49
    sr = selected_row
    bad_fields = []

    def get_num_from_row(*keys, default=0.0, fieldname=None):
        raw = pick(sr, *keys, default=None)
        val = safe_float(raw, default=default)
        if (raw is not None) and (val == default) and (raw != default):
            bad_fields.append((fieldname or keys[0], raw))
        return val

    # --- Geometry & basic section data (mm / kg/m / mm²) ---
    h_mm   = get_num_from_row("h", "H", "depth", fieldname="h_mm")
    b_mm   = get_num_from_row("b", "B", "width", fieldname="b_mm")
    tw_mm  = get_num_from_row("tw", "t_w", "tw_mm", "web_thk", fieldname="tw_mm")
    tf_mm  = get_num_from_row("tf", "t_f", "tf_mm", "flange_thk", fieldname="tf_mm")
    m_kg_m = get_num_from_row("m", "m_kgm", "kg/m", "mass", fieldname="m_kg_m")
    A_mm2  = get_num_from_row("A", "A_mm2", "area_mm2", fieldname="A_mm2")
    Avz_mm2 = get_num_from_row("Av,z", "Av_z", "Avz", "Avz_mm2", fieldname="Avz_mm2")
    Avy_mm2 = get_num_from_row("Av,y", "Av_y", "Avy", "Avy_mm2", fieldname="Avy_mm2")

    # --- Bending stiffness & section moduli (cm⁴ / cm³ / mm) ---
    Iy_cm4  = get_num_from_row("Iy", "I_y_cm4", "Iy_cm4", fieldname="Iy_cm4")
    iy_mm   = get_num_from_row("iy", "ry", "iy_mm", fieldname="iy_mm")
    Wel_y_cm3 = get_num_from_row("Wel,y", "Wel_y", "Wy_el", "Wel_y_cm3", fieldname="Wel_y_cm3")
    Wpl_y_cm3 = get_num_from_row("Wpl,y", "Wpl_y", "Wpl_y_cm3", fieldname="Wpl_y_cm3")

    Iz_cm4  = get_num_from_row("Iz", "I_z_cm4", "Iz_cm4", fieldname="Iz_cm4")
    iz_mm   = get_num_from_row("iz", "rz", "iz_mm", fieldname="iz_mm")
    Wel_z_cm3 = get_num_from_row("Wel,z", "Wel_z", "Wz_el", "Wel_z_cm3", fieldname="Wel_z_cm3")
    Wpl_z_cm3 = get_num_from_row("Wpl,z", "Wpl_z", "Wpl_z_cm3", fieldname="Wpl_z_cm3")

    # --- Torsion & warping (IT, WT, Iw, Ww in mm-based units) ---
    IT_k_mm4 = get_num_from_row("IT", "It", fieldname="IT_k_mm4")   # [×10³ mm⁴]
    WT_k_mm3 = get_num_from_row("WT", "Wt", fieldname="WT_k_mm3")   # [×10³ mm³]
    Iw_6_mm6 = get_num_from_row("Iw", "Iw6", fieldname="Iw_6_mm6")  # [×10⁶ mm⁶]
    Ww_k_mm4 = get_num_from_row("Ww", "Ww_k", fieldname="Ww_k_mm4") # [×10³ mm⁴]

    # --- Resistances (kN / kNm) ---
    Npl_Rd_kN   = get_num_from_row("Npl,Rd", "Npl_Rd", fieldname="Npl_Rd_kN")
    Vpl_Rd_z_kN = get_num_from_row("Vpl,Rd,z", "Vpl_Rd_z", fieldname="Vpl_Rd_z_kN")
    Vpl_Rd_y_kN = get_num_from_row("Vpl,Rd,y", "Vpl_Rd_y", fieldname="Vpl_Rd_y_kN")
    Mel_Rd_y_kNm = get_num_from_row("Mel,Rd,y", "Mel_Rd_y", fieldname="Mel_Rd_y_kNm")
    Mpl_Rd_y_kNm = get_num_from_row("Mpl,Rd,y", "Mpl_Rd_y", fieldname="Mpl_Rd_y_kNm")
    Mel_Rd_z_kNm = get_num_from_row("Mel,Rd,z", "Mel_Rd_z", fieldname="Mel_Rd_z_kNm")
    Mpl_Rd_z_kNm = get_num_from_row("Mpl,Rd,z", "Mpl_Rd_z", fieldname="Mpl_Rd_z_kNm")

    # --- Buckling curves & classes ---
    alpha_y = get_num_from_row("a y", "ay", "alpha_y", default=alpha_default_val, fieldname="alpha_y")
    alpha_z = get_num_from_row("a z", "az", "alpha_z", default=alpha_default_val, fieldname="alpha_z")

    CL_Wb = str(pick(sr, "CL Wb", "CL_Wb", "web_class_bend", default="n/a"))
    CL_Wc = str(pick(sr, "CL Wc", "CL_Wc", "web_class_comp", default="n/a"))
    CL_Fb = str(pick(sr, "CL Fb", "CL_Fb", "flange_class", default="n/a"))

    # --- Optional: c_max for torsion radius ---
    c_max_mm = get_num_from_row("c_max_mm", "c_mm", "c", "cmax", default=0.0, fieldname="c_max_mm")

    # --- Convert DB units to the old calculation format ---
    # Area: mm² -> cm²
    A_cm2 = A_mm2 / 100.0 if A_mm2 else 0.0

    # Shear area: mm² -> m² (for checks)
    # will be used as Av_m2 ~ 0.6*A, but we keep mm² here for info.

    # Torsion const: IT [×10³ mm⁴] -> true mm⁴ -> cm⁴
    #  IT_k * 10³ mm⁴ * 1e-4 = IT_k * 0.1 cm⁴
    J_cm4 = IT_k_mm4 * 0.1 if IT_k_mm4 else 0.0
    It_cm4 = J_cm4

    # Warping constant Iw: [×10⁶ mm⁶] -> true mm⁶ -> cm⁶
    #  1 mm⁶ = 1e-6 cm⁶ ⇒ Iw_cm6 = Iw_6_mm6 * 10⁶ * 1e-6 = Iw_6_mm6
    Iw_cm6 = Iw_6_mm6 if Iw_6_mm6 else 0.0

    # Use elastic moduli as section moduli for bending checks
    S_y_cm3 = Wel_y_cm3
    S_z_cm3 = Wel_z_cm3

    family = pick(sr, "Type", "family", "type", default="DB")
    name   = pick(sr, "Size", "name", "designation", default="DB")

    sr_display = {
        "family": family,
        "name":   name,

        # --- for Selected section summary ---
        "m_kg_per_m": m_kg_m,
        "Iy_cm4": Iy_cm4,
        "Iz_cm4": Iz_cm4,
        "Wel_y_cm3": Wel_y_cm3,
        "Wel_z_cm3": Wel_z_cm3,
        "It_cm4": It_cm4,

        # --- raw geometry & area (mm / kg/m / mm²) ---
        "h_mm": h_mm,
        "b_mm": b_mm,
        "tw_mm": tw_mm,
        "tf_mm": tf_mm,
        "A_mm2": A_mm2,
        "Avz_mm2": Avz_mm2,
        "Avy_mm2": Avy_mm2,
        "iy_mm": iy_mm,
        "iz_mm": iz_mm,

        # --- bending / torsion / warping ---
        "Iy_cm4": Iy_cm4,
        "Iz_cm4": Iz_cm4,
        "Wel_y_cm3": Wel_y_cm3,
        "Wpl_y_cm3": Wpl_y_cm3,
        "Wel_z_cm3": Wel_z_cm3,
        "Wpl_z_cm3": Wpl_z_cm3,
        "IT_k_mm4": IT_k_mm4,
        "WT_k_mm3": WT_k_mm3,
        "Iw_6_mm6": Iw_6_mm6,
        "Ww_k_mm4": Ww_k_mm4,

        # --- design resistances ---
        "Npl_Rd_kN": Npl_Rd_kN,
        "Vpl_Rd_z_kN": Vpl_Rd_z_kN,
        "Vpl_Rd_y_kN": Vpl_Rd_y_kN,
        "Mel_Rd_y_kNm": Mel_Rd_y_kNm,
        "Mpl_Rd_y_kNm": Mpl_Rd_y_kNm,
        "Mel_Rd_z_kNm": Mel_Rd_z_kNm,
        "Mpl_Rd_z_kNm": Mpl_Rd_z_kNm,

        # --- buckling curves & classes ---
        "alpha_y": alpha_y,
        "alpha_z": alpha_z,
        "CL_Wb": CL_Wb,
        "CL_Wc": CL_Wc,
        "CL_Fb": CL_Fb,

        # --- also keep legacy keys used by compute_checks() ---
        "A_cm2": A_cm2,
        "S_y_cm3": S_y_cm3,
        "S_z_cm3": S_z_cm3,
        "I_y_cm4": Iy_cm4,
        "I_z_cm4": Iz_cm4,
        "J_cm4": J_cm4,
        "c_max_mm": c_max_mm,
        "Iw_cm6": Iw_cm6,
        "It_cm4": It_cm4,
        # single alpha_curve (use major-axis value for now)
        "alpha_curve": alpha_y if alpha_y else alpha_default_val,
        "flange_class_db": CL_Fb,
        "web_class_bending_db": CL_Wb,
        "web_class_compression_db": CL_Wc,
    }

    return sr_display, bad_fields

def render_section_summary_like_props(material, sr_display, key_prefix="sum"):
    st.markdown("### Selected section summary")

    s1, s2, s3 = st.columns(3)
    s1.number_input(
        "m (kg/m)",
        value=float(sr_display.get("m_kg_per_m", 0.0)),
        disabled=True,
        key=f"{key_prefix}_m"
    )
    s2.number_input(
        "Iy (cm⁴)",
        value=float(sr_display.get("Iy_cm4", 0.0)),
        disabled=True,
        key=f"{key_prefix}_Iy"
    )
    s3.number_input(
        "Iz (cm⁴)",
        value=float(sr_display.get("Iz_cm4", 0.0)),
        disabled=True,
        key=f"{key_prefix}_Iz"
    )

    s4, s5, s6 = st.columns(3)
    s4.number_input(
        "Wel,y (cm³)",
        value=float(sr_display.get("Wel_y_cm3", 0.0)),
        disabled=True,
        key=f"{key_prefix}_Wel_y"
    )
    s5.number_input(
        "Wel,z (cm³)",
        value=float(sr_display.get("Wel_z_cm3", 0.0)),
        disabled=True,
        key=f"{key_prefix}_Wel_z"
    )
    s6.number_input(
        "It (cm⁴)",
        value=float(sr_display.get("It_cm4", 0.0)),
        disabled=True,
        key=f"{key_prefix}_It"
    )

def render_section_properties_readonly(sr_display, key_prefix="db"):
    # Row 1: basic dims
    c1, c2, c3 = st.columns(3)
    c1.number_input("Depth h (mm)",  value=float(sr_display.get("h_mm", 0.0)),  disabled=True, key=f"{key_prefix}_h")
    c2.number_input("Width b (mm)",  value=float(sr_display.get("b_mm", 0.0)),  disabled=True, key=f"{key_prefix}_b")
    c3.number_input("Web thickness tw (mm)", value=float(sr_display.get("tw_mm", 0.0)), disabled=True, key=f"{key_prefix}_tw")

    # Row 2: flange & basic section
    c4, c5, c6 = st.columns(3)
    c4.number_input("Flange thickness tf (mm)", value=float(sr_display.get("tf_mm", 0.0)), disabled=True, key=f"{key_prefix}_tf")
    c5.number_input("Weight m (kg/m)", value=float(sr_display.get("m_kg_per_m", 0.0)), disabled=True, key=f"{key_prefix}_m")
    c6.number_input("Area A (mm²)", value=float(sr_display.get("A_mm2", 0.0)), disabled=True, key=f"{key_prefix}_A")

    # Row 3: shear areas
    c7, c8, c9 = st.columns(3)
    c7.number_input("Shear area Av,z (mm²)", value=float(sr_display.get("Avz_mm2", 0.0)), disabled=True, key=f"{key_prefix}_Avz")
    c8.number_input("Shear area Av,y (mm²)", value=float(sr_display.get("Avy_mm2", 0.0)), disabled=True, key=f"{key_prefix}_Avy")
    c9.empty()

    # Row 4: major axis y-y second moment & radius
    c10, c11, c12 = st.columns(3)
    c10.number_input("Iy (cm⁴)", value=float(sr_display.get("Iy_cm4", 0.0)), disabled=True, key=f"{key_prefix}_Iy")
    c11.number_input("iy (mm)",  value=float(sr_display.get("iy_mm", 0.0)), disabled=True, key=f"{key_prefix}_iy")
    c12.number_input("Wel,y (cm³)", value=float(sr_display.get("Wel_y_cm3", 0.0)), disabled=True, key=f"{key_prefix}_Wel_y")

    # Row 5: major axis plastic modulus
    c13, c14, c15 = st.columns(3)
    c13.number_input("Wpl,y (cm³)", value=float(sr_display.get("Wpl_y_cm3", 0.0)), disabled=True, key=f"{key_prefix}_Wpl_y")
    c14.empty()
    c15.empty()

    # Row 6: minor axis z-z second moment & radius
    c16, c17, c18 = st.columns(3)
    c16.number_input("Iz (cm⁴)", value=float(sr_display.get("Iz_cm4", 0.0)), disabled=True, key=f"{key_prefix}_Iz")
    c17.number_input("iz (mm)",  value=float(sr_display.get("iz_mm", 0.0)), disabled=True, key=f"{key_prefix}_iz")
    c18.number_input("Wel,z (cm³)", value=float(sr_display.get("Wel_z_cm3", 0.0)), disabled=True, key=f"{key_prefix}_Wel_z")

    # Row 7: minor axis plastic modulus
    c19, c20, c21 = st.columns(3)
    c19.number_input("Wpl,z (cm³)", value=float(sr_display.get("Wpl_z_cm3", 0.0)), disabled=True, key=f"{key_prefix}_Wpl_z")
    c20.empty()
    c21.empty()

    # Row 8: torsion & warping
    c22, c23, c24 = st.columns(3)
    c22.number_input("Torsion constant IT (×10³ mm⁴)", value=float(sr_display.get("IT_k_mm4", 0.0)), disabled=True, key=f"{key_prefix}_IT")
    c23.number_input("Torsion modulus WT (×10³ mm³)", value=float(sr_display.get("WT_k_mm3", 0.0)), disabled=True, key=f"{key_prefix}_WT")
    c24.number_input("Warping constant Iw (×10⁶ mm⁶)", value=float(sr_display.get("Iw_6_mm6", 0.0)), disabled=True, key=f"{key_prefix}_Iw")

    # Row 9: warping modulus
    c25, c26, c27 = st.columns(3)
    c25.number_input("Warping modulus Ww (×10³ mm⁴)", value=float(sr_display.get("Ww_k_mm4", 0.0)), disabled=True, key=f"{key_prefix}_Ww")
    c26.empty()
    c27.empty()

    # Row 10: plastic axial & shear resistances
    c28, c29, c30 = st.columns(3)
    c28.number_input("Npl,Rd (kN)", value=float(sr_display.get("Npl_Rd_kN", 0.0)), disabled=True, key=f"{key_prefix}_Npl")
    c29.number_input("Vpl,Rd,z (kN)", value=float(sr_display.get("Vpl_Rd_z_kN", 0.0)), disabled=True, key=f"{key_prefix}_Vplz")
    c30.number_input("Vpl,Rd,y (kN)", value=float(sr_display.get("Vpl_Rd_y_kN", 0.0)), disabled=True, key=f"{key_prefix}_Vply")

    # Row 11: bending resistances
    c31, c32, c33 = st.columns(3)
    c31.number_input("Mel,Rd,y (kNm)", value=float(sr_display.get("Mel_Rd_y_kNm", 0.0)), disabled=True, key=f"{key_prefix}_Mely")
    c32.number_input("Mpl,Rd,y (kNm)", value=float(sr_display.get("Mpl_Rd_y_kNm", 0.0)), disabled=True, key=f"{key_prefix}_Mply")
    c33.number_input("Mel,Rd,z (kNm)", value=float(sr_display.get("Mel_Rd_z_kNm", 0.0)), disabled=True, key=f"{key_prefix}_Melz")

    # Row 12: (optional) plastic z-z if available
    c34, c35, c36 = st.columns(3)
    c34.number_input("Mpl,Rd,z (kNm)", value=float(sr_display.get("Mpl_Rd_z_kNm", 0.0)), disabled=True, key=f"{key_prefix}_Mplz")
    c35.empty()
    c36.empty()

    # Row 13: buckling curves & classes
    c37, c38, c39 = st.columns(3)
    c37.number_input("Buckling about major axis y-y (α_y)", value=float(sr_display.get("alpha_y", 0.0)), disabled=True, key=f"{key_prefix}_alpha_y")
    c38.number_input("Buckling about minor axis z-z (α_z)", value=float(sr_display.get("alpha_z", 0.0)), disabled=True, key=f"{key_prefix}_alpha_z")
    c39.text_input("Web class in pure bending (CL Wb)", value=str(sr_display.get("CL_Wb", "n/a")), disabled=True, key=f"{key_prefix}_CLWb")

    c40, c41, c42 = st.columns(3)
    c40.text_input("Web class in uniform compression (CL Wc)", value=str(sr_display.get("CL_Wc", "n/a")), disabled=True, key=f"{key_prefix}_CLWc")
    c41.text_input("Flange class in uniform compression (CL Fb)", value=str(sr_display.get("CL_Fb", "n/a")), disabled=True, key=f"{key_prefix}_CLFb")
    c42.empty()


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

def render_loads_readonly(inputs: dict, torsion_supported: bool, key_prefix="rpt_load"):
    """
    Show the ULS loads and buckling factors in the same layout as the Loads tab,
    but read-only (for the Report tab).
    """
    if not inputs:
        st.info("No design loads found. Run the Loads tab first.")
        return

    st.markdown("### 3. Design forces and moments (ULS)")
    st.caption("Positive N = compression.")

    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        st.number_input(
            "Element length L (m)",
            value=float(inputs.get("L", 0.0)),
            disabled=True,
            key=f"{key_prefix}_L"
        )
    with r1c2:
        st.number_input(
            "Axial force N (kN)",
            value=float(inputs.get("N_kN", 0.0)),
            disabled=True,
            key=f"{key_prefix}_N"
        )
    with r1c3:
        st.number_input(
            "Shear V_y (kN)",
            value=float(inputs.get("Vy_kN", 0.0)),
            disabled=True,
            key=f"{key_prefix}_Vy"
        )

    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        st.number_input(
            "Shear V_z (kN)",
            value=float(inputs.get("Vz_kN", 0.0)),
            disabled=True,
            key=f"{key_prefix}_Vz"
        )
    with r2c2:
        st.number_input(
            "Bending M_y (kN·m) about y",
            value=float(inputs.get("My_kNm", 0.0)),
            disabled=True,
            key=f"{key_prefix}_My"
        )
    with r2c3:
        st.number_input(
            "Bending M_z (kN·m) about z",
            value=float(inputs.get("Mz_kNm", 0.0)),
            disabled=True,
            key=f"{key_prefix}_Mz"
        )

    st.markdown("### 3.1 Buckling effective length factors (K)")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.number_input(
            "K_y",
            value=float(inputs.get("K_y", 1.0)),
            disabled=True,
            key=f"{key_prefix}_Ky"
        )
    with k2:
        st.number_input(
            "K_z",
            value=float(inputs.get("K_z", 1.0)),
            disabled=True,
            key=f"{key_prefix}_Kz"
        )
    with k3:
        st.number_input(
            "K_LT",
            value=float(inputs.get("K_LT", 1.0)),
            disabled=True,
            key=f"{key_prefix}_KLT"
        )
    with k4:
        st.number_input(
            "K_T",
            value=float(inputs.get("K_T", 1.0)),
            disabled=True,
            key=f"{key_prefix}_KT"
        )

    if torsion_supported:
        st.markdown("### 3.2 Torsion (only for open I/H/U sections)")
        st.number_input(
            "Torsion T_x (kN·m)",
            value=float(inputs.get("Tx_kNm", 0.0)),
            disabled=True,
            key=f"{key_prefix}_Tx"
        )

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

def get_beam_summary_for_diagrams(x, V, M, delta, L):
    """
    From diagram arrays, extract:
      - max deflection and its location
      - comparison vs L/300, L/600, L/900
      - max bending moment and location
      - shear at the max-moment location
      - support reactions (from ends of V)
    Units:
      x in m, V in kN, M in kN·m, delta in m, L in m
    """
    summary = {}

    # ---- Max deflection ----
    if delta is None:
        summary["defl_available"] = False
    else:
        try:
            idx_defl_max = int(np.nanargmax(np.abs(delta)))
            w_max_m = float(delta[idx_defl_max])
            x_defl_max = float(x[idx_defl_max])
            summary["defl_available"] = True
            summary["w_max_m"] = w_max_m
            summary["w_max_mm"] = w_max_m * 1000.0
            summary["x_defl_max"] = x_defl_max

            L_mm = float(L) * 1000.0 if L else (float(x[-1] - x[0]) * 1000.0)
            summary["limit_L300"] = L_mm / 300.0 if L_mm > 0 else None
            summary["limit_L600"] = L_mm / 600.0 if L_mm > 0 else None
            summary["limit_L900"] = L_mm / 900.0 if L_mm > 0 else None
        except Exception:
            summary["defl_available"] = False

    # ---- Max bending moment ----
    if M is not None:
        try:
            idx_M_max = int(np.nanargmax(np.abs(M)))
            M_max = float(M[idx_M_max])
            x_M_max = float(x[idx_M_max])
            summary["M_max"] = M_max
            summary["x_M_max"] = x_M_max
            summary["idx_M_max"] = idx_M_max
        except Exception:
            summary["M_max"] = None
    else:
        summary["M_max"] = None

    # ---- Shear at max-moment location ----
    if summary.get("M_max") is not None and V is not None:
        idx_M_max = summary.get("idx_M_max", 0)
        try:
            summary["V_at_Mmax"] = float(V[idx_M_max])
        except Exception:
            summary["V_at_Mmax"] = None
    else:
        summary["V_at_Mmax"] = None

    # ---- Support reactions (approx from ends of V) ----
    if V is not None and len(V) >= 2:
        try:
            summary["R_left"] = float(V[0])          # kN (signed)
            summary["R_right"] = float(-V[-1])       # kN (signed)
        except Exception:
            summary["R_left"] = None
            summary["R_right"] = None
    else:
        summary["R_left"] = None
        summary["R_right"] = None

    return summary


def render_beam_diagrams_panel():
    """
    Draw V(x) and M(x) diagrams.
    Show deflection and internal force summaries ABOVE the diagrams
    in the same 'box' style as other inputs.
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

    # ---- Section stiffness for deflection ----
    E = 210e9  # Pa

    I_y_m4 = float(sr_display.get("I_y_cm4", 0.0)) * 1e-8 if sr_display else 0.0
    I_z_m4 = float(sr_display.get("I_z_cm4", 0.0)) * 1e-8 if sr_display else 0.0

    bending_axis = st.session_state.get("bending_axis", "y")
    if bending_axis == "z":
        I_m4 = I_z_m4
    else:
        I_m4 = I_y_m4

    if I_m4 <= 0:
        I_m4 = None  # allow V/M but disable deflection if no inertia

    # arguments in the same order as selected_case["inputs"]
    args = [input_vals[k] for k in selected_case["inputs"].keys()]
    x, V, M, delta = diag_func(*args, E=E, I=I_m4)

    # extract L from inputs (fallback to x-range)
    L_val = float(input_vals.get("L", 0.0))
    if (not L_val or L_val <= 0.0) and x is not None and len(x) > 1:
        L_val = float(x[-1] - x[0])

    # ---- Summary from diagrams (δ_max, M_max, shear, reactions) ----
    summary = get_beam_summary_for_diagrams(x, V, M, delta, L_val)
    # Store diagram summary for use in Report tab
    st.session_state["diag_summary"] = summary

    # =====================================================
    # SUMMARY BLOCKS ABOVE DIAGRAMS (box style like inputs)
    # =====================================================
    st.markdown("### Diagram-based summary")

    # ---- Deflection summary ----
        # ---- Deflection summary ----
    if summary.get("defl_available"):
        # First line: ONLY maximum deflection (same box style)
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            st.number_input(
                "Maximum deflection δ_max (mm)",
                value=float(summary["w_max_mm"]),
                disabled=True,
                key="diag_dmax"
            )
        with r1c2:
            st.empty()
        with r1c3:
            st.empty()

        # Second line: L/300, L/600, L/900 (mm)
        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            st.number_input(
                "Limit L/300 (mm)",
                value=float(summary["limit_L300"]) if summary.get("limit_L300") else 0.0,
                disabled=True,
                key="diag_L300"
            )
        with r2c2:
            st.number_input(
                "Limit L/600 (mm)",
                value=float(summary["limit_L600"]) if summary.get("limit_L600") else 0.0,
                disabled=True,
                key="diag_L600"
            )
        with r2c3:
            st.number_input(
                "Limit L/900 (mm)",
                value=float(summary["limit_L900"]) if summary.get("limit_L900") else 0.0,
                disabled=True,
                key="diag_L900"
            )
    else:
        st.info("Deflection summary not available: missing section inertia or deflection formula.")

    # ---- Internal forces summary ----
    st.markdown(
    "<div style='font-size:0.9rem;font-weight:600;margin-top:0.3rem;'>"
    "Internal forces summary (from diagrams)"
    "</div>",
    unsafe_allow_html=True
)

    f1, f2, f3, f4 = st.columns(4)

    with f1:
        st.number_input(
            "Max bending moment M_max (kN·m)",
            value=float(summary["M_max"]) if summary.get("M_max") is not None else 0.0,
            disabled=True,
            key="diag_Mmax"
        )
    with f2:
        st.number_input(
            "Shear at M_max, V(x_Mmax) (kN)",
            value=float(summary["V_at_Mmax"]) if summary.get("V_at_Mmax") is not None else 0.0,
            disabled=True,
            key="diag_VatMmax"
        )
    with f3:
        st.number_input(
            "Left reaction R_A (kN)",
            value=float(summary["R_left"]) if summary.get("R_left") is not None else 0.0,
            disabled=True,
            key="diag_Rleft"
        )
    with f4:
        st.number_input(
            "Right reaction R_B (kN)",
            value=float(summary["R_right"]) if summary.get("R_right") is not None else 0.0,
            disabled=True,
            key="diag_Rright"
        )

    # =====================================================
    # DIAGRAMS (labels with smaller font)
    # =====================================================
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

        import io
        buf_v = io.BytesIO()
        fig1.savefig(buf_v, format="png", dpi=200, bbox_inches="tight")
        buf_v.seek(0)
        st.session_state["diag_V_png"] = buf_v.getvalue()

        st.pyplot(fig1)

    with colM:
        st.markdown("#### Bending moment diagram M(x)")
        fig2, ax2 = plt.subplots()
        ax2.plot(x, M)
        ax2.axhline(0, linewidth=1)
        ax2.set_xlabel("x (m)")
        ax2.set_ylabel("M (kN·m)")
        ax2.grid(True)

        buf_m = io.BytesIO()
        fig2.savefig(buf_m, format="png", dpi=200, bbox_inches="tight")
        buf_m.seek(0)
        st.session_state["diag_M_png"] = buf_m.getvalue()

        st.pyplot(fig2)

    # ---- Optional deflection diagram ----
    show_defl = st.checkbox("Show deflection diagram δ(x)", value=False, key="show_defl_diag")

    if show_defl:
        if delta is None or I_m4 is None:
            st.warning("Deflection diagram not available: missing Iy or case deflection formula.")
            return

        colD, colEmpty = st.columns(2)

        with colD:
            st.markdown("#### Deflection diagram δ(x)")
            fig3, ax3 = plt.subplots(figsize=(6, 3.5))
            ax3.plot(x, delta * 1000.0)  # mm
            ax3.axhline(0, linewidth=1)
            ax3.set_xlabel("x (m)")
            ax3.set_ylabel("δ (mm)")
            ax3.grid(True)

            buf_d = io.BytesIO()
            fig3.savefig(buf_d, format="png", dpi=200, bbox_inches="tight")
            buf_d.seek(0)
            st.session_state["diag_D_png"] = buf_d.getvalue()

            st.pyplot(fig3)

        with colEmpty:
            st.empty()

# =========================================================
# REPORT TAB & PDF HELPERS — ENGISNAP FULL REPORT
# =========================================================

def render_loads_readonly(inputs: dict, torsion_supported: bool, key_prefix="rpt_load"):
    """
    Read-only version of the Loads form, for the Report tab.
    """
    if not inputs:
        st.info("No design loads found. Run the Loads tab first.")
        return

    st.markdown("### 3. Loading")

    # 3.1 Load description (short text)
    with st.expander("3.1 Load description", expanded=True):
        ready_case = st.session_state.get("ready_selected_case")
        if ready_case:
            st.write(f"Ready-case: **{ready_case.get('key','')} — {ready_case.get('label','')}**")
        L = inputs.get("L", 0.0)
        st.write(f"Span length L = **{L:.3f} m**")
        st.write(f"N = {inputs.get('N_kN', 0.0):.3f} kN, "
                 f"Vy = {inputs.get('Vy_kN', 0.0):.3f} kN, "
                 f"Vz = {inputs.get('Vz_kN', 0.0):.3f} kN, "
                 f"My = {inputs.get('My_kNm', 0.0):.3f} kNm, "
                 f"Mz = {inputs.get('Mz_kNm', 0.0):.3f} kNm")

    # 3.2 Internal forces summary (from diagrams / loads)
    st.markdown("#### 3.2 Internal forces summary (ULS)")
    c1, c2, c3, c4 = st.columns(4)
    Vmax = st.session_state.get("diag_Vmax_kN")
    Mmax = st.session_state.get("diag_Mmax_kNm")
    R1 = st.session_state.get("diag_R1_kN")
    R2 = st.session_state.get("diag_R2_kN")
    x_Mmax = st.session_state.get("diag_x_Mmax_m")
    delta_max_mm = st.session_state.get("diag_delta_max_mm")

    with c1:
        st.text_input("V_max (kN)", value=f"{Vmax:.3f}" if Vmax is not None else "n/a",
                      disabled=True, key=f"{key_prefix}_Vmax")
    with c2:
        st.text_input("M_max (kN·m)", value=f"{Mmax:.3f}" if Mmax is not None else "n/a",
                      disabled=True, key=f"{key_prefix}_Mmax")
    with c3:
        st.text_input("R1 support (kN)", value=f"{R1:.3f}" if R1 is not None else "n/a",
                      disabled=True, key=f"{key_prefix}_R1")
    with c4:
        st.text_input("R2 support (kN)", value=f"{R2:.3f}" if R2 is not None else "n/a",
                      disabled=True, key=f"{key_prefix}_R2")

    c5, c6 = st.columns(2)
    with c5:
        st.text_input("Position of M_max (m)",
                      value=f"{x_Mmax:.3f}" if x_Mmax is not None else "n/a",
                      disabled=True, key=f"{key_prefix}_x_Mmax")
    with c6:
        st.text_input("δ_max (mm)",
                      value=f"{delta_max_mm:.3f}" if delta_max_mm is not None else "n/a",
                      disabled=True, key=f"{key_prefix}_delta_max")

    # 3.3 Design forces & effective lengths (detailed numeric)
    st.markdown("#### 3.3 Design forces and effective lengths (ULS)")

    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        st.number_input(
            "Element length L (m)",
            value=float(inputs.get("L", 0.0)),
            disabled=True,
            key=f"{key_prefix}_L"
        )
    with r1c2:
        st.number_input(
            "Axial force N (kN)",
            value=float(inputs.get("N_kN", 0.0)),
            disabled=True,
            key=f"{key_prefix}_N"
        )
    with r1c3:
        st.number_input(
            "Shear V_y (kN)",
            value=float(inputs.get("Vy_kN", 0.0)),
            disabled=True,
            key=f"{key_prefix}_Vy"
        )

    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        st.number_input(
            "Shear V_z (kN)",
            value=float(inputs.get("Vz_kN", 0.0)),
            disabled=True,
            key=f"{key_prefix}_Vz"
        )
    with r2c2:
        st.number_input(
            "Bending M_y (kN·m)",
            value=float(inputs.get("My_kNm", 0.0)),
            disabled=True,
            key=f"{key_prefix}_My"
        )
    with r2c3:
        st.number_input(
            "Bending M_z (kN·m)",
            value=float(inputs.get("Mz_kNm", 0.0)),
            disabled=True,
            key=f"{key_prefix}_Mz"
        )

    st.markdown("##### Buckling effective length factors (K)")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.number_input(
            "K_y",
            value=float(inputs.get("K_y", 1.0)),
            disabled=True,
            key=f"{key_prefix}_Ky"
        )
    with k2:
        st.number_input(
            "K_z",
            value=float(inputs.get("K_z", 1.0)),
            disabled=True,
            key=f"{key_prefix}_Kz"
        )
    with k3:
        st.number_input(
            "K_LT",
            value=float(inputs.get("K_LT", 1.0)),
            disabled=True,
            key=f"{key_prefix}_KLT"
        )
    with k4:
        st.number_input(
            "K_T",
            value=float(inputs.get("K_T", 1.0)),
            disabled=True,
            key=f"{key_prefix}_KT"
        )

    if torsion_supported:
        st.markdown("##### Torsion")
        st.number_input(
            "Torsion T_x (kN·m)",
            value=float(inputs.get("Tx_kNm", 0.0)),
            disabled=True,
            key=f"{key_prefix}_Tx"
        )


def build_pdf_report(meta, material, sr_display, inputs, df_rows, overall_ok, governing, extras):
    """
    Full ENGISNAP report PDF, with sections 1–9 as discussed.
    """
    if not HAS_RL:
        return None

    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )
    styles = getSampleStyleSheet()
    N = styles["Normal"]
    H = styles["Heading3"]

    story = []

    # -----------------------
    # Basic data
    # -----------------------
    if meta:
        doc_title, project_name, position, requested_by, revision, run_date = meta
    else:
        doc_title = "Beam check"
        project_name = position = requested_by = revision = ""
        run_date = date.today()

    fam = sr_display.get("family", "") if sr_display else ""
    name = sr_display.get("name", "") if sr_display else ""
    fy = material_to_fy(material)
    gov_check, gov_util = governing if governing else (None, None)
    status_txt = "OK" if overall_ok else "NOT OK"

    L = inputs.get("L", 0.0) if inputs else 0.0
    Ky = inputs.get("K_y", 1.0) if inputs else 1.0
    Kz = inputs.get("K_z", 1.0) if inputs else 1.0
    KLT = inputs.get("K_LT", 1.0) if inputs else 1.0
    Leff_y = Ky * L
    Leff_z = Kz * L
    Leff_LT = KLT * L

    Vmax = st.session_state.get("diag_Vmax_kN")
    Mmax = st.session_state.get("diag_Mmax_kNm")
    R1 = st.session_state.get("diag_R1_kN")
    R2 = st.session_state.get("diag_R2_kN")
    x_Mmax = st.session_state.get("diag_x_Mmax_m")
    delta_max_mm = st.session_state.get("diag_delta_max_mm")

    # -----------------------
    # 1. Project information
    # -----------------------
    story.append(Paragraph("1. Project information", H))
    story.append(Paragraph(f"Project name: {project_name}", N))
    story.append(Paragraph(f"Designer: {requested_by}", N))
    story.append(Paragraph(f"Date: {run_date}", N))
    story.append(Paragraph("App: EngiSnap – Beam design (prototype)", N))
    story.append(Spacer(1, 6))
    story.append(Paragraph("National Annex: (not specified)", N))
    story.append(Paragraph("Notes / comments: –", N))
    story.append(Spacer(1, 12))

    # -----------------------
    # 2. Beam definition & material
    # -----------------------
    story.append(Paragraph("2. Beam definition & material", H))

    # 2.1 Member definition
    story.append(Paragraph("2.1 Member definition", styles["Heading4"]))
    ready_case = st.session_state.get("ready_selected_case")
    member_type = "Standard beam"
    if ready_case:
        member_type = f"Beam case: {ready_case.get('key','')} — {ready_case.get('label','')}"

    story.append(Paragraph(f"Member type: {member_type}", N))
    story.append(Paragraph(f"Span length L = {L:.3f} m", N))
    story.append(Paragraph("Support conditions: simply supported (from ready case)", N))
    story.append(Paragraph(
        f"Effective lengths: L_y = {Leff_y:.3f} m, "
        f"L_z = {Leff_z:.3f} m, "
        f"L_LT = {Leff_LT:.3f} m",
        N
    ))
    story.append(Spacer(1, 6))

    # 2.2 Material
    story.append(Paragraph("2.2 Material", styles["Heading4"]))
    story.append(Paragraph(f"Steel grade: {material}", N))
    story.append(Paragraph(f"f_y = {fy:.0f} MPa, f_u ≈ (not specified)", N))
    story.append(Paragraph("E = 210000 MPa, G = 81000 MPa", N))
    story.append(Paragraph("Partial factors: γ_M0 = 1.0, γ_M1 = 1.0 (per DB assumption)", N))
    story.append(Spacer(1, 6))

    # 2.3 Cross-section
    story.append(Paragraph("2.3 Cross-section", styles["Heading4"]))
    story.append(Paragraph(f"Family: {fam}", N))
    story.append(Paragraph(f"Size: {name}", N))

    if sr_display:
        A_mm2 = sr_display.get("A_mm2", 0.0)
        Iy_cm4 = sr_display.get("Iy_cm4", 0.0)
        Iz_cm4 = sr_display.get("Iz_cm4", 0.0)
        Wel_y_cm3 = sr_display.get("Wel_y_cm3", 0.0)
        Wel_z_cm3 = sr_display.get("Wel_z_cm3", 0.0)
        It_cm4 = sr_display.get("It_cm4", 0.0)
        Iw_cm6 = sr_display.get("Iw_cm6", sr_display.get("Iw_6_mm6", 0.0))

        story.append(Paragraph(
            f"A = {A_mm2:.1f} mm²; Iy = {Iy_cm4:.1f} cm⁴; Iz = {Iz_cm4:.1f} cm⁴",
            N
        ))
        story.append(Paragraph(
            f"Wel,y = {Wel_y_cm3:.1f} cm³; Wel,z = {Wel_z_cm3:.1f} cm³",
            N
        ))
        story.append(Paragraph(
            f"It ≈ {It_cm4:.1f} cm⁴; Iw ≈ {Iw_cm6:.1f} cm⁶",
            N
        ))

    story.append(Spacer(1, 6))

    # cross-section image
    img_path = get_section_image(fam) if fam else None
    if img_path:
        try:
            story.append(Spacer(1, 4))
            story.append(Image(img_path, width=70 * mm, preserveAspectRatio=True, hAlign="CENTER"))
        except Exception:
            pass

    story.append(Spacer(1, 12))

    # -----------------------
    # 3. Loading (summary)
    # -----------------------
    story.append(Paragraph("3. Loading", H))

    story.append(Paragraph("3.1 Load description", styles["Heading4"]))
    if ready_case:
        story.append(Paragraph(
            f"Ready-case: {ready_case.get('key','')} — {ready_case.get('label','')}",
            N
        ))
    if inputs:
        story.append(Paragraph(
            f"N = {inputs.get('N_kN', 0.0):.3f} kN, "
            f"Vy = {inputs.get('Vy_kN', 0.0):.3f} kN, "
            f"Vz = {inputs.get('Vz_kN', 0.0):.3f} kN, "
            f"My = {inputs.get('My_kNm', 0.0):.3f} kNm, "
            f"Mz = {inputs.get('Mz_kNm', 0.0):.3f} kNm",
            N
        ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("3.2 Internal forces summary", styles["Heading4"]))
    data_forces = [
        ["Quantity", "Value"],
        ["V_max (kN)", f"{Vmax:.3f}" if Vmax is not None else "n/a"],
        ["M_max (kN·m)", f"{Mmax:.3f}" if Mmax is not None else "n/a"],
        ["R1 (kN)", f"{R1:.3f}" if R1 is not None else "n/a"],
        ["R2 (kN)", f"{R2:.3f}" if R2 is not None else "n/a"],
        ["x(M_max) (m)", f"{x_Mmax:.3f}" if x_Mmax is not None else "n/a"],
        ["δ_max (mm)", f"{delta_max_mm:.3f}" if delta_max_mm is not None else "n/a"],
    ]
    from reportlab.platypus import Table, TableStyle
    forces_table = Table(data_forces, hAlign="LEFT")
    forces_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    story.append(forces_table)
    story.append(Spacer(1, 12))

    # 3.3 Diagrams are in section 6 (later)
    story.append(Paragraph("3.3 Diagrams", styles["Heading4"]))
    story.append(Paragraph("See section 6: Diagrams & cross-section view.", N))
    story.append(Spacer(1, 12))

    # -----------------------
    # 4. Serviceability (SLS)
    # -----------------------
    story.append(Paragraph("4. Serviceability (SLS)", H))
    story.append(Paragraph("4.1 Maximum deflection", styles["Heading4"]))
    story.append(Paragraph(f"δ_max = {delta_max_mm:.3f} mm" if delta_max_mm is not None else "δ_max = n/a", N))
    if L > 0:
        Lmm = L * 1000.0
        limit_300 = Lmm / 300.0
        limit_600 = Lmm / 600.0
        limit_900 = Lmm / 900.0
        story.append(Paragraph(f"L/300 = {limit_300:.1f} mm", N))
        story.append(Paragraph(f"L/600 = {limit_600:.1f} mm", N))
        story.append(Paragraph(f"L/900 = {limit_900:.1f} mm", N))
    else:
        story.append(Paragraph("Span length not available → limits not evaluated.", N))
    story.append(Spacer(1, 6))

    story.append(Paragraph("4.2 SLS pass / fail", styles["Heading4"]))
    story.append(Paragraph("Deflection check not fully implemented – indicative only.", N))
    story.append(Spacer(1, 12))

    # -----------------------
    # 5. Section classification (EC3 §5)
    # -----------------------
    story.append(Paragraph("5. Section classification (EC3 §5)", H))
    if sr_display:
        story.append(Paragraph(
            f"Flange class (bending): {sr_display.get('flange_class_db','n/a')}",
            N
        ))
        story.append(Paragraph(
            f"Web class (bending): {sr_display.get('web_class_bending_db','n/a')}",
            N
        ))
        story.append(Paragraph(
            f"Web class (compression): {sr_display.get('web_class_compression_db','n/a')}",
            N
        ))
        story.append(Paragraph("Overall cross-section class: (not explicitly derived)", N))
    else:
        story.append(Paragraph("No section classification data available.", N))
    story.append(Spacer(1, 12))

    # -----------------------
    # 6. Verification of cross-section strength (ULS)
    # -----------------------
    story.append(Paragraph("6. Verification of cross-section strength (ULS)", H))
    story.append(Paragraph(
        "(1) Tension; (2) Compression; (3)-(4) Bending; "
        "(5)-(6) Shear; (7)-(8) Bending + shear; "
        "(9)-(14) Bending, shear and axial force (interaction).",
        N
    ))
    story.append(Spacer(1, 6))

    if df_rows is not None:
        data_checks = [["Check", "Applied", "Resistance", "Utilisation", "Status"]]
        for idx, row in df_rows.iterrows():
            data_checks.append([
                str(idx),
                str(row["Applied"]),
                str(row["Resistance"]),
                str(row["Utilization"]),
                str(row["Status"]),
            ])
        checks_table = Table(data_checks, hAlign="LEFT", repeatRows=1)
        checks_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]))
        story.append(checks_table)
    story.append(Spacer(1, 12))

    # -----------------------
    # 7. Verification of member stability (ULS buckling)
    # -----------------------
    story.append(Paragraph("7. Verification of member stability (ULS buckling)", H))
    story.append(Paragraph(
        "(15)-(16) Flexural buckling; (17) Torsional / torsional-flexural buckling; "
        "(18) Lateral–torsional buckling; (19)-(22) Buckling interaction.",
        N
    ))
    story.append(Spacer(1, 4))
    if extras and extras.get("buck_results"):
        for axis_label, Ncr, lambda_bar, chi, N_b_Rd_N, status in extras["buck_results"]:
            if N_b_Rd_N:
                story.append(Paragraph(
                    f"Axis {axis_label}: Ncr = {Ncr/1e3:.2f} kN, "
                    f"λ̄ = {lambda_bar:.3f}, χ = {chi:.3f}, "
                    f"Nb,Rd = {N_b_Rd_N/1e3:.2f} kN → {status}",
                    N
                ))
    else:
        story.append(Paragraph("Buckling results not available.", N))
    story.append(Spacer(1, 12))

    # -----------------------
    # 8. Summary of checks
    # -----------------------
    story.append(Paragraph("8. Summary of checks", H))
    data_summary = [
        ["Check group", "Status", "Governing", "Ratio"],
    ]
    # We only have partial info; fill what we can.
    data_summary.append([
        "Cross-section ULS",
        status_txt,
        gov_check or "n/a",
        f"{gov_util:.3f}" if gov_util is not None else "n/a",
    ])
    data_summary.append(["SLS (deflection)", "not fully implemented", "L/300", "n/a"])
    data_summary.append(["Buckling", "see section 7", "–", "–"])
    data_summary.append(["Global", status_txt, gov_check or "n/a",
                         f"{gov_util:.3f}" if gov_util is not None else "n/a"])

    summ_table = Table(data_summary, hAlign="LEFT")
    summ_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    story.append(summ_table)
    story.append(Spacer(1, 12))

    # -----------------------
    # 9. Appendix (equations / references)
    # -----------------------
    story.append(PageBreak())
    story.append(Paragraph("9. Appendix & references", H))
    story.append(Paragraph("This prototype does not print all intermediate formula steps.", N))
    story.append(Paragraph(
        "Equations and clause references follow EN 1993-1-1, EN 1990 and EN 1991 series "
        "(see also National Annex).",
        N
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Key references:", styles["Heading4"]))
    story.append(Paragraph("1) EN 1993-1-1:2005 + A1:2014 – Eurocode 3: Design of steel structures – Part 1-1.", N))
    story.append(Paragraph("2) EN 1990:2002 – Eurocode: Basis of structural design.", N))
    story.append(Paragraph("3) EN 1991 series – Actions on structures.", N))
    story.append(Paragraph("4) National Annex to EN 1993-1-1 (where applicable).", N))

    doc.build(story)
    buffer.seek(0)
    return buffer


def render_report_tab():
    """
    FULL ENGISNAP report in the app, matching the PDF structure.
    """
    sr_display = st.session_state.get("sr_display")
    inputs = st.session_state.get("inputs")
    df_rows = st.session_state.get("df_rows")
    overall_ok = st.session_state.get("overall_ok", True)
    governing = st.session_state.get("governing", (None, None))
    extras = st.session_state.get("extras")
    meta = st.session_state.get("meta")
    material = st.session_state.get("material", "S355")

    if sr_display is None or inputs is None or df_rows is None or meta is None:
        st.info("To see the report: select a section, define loads, run the check, then return here.")
        return

    doc_title, project_name, position, requested_by, revision, run_date = meta
    fam = sr_display.get("family", "")
    name = sr_display.get("name", "")
    fy = material_to_fy(material)
    gov_check, gov_util = governing
    status_txt = "OK" if overall_ok else "NOT OK"
    L = inputs.get("L", 0.0)
    Ky = inputs.get("K_y", 1.0)
    Kz = inputs.get("K_z", 1.0)
    KLT = inputs.get("K_LT", 1.0)
    Leff_y = Ky * L
    Leff_z = Kz * L
    Leff_LT = KLT * L

    Vmax = st.session_state.get("diag_Vmax_kN")
    Mmax = st.session_state.get("diag_Mmax_kNm")
    R1 = st.session_state.get("diag_R1_kN")
    R2 = st.session_state.get("diag_R2_kN")
    x_Mmax = st.session_state.get("diag_x_Mmax_m")
    delta_max_mm = st.session_state.get("diag_delta_max_mm")

    # ---- PDF download ----
    if HAS_RL:
        pdf_buf = build_pdf_report(meta, material, sr_display, inputs, df_rows, overall_ok, governing, extras)
        if pdf_buf:
            st.download_button(
                "Download full ENGISNAP report (PDF)",
                data=pdf_buf,
                file_name="EngiSnap_Beam_Report.pdf",
                mime="application/pdf",
                key="rpt_pdf_btn",
            )
    else:
        st.warning("PDF export not available (reportlab not installed).")

    st.markdown("## 1. Project information")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.text_input("Project name", value=str(project_name), disabled=True, key="rpt_proj_name")
        st.text_input("Designer", value=str(requested_by), disabled=True, key="rpt_designer")
    with c2:
        st.text_input("Document title", value=str(doc_title), disabled=True, key="rpt_doc_title")
        st.text_input("Revision", value=str(revision), disabled=True, key="rpt_revision")
    with c3:
        st.text_input("Date", value=str(run_date), disabled=True, key="rpt_date")
        st.text_input("App", value="EngiSnap – Beam design (prototype)", disabled=True, key="rpt_app")

    st.text_input("National Annex", value="(not specified)", disabled=True, key="rpt_na")
    st.text_area("Notes / comments", value="–", disabled=True, key="rpt_notes")
    st.markdown("---")

    st.markdown("## 2. Beam definition & material")

    # 2.1 Member definition
    st.markdown("### 2.1 Member definition")
    ready_case = st.session_state.get("ready_selected_case")
    member_type = "Standard beam"
    if ready_case:
        member_type = f"Beam case: {ready_case.get('key','')} — {ready_case.get('label','')}"

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.text_input("Member type", value=member_type, disabled=True, key="rpt_mem_type")
        st.number_input("Span length L (m)", value=float(L), disabled=True, key="rpt_L")
    with mc2:
        st.text_input("Support conditions", value="Simply supported (from ready case)", disabled=True, key="rpt_support")
        st.number_input("L_y (m)", value=float(Leff_y), disabled=True, key="rpt_Leff_y")
    with mc3:
        st.number_input("L_z (m)", value=float(Leff_z), disabled=True, key="rpt_Leff_z")
        st.number_input("L_LT (m)", value=float(Leff_LT), disabled=True, key="rpt_Leff_LT")

    # 2.2 Material
    st.markdown("### 2.2 Material")
    mat1, mat2, mat3 = st.columns(3)
    with mat1:
        st.text_input("Steel grade", value=material, disabled=True, key="rpt_steel")
        st.number_input("f_y (MPa)", value=float(fy), disabled=True, key="rpt_fy")
    with mat2:
        st.number_input("E (MPa)", value=210000.0, disabled=True, key="rpt_E")
        st.number_input("G (MPa)", value=81000.0, disabled=True, key="rpt_G")
    with mat3:
        st.number_input("γ_M0", value=1.0, disabled=True, key="rpt_gM0")
        st.number_input("γ_M1", value=1.0, disabled=True, key="rpt_gM1")

    # 2.3 Cross-section
    st.markdown("### 2.3 Cross-section")
    cs1, cs2 = st.columns(2)
    with cs1:
        st.text_input("Family", value=fam, disabled=True, key="rpt_cs_fam")
        st.text_input("Size", value=name, disabled=True, key="rpt_cs_size")
        render_section_summary_like_props(material, sr_display, key_prefix="rpt_sum")
    with cs2:
        img_path = get_section_image(fam)
        if img_path:
            st.markdown("Cross-section view")
            st.image(img_path, width=260)

    with st.expander("Key properties", expanded=False):
        render_section_properties_readonly(sr_display, key_prefix="rpt_props")

    st.markdown("---")

    # 3. Loading (reuse helper)
    torsion_supported = supports_torsion_and_warping(fam)
    render_loads_readonly(inputs, torsion_supported, key_prefix="rpt_load")
    st.markdown("---")

    # 4. Serviceability (SLS)
    st.markdown("## 4. Serviceability (SLS)")
    st.markdown("### 4.1 Maximum deflection")
    if L > 0:
        Lmm = L * 1000.0
        limit_300 = Lmm / 300.0
        limit_600 = Lmm / 600.0
        limit_900 = Lmm / 900.0
    else:
        limit_300 = limit_600 = limit_900 = 0.0

    dc1, dc2, dc3, dc4 = st.columns(4)
    with dc1:
        st.text_input("δ_max (mm)",
                      value=f"{delta_max_mm:.3f}" if delta_max_mm is not None else "n/a",
                      disabled=True, key="rpt_delta_max")
    with dc2:
        st.text_input("L/300 (mm)",
                      value=f"{limit_300:.1f}" if L > 0 else "n/a",
                      disabled=True, key="rpt_L300")
    with dc3:
        st.text_input("L/600 (mm)",
                      value=f"{limit_600:.1f}" if L > 0 else "n/a",
                      disabled=True, key="rpt_L600")
    with dc4:
        st.text_input("L/900 (mm)",
                      value=f"{limit_900:.1f}" if L > 0 else "n/a",
                      disabled=True, key="rpt_L900")

    st.markdown("### 4.2 SLS pass / fail")
    st.info("Deflection checks are indicative only in this prototype; full SLS not yet implemented.")
    st.markdown("---")

    # 5. Section classification
    st.markdown("## 5. Section classification (EC3 §5)")
    if sr_display:
        st.text_input("Flange class (bending)",
                      value=sr_display.get("flange_class_db", "n/a"),
                      disabled=True, key="rpt_flange_class")
        st.text_input("Web class (bending)",
                      value=sr_display.get("web_class_bending_db", "n/a"),
                      disabled=True, key="rpt_web_bend")
        st.text_input("Web class (compression)",
                      value=sr_display.get("web_class_compression_db", "n/a"),
                      disabled=True, key="rpt_web_comp")
        st.text_input("Overall cross-section class",
                      value="(not explicitly derived)",
                      disabled=True, key="rpt_cs_class")
    else:
        st.info("No section classification data available.")
    st.markdown("---")

    # 6. Cross-section strength
    st.markdown("## 6. Verification of cross-section strength (ULS)")
    st.caption(
        "(1) Tension; (2) Compression; (3)-(4) Bending; (5)-(6) Shear; "
        "(7)-(8) Bending + shear; (9)-(14) Bending, shear and axial force."
    )

    r1, r2, r3 = st.columns(3)
    with r1:
        st.text_input("Overall status", value=status_txt, disabled=True, key="rpt_status")
    with r2:
        st.text_input("Governing check", value=gov_check or "n/a", disabled=True, key="rpt_gov")
    with r3:
        st.text_input("Max utilisation",
                      value=f"{gov_util:.3f}" if gov_util is not None else "n/a",
                      disabled=True, key="rpt_util")

    st.markdown("### 6.1 Detailed checks")
    def _hl(row):
        s = row["Status"]
        if s == "OK":
            color = "background-color: #e6f7e6"
        elif s == "EXCEEDS":
            color = "background-color: #fde6e6"
        else:
            color = "background-color: #f0f0f0"
        return [color] * len(row)

    st.write(df_rows.style.apply(_hl, axis=1))
    st.markdown("---")

    # 7. Member stability
    st.markdown("## 7. Verification of member stability (ULS buckling)")
    st.caption(
        "(15)-(16) Flexural buckling; (17) Torsional / torsional-flexural; "
        "(18) Lateral–torsional buckling; (19)-(22) Buckling interaction."
    )
    if extras and extras.get("buck_results"):
        for axis_label, Ncr, lambda_bar, chi, N_b_Rd_N, status in extras["buck_results"]:
            if N_b_Rd_N:
                st.write(
                    f"Axis {axis_label}: Ncr = {Ncr/1e3:.2f} kN, "
                    f"λ̄ = {lambda_bar:.3f}, χ = {chi:.3f}, "
                    f"Nb,Rd = {N_b_Rd_N/1e3:.2f} kN → {status}"
                )
    else:
        st.info("Buckling results not available.")
    st.markdown("---")

    # 8. Summary of checks
    st.markdown("## 8. Summary of checks")
    data_summary = [
        ["Check group", "Status", "Governing", "Ratio"],
        ["Cross-section ULS", status_txt, gov_check or "n/a",
         f"{gov_util:.3f}" if gov_util is not None else "n/a"],
        ["SLS (deflection)", "not fully implemented", "L/300", "n/a"],
        ["Buckling", "see section 7", "–", "–"],
        ["Global", status_txt, gov_check or "n/a",
         f"{gov_util:.3f}" if gov_util is not None else "n/a"],
    ]
    summ_df = pd.DataFrame(data_summary[1:], columns=data_summary[0])
    st.dataframe(summ_df, use_container_width=True)
    st.markdown("---")

    # 9. References
    st.markdown("## 9. Appendix & references")
    st.markdown(
        """
1) EN 1993-1-1:2005 + A1:2014 – Eurocode 3: Design of steel structures – Part 1-1.  
2) EN 1990:2002 – Eurocode: Basis of structural design.  
3) EN 1991 series – Actions on structures.  
4) National Annex to EN 1993-1-1 (where applicable).
"""
    )

# =========================================================
# APP ENTRY
# =========================================================
# --- PAGE CONFIG ---
st.set_page_config(
    page_title="EngiSnap Beam Design Eurocode Checker",
    page_icon="EngiSnap-Logo.png",
    layout="wide"
)

# --- CUSTOM GLOBAL CSS ---
custom_css = """
<style>
html, body, [class*="css"]  {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* Headings */
h1 {font-size: 1.6rem !important; font-weight: 650 !important;}
h2 {font-size: 1.25rem !important; font-weight: 600 !important;}
h3 {font-size: 1.05rem !important; font-weight: 600 !important;}

/* Main container – enough top padding so header isn't clipped */
div.block-container {
    padding-top: 1.6rem;
    max-width: 1200px;
}

/* Expander look */
.stExpander {
    border-radius: 8px !important;
    border: 1px solid #e0e0e0 !important;
}

/* Labels a bit smaller & bolder */
.stNumberInput > label {
    font-size: 0.85rem;
    font-weight: 500;
}

/* Hide Streamlit default menu & footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- SMALL SPACER SO NOTHING TOUCHES TOP EDGE ---
st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

# --- HEADER WITH LOGO + TITLE ---
header_col1, header_col2 = st.columns([1, 4])

with header_col1:
    st.image("EngiSnap-Logo.png", width=140)

with header_col2:
    st.markdown(
        """
        <div style="padding-top:10px;">
            <div style="font-size:1.5rem;font-weight:650;margin-bottom:0.1rem;">
                EngiSnap — Standard steel beam design & selection
            </div>
            <div style="color:#555;font-size:0.9rem;">
                Eurocode-based analysis and member selection for rolled steel sections
                <br/>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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

        # dynamic key so UI refreshes when section changes
        prefix_id = f"{family}_{selected_name}".replace(" ", "_")

        # --- Selected section summary ---
        render_section_summary_like_props(
            material,
            sr_display,
            key_prefix=f"sum_tab1_{prefix_id}"
        )

        # --- Cross-section preview (centered, with image if available) ---
        st.markdown("### Cross-section preview")

        img_path = get_section_image(sr_display.get("family", ""))

        # Perfect centering: middle column wider than sides
        left, center, right = st.columns([3, 4, 3])

        with center:
            if img_path:
                st.image(
                    img_path,
                    width=320,           # adjust size here
                    use_container_width=False
                )
            else:
                render_section_preview_placeholder(
                    title=f"{sr_display.get('family','')} {sr_display.get('name','')}",
                    key_prefix=f"tab1_prev_{prefix_id}"
                )


        # --- DB properties ---
        with st.expander("Section properties", expanded=False):
            render_section_properties_readonly(
                sr_display,
                key_prefix=f"tab1_db_{prefix_id}"
            )

        # --- Warnings for DB parsing ---
        if bad_fields:
            st.warning("Some DB fields could not be parsed. See debug in Results tab.")

    else:
        st.info("Select a section to continue.")

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



        except Exception as e:
            st.error(f"Computation error: {e}")

with tab4:
    render_report_tab()







