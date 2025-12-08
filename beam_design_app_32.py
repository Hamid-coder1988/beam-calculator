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
        
# =========================================================
# GLOBAL SAFETY FACTORS (EN 1993)
# =========================================================
gamma_M0 = 1.0
gamma_M1 = 1.0
# If later needed:
# GAMMA_M2 = 1.25

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
# CASE 3: SSB-C3
# Simply supported beam, 2 unequal point loads + partial UDL
# ============================

def ssb_c3_diagram(L, P1, a1, P2, a2, w, a_udl, b_udl, E=None, I=None, n=801):
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


def ssb_c3_case(L, P1, a1, P2, a2, w, a_udl, b_udl):
    """
    Case function for SSB-C1 used to prefill Loads tab.
    Returns (N, My, Mz, Vy, Vz) maxima (strong axis).
    """
    x, V, M, _ = ssb_c3_diagram(L, P1, a1, P2, a2, w, a_udl, b_udl, E=None, I=None)
    Vmax = float(np.nanmax(np.abs(V))) if V is not None else 0.0
    Mmax = float(np.nanmax(np.abs(M))) if M is not None else 0.0
    return (0.0, Mmax, 0.0, Vmax, 0.0)

def ssb_c4_diagram(L, a, b, c, w1, w2, E=None, I=None, n=400):
    """
    SSB - C4:
      Simply supported beam, two different partial UDLs:
        w1 over length a from the left support,
        w2 over length c starting at x = a + b.
      Total span = L.

    Inputs (all in m / kN/m):
        L  - span length
        a  - length of left UDL w1
        b  - distance between the two UDLs
        c  - length of right UDL w2
        w1 - left UDL intensity (kN/m)
        w2 - right UDL intensity (kN/m)

    Returns:
        x (m), V (kN), M (kN·m), delta (m or None if E/I not given)
    """
    L = float(L)
    a = float(a)
    b = float(b)
    c = float(c)
    w1 = float(w1)
    w2 = float(w2)

    # Resultant loads and centroids
    W1 = w1 * a
    x1 = a / 2.0                      # centroid of left UDL

    x0_2 = a + b                      # start of right UDL
    W2 = w2 * c
    x2 = x0_2 + c / 2.0               # centroid of right UDL

    # Reactions from global equilibrium
    R2 = (W1 * x1 + W2 * x2) / L
    R1 = W1 + W2 - R2

    # Discretisation
    x = np.linspace(0.0, L, n)

    # ------------------
    # Shear diagram V(x)
    # ------------------
    V = np.full_like(x, R1, dtype=float)

    # UDL1: from 0 to a, intensity w1
    mask1 = (x >= 0.0) & (x <= a)
    V[mask1] -= w1 * (x[mask1] - 0.0)

    mask1_right = (x > a)
    V[mask1_right] -= w1 * a

    # UDL2: from x0_2 = a + b to x0_2 + c, intensity w2
    x0_2_end = x0_2 + c
    mask2 = (x >= x0_2) & (x <= x0_2_end)
    V[mask2] -= w2 * (x[mask2] - x0_2)

    mask2_right = (x > x0_2_end)
    V[mask2_right] -= w2 * c

    # ------------------
    # Bending moment M(x) via numeric integration of V(x)
    # ------------------
    M = np.zeros_like(x)
    dx = np.diff(x)
    for i in range(len(x) - 1):
        M[i+1] = M[i] + 0.5 * (V[i] + V[i+1]) * dx[i]  # kN·m

    # ------------------
    # Deflection δ(x) via numeric double integration of M/EI
    # ------------------
    delta = None
    if E and I and I > 0 and L > 0:
        M_Nm = M * 1000.0  # kN·m → N·m
        curvature = M_Nm / (E * I)  # 1/m

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


def ssb_c4_case(L, a, b, c, w1, w2):
    """
    Case function for SSB - C4 used to prefill Loads tab.
    Returns (N, My, Mz, Vy, Vz) maxima (strong axis).
    """
    x, V, M, _ = ssb_c4_diagram(L, a, b, c, w1, w2, E=None, I=None)
    Vmax = float(np.nanmax(np.abs(V))) if V is not None else 0.0
    Mmax = float(np.nanmax(np.abs(M))) if M is not None else 0.0
    return (0.0, Mmax, 0.0, Vmax, 0.0)
# ============================
# CASE 5: SSB-C5
# Simply supported beam, full-span UDL + central point load
# Superposition of C1 (UDL) and C2 (central point load)
# ============================

# ============================
# CASE 5: SSB-C5
# Simply supported beam, UDL + mid-point load + end moments M1, M2
# ============================

def ssb_c5_diagram(L, w, P, M1, M2, E=None, I=None, n=801):
    """
    Simply supported beam with:
      - uniform load w [kN/m] over full span L
      - central point load P [kN] at x = L/2
      - end moments M1, M2 [kN·m] at x = 0 and x = L

    Returns:
      x (m), V (kN), M (kN·m), delta (m or None)
    """
    L = float(L)
    w = float(w)
    P = float(P)
    M1 = float(M1)
    M2 = float(M2)

    if L <= 0.0:
        x = np.array([0.0, 1.0])
        V = np.array([0.0, 0.0])
        M = np.array([0.0, 0.0])
        return x, V, M, None

    # ------------------
    # Reactions (superposition of UDL + point load + end moments)
    # ------------------
    # Total vertical load:
    #   W = wL + P
    # Sum of moments about left support:
    #   -M1 + R2*L - wL*(L/2) - P*(L/2) + M2 = 0
    # → R2 = wL/2 + P/2 + (M1 - M2)/L
    W = w * L + P
    R2 = w * L / 2.0 + P / 2.0 + (M1 - M2) / L
    R1 = W - R2

    # ------------------
    # Shear V(x)
    # ------------------
    x = np.linspace(0.0, L, n)
    V = np.full_like(x, R1, dtype=float)

    # UDL over full span
    V -= w * x

    # Point load P at midspan
    maskP = (x >= L / 2.0)
    V[maskP] -= P

    # ------------------
    # Bending moment M(x) by integrating V(x), starting from M1 at x=0+
    # ------------------
    M = np.zeros_like(x)
    M[0] = M1
    dx = np.diff(x)
    for i in range(len(x) - 1):
        M[i+1] = M[i] + 0.5 * (V[i] + V[i+1]) * dx[i]

    # ------------------
    # Deflection δ(x) via numeric double integration of M/EI
    # ------------------
    delta = None
    if E and I and I > 0 and L > 0:
        M_Nm = M * 1000.0  # kN·m -> N·m
        curvature = M_Nm / (E * I)  # 1/m

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


def ssb_c5_case(L, w, P, M1, M2):
    """
    Case function for SSB - C5 used to prefill Loads tab.
    Returns (N, My, Mz, Vy, Vz) based on max |M| and |V|.
    """
    x, V, M, _ = ssb_c5_diagram(L, w, P, M1, M2, E=None, I=None, n=801)
    Vmax = float(np.nanmax(np.abs(V))) if V is not None else 0.0
    Mmax = float(np.nanmax(np.abs(M))) if M is not None else 0.0
    # N, My, Mz, Vy, Vz → strong axis
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
}

# Image mapping for ready beam cases (only cases 1 and 2 for now)
CASE_IMAGE_MAP = {
    "SS-01": "beam_case_img/SSB-C1.png",
    "SS-02": "beam_case_img/SSB-C2.png",
    "SS-03": "beam_case_img/SSB-C3.png",
    "SS-04": "beam_case_img/SSB-C4.png",
    "SS-05": "beam_case_img/SSB-C5.png",
}

# Apply image paths to cases
if "Beam" in READY_CATALOG:
    for category, cases in READY_CATALOG["Beam"].items():
        for case in cases:
            key = case.get("key")
            if key in CASE_IMAGE_MAP:
                case["img_path"] = CASE_IMAGE_MAP[key]


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
READY_CATALOG["Beam"]["Simply Supported Beams (5 cases)"][2]["func"] = ssb_c3_case
READY_CATALOG["Beam"]["Simply Supported Beams (5 cases)"][2]["diagram_func"] = ssb_c3_diagram

# ---- Patch Case 4 of Simply Supported Beams: SSB-C4 (two partial UDLs) ----
READY_CATALOG["Beam"]["Simply Supported Beams (5 cases)"][3]["label"] = "SSB - C4"
READY_CATALOG["Beam"]["Simply Supported Beams (5 cases)"][3]["inputs"] = {
    "L": 6.0,   # total span (m)
    "a": 2.0,   # length of left UDL w1
    "b": 2.0,   # gap between UDLs
    "c": 2.0,   # length of right UDL w2
    "w1": 20.0, # left UDL (kN/m)
    "w2": 10.0, # right UDL (kN/m)
}
READY_CATALOG["Beam"]["Simply Supported Beams (5 cases)"][3]["func"] = ssb_c4_case
READY_CATALOG["Beam"]["Simply Supported Beams (5 cases)"][3]["diagram_func"] = ssb_c4_diagram

# ---- Patch Case 5 of Simply Supported Beams: SSB-C5 (UDL + mid-point load + end moments) ----
READY_CATALOG["Beam"]["Simply Supported Beams (5 cases)"][4]["label"] = "SSB - C5"
READY_CATALOG["Beam"]["Simply Supported Beams (5 cases)"][4]["inputs"] = {
    "L": 6.0,   # span [m]
    "w": 10.0,  # UDL [kN/m]
    "P": 20.0,  # midspan point load [kN]
    "M1": 50.0, # end moment at x=0 [kN·m]
    "M2": 20.0, # end moment at x=L [kN·m]
}
READY_CATALOG["Beam"]["Simply Supported Beams (5 cases)"][4]["func"] = ssb_c5_case
READY_CATALOG["Beam"]["Simply Supported Beams (5 cases)"][4]["diagram_func"] = ssb_c5_diagram

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
    """
    Ready design cases — BEAM ONLY gallery.
    """
    with st.expander("Ready design cases (Beam) — Gallery", expanded=True):

        # --- Step 1 ---
        chosen_type = "Beam"
        categories = list(READY_CATALOG[chosen_type].keys())
        chosen_cat = st.selectbox(
            "Step 1 — Beam category",
            categories,
            key="ready_cat_gallery",
        )

        last_cat = st.session_state.get("_ready_last_cat")
        if last_cat != chosen_cat:
            st.session_state["ready_case_key"] = None
            st.session_state["_ready_last_cat"] = chosen_cat

        # --- Step 2 ---
        st.markdown("Step 2 — Choose a case:")
        clicked_key = render_case_gallery(chosen_type, chosen_cat, n_per_row=5)

        if clicked_key:
            st.session_state["ready_case_key"] = clicked_key

        case_key = st.session_state.get("ready_case_key")

        if not case_key:
            st.info("Select a case above to see parameters and diagrams.")
            st.session_state["ready_selected_case"] = None
            st.session_state["ready_input_vals"] = None
            return

        current_cases = READY_CATALOG[chosen_type][chosen_cat]
        current_keys = {c["key"] for c in current_cases}
        if case_key not in current_keys:
            st.session_state["ready_case_key"] = None
            st.info("Selected case was from another category. Pick again.")
            return

        selected_case = next(c for c in current_cases if c["key"] == case_key)
        st.markdown(f"**Selected:** {selected_case['key']} — {selected_case['label']}")

        # --- Step 3: bending axis ---
        axis_choice = st.radio(
            "Step 3 — Bending axis for this case",
            ["Strong axis (y)", "Weak axis (z)"],
            horizontal=True,
            key=f"axis_choice_{case_key}",
        )

        # Map UI choice → simple flag for diagrams/deflection
        if axis_choice.startswith("Strong"):
            st.session_state["bending_axis"] = "y"
        else:
            st.session_state["bending_axis"] = "z"

        # --- Step 4: Case inputs ---
        input_vals = {}
        inputs_dict = selected_case.get("inputs", {})
        keys = list(inputs_dict.keys())

        for i in range(0, len(keys), 3):
            cols = st.columns(3)
            for col, k in zip(cols, keys[i : i + 3]):
                with col:
                    input_vals[k] = st.number_input(
                        k,
                        value=float(inputs_dict[k]),
                        key=f"ready_param_{case_key}_{k}",
                    )

        st.session_state["ready_selected_case"] = selected_case
        st.session_state["ready_input_vals"] = input_vals

        # --- Step 5: Apply case to Loads ---
        if st.button("Apply case to Loads", key=f"apply_case_{case_key}"):

            func = selected_case.get("func")
            if func is None:
                st.warning("Case has no calculation function.")
                return

            try:
                N, My, Mz, Vy, Vz = func(**input_vals)
            except Exception as e:
                st.error(f"Error computing case: {e}")
                return

            # Basic mapping
            st.session_state["L_in"] = float(input_vals.get("L", 6.0))
            st.session_state["N_in"] = float(N)

            # Get axis choice again (same key as above)
            axis_choice = st.session_state.get(
                f"axis_choice_{case_key}", "Strong axis (y)"
            )
            # Make sure bending_axis is in sync
            if axis_choice.startswith("Strong"):
                st.session_state["bending_axis"] = "y"
            else:
                st.session_state["bending_axis"] = "z"

            # Moments
            if axis_choice.startswith("Strong"):
                st.session_state["My_in"] = float(My)
                st.session_state["Mz_in"] = 0.0
            else:
                st.session_state["My_in"] = 0.0
                st.session_state["Mz_in"] = float(My)

            # Shear (same magnitude, swapped axis)
            V_case = Vy
            if axis_choice.startswith("Strong"):
                st.session_state["Vy_in"] = 0.0
                st.session_state["Vz_in"] = float(V_case)
            else:
                st.session_state["Vy_in"] = float(V_case)
                st.session_state["Vz_in"] = 0.0

            st.success("Ready case applied to Loads — you can edit the forces.")

        # --- Step 6: Show diagrams (always uses current bending_axis) ---
        render_beam_diagrams_panel()

# =========================================================
# UI RENDERERS
# =========================================================

def small_title(text):
    st.markdown(
        f"<div style='font-weight:600; margin-bottom:6px; font-size:0.95rem;'>{text}</div>",
        unsafe_allow_html=True
    )

def render_sidebar_guidelines():
    with st.sidebar:
        st.markdown("## Workflow")

        st.markdown(
            """
1. **Project info** – fill basic data  
2. **Loads** – choose γ_F, instability length ratios and define loading  
3. **Section** – pick material and DB section  
4. **Results** – press **Run check** and review utilisations  
5. **Report** – review full report and export PDF
            """
        )

        st.markdown("---")
        st.markdown("## Current setup")

        # --- Material & section ---
        material = st.session_state.get("material", "–")
        sr = st.session_state.get("sr_display")
        if isinstance(sr, dict):
            sec_txt = f"{sr.get('family', '–')} {sr.get('name', '')}".strip()
        else:
            sec_txt = "No section selected"

        st.write(f"**Material:** {material}")
        st.write(f"**Section:** {sec_txt}")

        # --- Loads / ready case ---
        load_mode = st.session_state.get("load_mode_choice", "Use ready beam case")
        gamma_F = float(st.session_state.get("gamma_F", 1.50))
        manual_type = st.session_state.get("manual_forces_type", "Characteristic")

        st.write(f"**Load mode:** {load_mode}")
        if load_mode.startswith("Use ready"):
            rc = st.session_state.get("ready_selected_case")
            if rc:
                rc_txt = f"{rc.get('key','')} — {rc.get('label','')}"
            else:
                rc_txt = "No ready case selected"
            st.write(f"**Ready case:** {rc_txt}")

        st.write(f"**γ_F:** {gamma_F:.2f}")
        st.write(f"**Forces as:** {manual_type}")

        # --- Quick check status ---
        st.markdown("---")
        run_done = st.session_state.get("run_clicked", False)
        overall_ok = st.session_state.get("overall_ok", None)

        if run_done and overall_ok is not None:
            emoji = "✅" if overall_ok else "⚠️"
            status_txt = "OK" if overall_ok else "NOT OK"
            st.markdown(f"**Design status:** {emoji} {status_txt}")
        else:
            st.markdown("**Design status:** ⏳ Not run yet")

        # --- Quick tips ---
        st.markdown("---")
        st.markdown("## Quick tips")
        st.markdown(
            """
- Positive **N** = compression  
- **M_y** = bending about strong axis (y)  
- **M_z** = bending about weak axis (z)  
- DB section properties are **read-only**  
- Use *ready beam cases* to auto-fill loads & diagrams
            """
        )

def render_project_data():
    """Project data at top of Tab 1 – no expander."""
    st.markdown("### Project data")

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

    # Notes / comments block (same as before, just not inside expander)
    notes = st.text_area(
        "Notes / comments",
        value="",
        key="notes_in"
    )

    st.markdown("---")

    # Return same 7-tuple as before (report logic still works)
    return doc_name, project_name, position, requested_by, revision, run_date, notes

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
    # Row 1
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.number_input("h (mm)", value=float(sr_display.get("h_mm", 0.0)), disabled=True, key=f"{key_prefix}_h")
    c2.number_input("b (mm)", value=float(sr_display.get("b_mm", 0.0)), disabled=True, key=f"{key_prefix}_b")
    c3.number_input("tw (mm)", value=float(sr_display.get("tw_mm", 0.0)), disabled=True, key=f"{key_prefix}_tw")
    c4.number_input("tf (mm)", value=float(sr_display.get("tf_mm", 0.0)), disabled=True, key=f"{key_prefix}_tf")
    c5.number_input("m (kg/m)", value=float(sr_display.get("m_kg_per_m", 0.0)), disabled=True, key=f"{key_prefix}_m")
    c6.number_input("A (mm²)", value=float(sr_display.get("A_mm2", 0.0)), disabled=True, key=f"{key_prefix}_A")

    # Row 2
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.number_input("Av,z (mm²)", value=float(sr_display.get("Avz_mm2", 0.0)), disabled=True, key=f"{key_prefix}_Avz")
    c2.number_input("Av,y (mm²)", value=float(sr_display.get("Avy_mm2", 0.0)), disabled=True, key=f"{key_prefix}_Avy")
    c3.number_input("Iy (cm⁴)", value=float(sr_display.get("Iy_cm4", 0.0)), disabled=True, key=f"{key_prefix}_Iy")
    c4.number_input("iy (mm)", value=float(sr_display.get("iy_mm", 0.0)), disabled=True, key=f"{key_prefix}_iy")
    c5.number_input("Wel,y (cm³)", value=float(sr_display.get("Wel_y_cm3", 0.0)), disabled=True, key=f"{key_prefix}_Wel_y")
    c6.number_input("Wpl,y (cm³)", value=float(sr_display.get("Wpl_y_cm3", 0.0)), disabled=True, key=f"{key_prefix}_Wpl_y")

    # Row 3
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.number_input("Iz (cm⁴)", value=float(sr_display.get("Iz_cm4", 0.0)), disabled=True, key=f"{key_prefix}_Iz")
    c2.number_input("iz (mm)", value=float(sr_display.get("iz_mm", 0.0)), disabled=True, key=f"{key_prefix}_iz")
    c3.number_input("Wel,z (cm³)", value=float(sr_display.get("Wel_z_cm3", 0.0)), disabled=True, key=f"{key_prefix}_Wel_z")
    c4.number_input("Wpl,z (cm³)", value=float(sr_display.get("Wpl_z_cm3", 0.0)), disabled=True, key=f"{key_prefix}_Wpl_z")
    c5.number_input("IT (×10³ mm⁴)", value=float(sr_display.get("IT_k_mm4", 0.0)), disabled=True, key=f"{key_prefix}_IT")
    c6.number_input("WT (×10³ mm³)", value=float(sr_display.get("WT_k_mm3", 0.0)), disabled=True, key=f"{key_prefix}_WT")

    # Row 4
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.number_input("Iw (×10⁶ mm⁶)", value=float(sr_display.get("Iw_6_mm6", 0.0)), disabled=True, key=f"{key_prefix}_Iw")
    c2.number_input("Ww (×10³ mm⁴)", value=float(sr_display.get("Ww_k_mm4", 0.0)), disabled=True, key=f"{key_prefix}_Ww")
    c3.number_input("Npl,Rd (kN)", value=float(sr_display.get("Npl_Rd_kN", 0.0)), disabled=True, key=f"{key_prefix}_Npl")
    c4.number_input("Vpl,Rd,z (kN)", value=float(sr_display.get("Vpl_Rd_z_kN", 0.0)), disabled=True, key=f"{key_prefix}_Vplz")
    c5.number_input("Vpl,Rd,y (kN)", value=float(sr_display.get("Vpl_Rd_y_kN", 0.0)), disabled=True, key=f"{key_prefix}_Vply")
    c6.number_input("Mel,Rd,y (kNm)", value=float(sr_display.get("Mel_Rd_y_kNm", 0.0)), disabled=True, key=f"{key_prefix}_Mely")

    # Row 5
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.number_input("Mpl,Rd,y (kNm)", value=float(sr_display.get("Mpl_Rd_y_kNm", 0.0)), disabled=True, key=f"{key_prefix}_Mply")
    c2.number_input("Mel,Rd,z (kNm)", value=float(sr_display.get("Mel_Rd_z_kNm", 0.0)), disabled=True, key=f"{key_prefix}_Melz")
    c3.number_input("Mpl,Rd,z (kNm)", value=float(sr_display.get("Mpl_Rd_z_kNm", 0.0)), disabled=True, key=f"{key_prefix}_Mplz")
    c4.number_input("Buckling α_y", value=float(sr_display.get("alpha_y", 0.0)), disabled=True, key=f"{key_prefix}_alpha_y")
    c5.number_input("Buckling α_z", value=float(sr_display.get("alpha_z", 0.0)), disabled=True, key=f"{key_prefix}_alpha_z")
    c6.text_input("CL Wb", value=str(sr_display.get("CL_Wb", "n/a")), disabled=True, key=f"{key_prefix}_CLWb")

    # Row 6
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.text_input("CL Wc", value=str(sr_display.get("CL_Wc", "n/a")), disabled=True, key=f"{key_prefix}_CLWc")
    c2.text_input("CL Fb", value=str(sr_display.get("CL_Fb", "n/a")), disabled=True, key=f"{key_prefix}_CLFb")
    c3.empty()
    c4.empty()
    c5.empty()
    c6.empty()


def render_loads_form(family_for_torsion: str, read_only: bool = False):
    prefill = st.session_state.get("prefill_from_case", False)
    defval = lambda key, fallback: float(st.session_state.get(key, fallback)) if prefill else fallback
    torsion_supported = supports_torsion_and_warping(family_for_torsion)

    with st.container():
        if read_only:
            st.subheader("Design forces and moments (ULS) — from ready case")
            st.caption(
                "Values come from the selected ready beam case. "
                "You can still adjust K-factors (buckling) below."
            )
        else:
            st.subheader("Design forces and moments (ULS) — INPUT")
            st.caption("Positive N = compression. Enter characteristic or design forces based on Tab 2 settings.")
            r1c1, r1c2, r1c3 = st.columns(3)
            
            with r1c1:
                L = st.number_input(
                    "Element length L (m)",
                    min_value=0.0,
                    key="L_in",
                    disabled=read_only,
                )
            
            with r1c2:
                N_kN = st.number_input(
                    "Axial force N (kN)",
                    key="N_in",
                    disabled=read_only,
                )
            
            with r1c3:
                Vy_kN = st.number_input(
                    "Shear V_y (kN)",
                    key="Vy_in",
                    disabled=read_only,
                )
            
            r2c1, r2c2, r2c3 = st.columns(3)
            
            with r2c1:
                Vz_kN = st.number_input(
                    "Shear V_z (kN)",
                    key="Vz_in",
                    disabled=read_only,
                )
            
            with r2c2:
                My_kNm = st.number_input(
                    "Bending M_y (kN·m) about y",
                    key="My_in",
                    disabled=read_only,
                )
            
            with r2c3:
                Mz_kNm = st.number_input(
                    "Bending M_z (kN·m) about z",
                    key="Mz_in",
                    disabled=read_only,
                )

            

def store_design_forces_from_state():
    """Compute design ULS forces from current Loads inputs in session_state
    and store them into st.session_state['inputs']. This replaces the old
    Run button inside the Loads tab; it is now triggered from the Results tab.
    """
    # Raw inputs from Loads form
    L = float(st.session_state.get("L_in", 0.0))
    N_kN = float(st.session_state.get("N_in", 0.0))
    Vy_kN = float(st.session_state.get("Vy_in", 0.0))
    Vz_kN = float(st.session_state.get("Vz_in", 0.0))
    My_kNm = float(st.session_state.get("My_in", 0.0))
    Mz_kNm = float(st.session_state.get("Mz_in", 0.0))
    Tx_kNm = float(st.session_state.get("Tx_in", 0.0)) if "Tx_in" in st.session_state else 0.0

    K_y  = float(st.session_state.get("Ky_in", 1.0))
    K_z  = float(st.session_state.get("Kz_in", 1.0))
    K_LT = float(st.session_state.get("KLT_in", 1.0))
    K_T  = float(st.session_state.get("KT_in", 1.0))

    # Design settings from Tab 2
    gamma_F = st.session_state.get("gamma_F", 1.50)
    manual_forces_type = st.session_state.get("manual_forces_type", "Characteristic")

    if str(manual_forces_type).startswith("Characteristic"):
        factor = gamma_F
    else:
        factor = 1.0

    # Apply factor to get DESIGN forces (N_Ed, V_Ed, M_Ed)
    N_design_kN   = N_kN   * factor
    Vy_design_kN  = Vy_kN  * factor
    Vz_design_kN  = Vz_kN  * factor
    My_design_kNm = My_kNm * factor
    Mz_design_kNm = Mz_kNm * factor
    Tx_design_kNm = Tx_kNm * factor

    st.session_state["run_clicked"] = True
    st.session_state["inputs"] = dict(
        L=L,
        N_kN=N_design_kN,
        Vy_kN=Vy_design_kN,
        Vz_kN=Vz_design_kN,
        My_kNm=My_design_kNm,
        Mz_kNm=Mz_design_kNm,
        Tx_kNm=Tx_design_kNm,
        K_y=K_y,
        K_z=K_z,
        K_LT=K_LT,
        K_T=K_T,
    )

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
    
    # -----------------------------------------------
    # Cross-section checks: axial tension / compression
    # Sign convention: N > 0 tension, N < 0 compression
    # -----------------------------------------------
    rows = []  # if you already have rows defined earlier, remove this line

    # Design axial force in N (N_N already defined)
    # Tension: N > 0
    N_ten_N = max(N_N, 0.0)
    # Compression: N < 0 (store magnitude as positive)
    N_comp_N = max(-N_N, 0.0)

    # Plastic axial resistance of gross section
    # N_Rd_N was defined above:
    # N_Rd_N = A_m2 * fy * 1e6 / gamma_M0
    T_Rd_N = N_Rd_N       # tension resistance
    Nc_Rd_N = N_Rd_N      # compression resistance (cross-section only)

    # --- (1) Tension (N > 0) ---
    if T_Rd_N > 0.0 and N_ten_N > 0.0:
        util_ten = N_ten_N / T_Rd_N
        status_ten = "OK" if util_ten <= 1.0 else "EXCEEDS"
    else:
        util_ten = 0.0
        status_ten = "OK"  # no tension → check is trivially OK

    rows.append({
        "Check":      "Tension (N>0)",
        "Applied":    f"{N_ten_N/1e3:.3f} kN",
        "Resistance": f"{T_Rd_N/1e3:.3f} kN",
        "Utilization": f"{util_ten:.3f}",
        "Status":      status_ten,
    })

    # --- (2) Compression (N < 0) – cross-section resistance only ---
    if Nc_Rd_N > 0.0 and N_comp_N > 0.0:
        util_comp = N_comp_N / Nc_Rd_N
        status_comp = "OK" if util_comp <= 1.0 else "EXCEEDS"
    else:
        util_comp = 0.0
        status_comp = "OK"

    rows.append({
        "Check":      "Compression (N<0)",
        "Applied":    f"{-N_comp_N/1e3:.3f} kN",  # negative sign shown in kN
        "Resistance": f"{Nc_Rd_N/1e3:.3f} kN",
        "Utilization": f"{util_comp:.3f}",
        "Status":      status_comp,
    })

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
    
    applied_N = -N_N if N_N < 0 else 0.0
    status_comp, util_comp = status_and_util(applied_N, compression_resistance_N)
    rows.append({"Check":"Compression (N<0)","Applied":f"{applied_N/1e3:.3f} kN",
                 "Resistance":f"{compression_resistance_N/1e3:.3f} kN",
                 "Utilization": f"{util_comp:.3f}" if util_comp is not None else "n/a",
                 "Status":status_comp})

    applied_tension_N = N_N if N_N >= 0 else 0.0
    status_ten, util_ten = status_and_util(applied_tension_N, T_Rd_N)
    rows.append({"Check":"Tension (N≥0)","Applied":f"{applied_tension_N/1e3:.3f} kN",
                 "Resistance":f"{T_Rd_N/1e3:.3f} kN",
                 "Utilization": f"{util_ten:.3f}" if util_ten is not None else "n/a",
                 "Status":status_ten})
    
    applied_shear_N = math.sqrt(Vy_N**2 + Vz_N**2)
    status_shear, util_shear = status_and_util(applied_shear_N, V_Rd_N)
    rows.append({"Check":"Shear (resultant Vy & Vz)","Applied":f"{applied_shear_N/1e3:.3f} kN",
                 "Resistance":f"{V_Rd_N/1e3:.3f} kN",
                 "Utilization": f"{util_shear:.3f}" if util_shear is not None else "n/a",
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
    """
    Results tab: two aligned tables with color coding and bold governing row,
    nicer card-style formatting.
    """

    gov_check, gov_util = governing
    status_txt = "OK" if overall_ok else "NOT OK"

    # -------------------------------------------------
    # Top summary
    # -------------------------------------------------
    small_title("Result summary")
    if gov_util is not None:
        st.caption(
            f"Overall status: **{status_txt}**  |  "
            f"Governing check: **{gov_check or 'n/a'}**  |  "
            f"Max utilisation: **{gov_util:.3f}**"
        )
    else:
        st.caption(f"Overall status: **{status_txt}**")

    # -------------------------------------------------
    # Deflection summary (from diagrams)
    # -------------------------------------------------
    diag_summary = st.session_state.get("diag_summary")

    if diag_summary and diag_summary.get("defl_available"):
        w_max_mm   = diag_summary.get("w_max_mm")
        limit_L300 = diag_summary.get("limit_L300")
        limit_L600 = diag_summary.get("limit_L600")
        limit_L900 = diag_summary.get("limit_L900")

        # format as strings
        val_delta = f"{w_max_mm:.3f}" if w_max_mm is not None else "n/a"
        val_L300  = f"{limit_L300:.3f}" if limit_L300 is not None else "n/a"
        val_L600  = f"{limit_L600:.3f}" if limit_L600 is not None else "n/a"
        val_L900  = f"{limit_L900:.3f}" if limit_L900 is not None else "n/a"

        # *** IMPORTANT ***
        # overwrite widget state so text_inputs show the latest values
        st.session_state["res_delta_max_mm"] = val_delta
        st.session_state["res_L300_mm"]      = val_L300
        st.session_state["res_L600_mm"]      = val_L600
        st.session_state["res_L900_mm"]      = val_L900

        small_title("Deflection (serviceability)")

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.text_input(
                "δ_max [mm]",
                value=st.session_state["res_delta_max_mm"],
                disabled=True,
                key="res_delta_max_mm",
            )
        with d2:
            st.text_input(
                "Limit L/300 [mm]",
                value=st.session_state["res_L300_mm"],
                disabled=True,
                key="res_L300_mm",
            )
        with d3:
            st.text_input(
                "Limit L/600 [mm]",
                value=st.session_state["res_L600_mm"],
                disabled=True,
                key="res_L600_mm",
            )
        with d4:
            st.text_input(
                "Limit L/900 [mm]",
                value=st.session_state["res_L900_mm"],
                disabled=True,
                key="res_L900_mm",
            )
    else:
        st.caption(
            "Deflection summary (δ_max, L/300, L/600, L/900) will appear here "
            "after running the diagrams in the Loads tab."
        )

    # -------------------------------------------------
    # -----------------------------------
    # Data for the two tables
    # -------------------------------------------------
    cs_checks = [
        "N (tension)",                  # 1
        "N (compression)",              # 2
        "My",                           # 3
        "Mz",                           # 4
        "Vy",                           # 5
        "Vz",                           # 6
        "My + Vy",                      # 7
        "Mz + Vz",                      # 8
        "My + N",                       # 9
        "Mz + N",                       # 10
        "My + Mz + N",                  # 11
        "My + N + V",                   # 12
        "Mz + N + V",                   # 13
        "My + Mz + N + V",              # 14
    ]
    buck_checks = [
        "Flexural buckling y–y",                         # 15
        "Flexural buckling z–z",                         # 16
        "Torsional / torsional-flexural buckling z",     # 17
        "Lateral-torsional buckling",                    # 18
        "Bending + axial compression (Method 1)",        # 19
        "Bending + axial compression (Method 2)",        # 20
    ]

    # For now: placeholders (you'll later replace with real values)
    cs_util = ["" for _ in cs_checks]
    cs_status = ["" for _ in cs_checks]
    buck_util = ["" for _ in buck_checks]
    buck_status = ["" for _ in buck_checks]
    # Map detailed check results into the summary table
    # 1) Tension row: "N (tension)" ← "Tension (N≥0)" from df_rows
    if df_rows is not None and "Tension (N≥0)" in df_rows.index:
        ten_row = df_rows.loc["Tension (N≥0)"]
        cs_util[0] = ten_row.get("Utilization", "")
        cs_status[0] = ten_row.get("Status", "")
        
    # 2) Compression row: "N (compression)" ← "Compression (N<0)" from df_rows
    if df_rows is not None and "Compression (N<0)" in df_rows.index:
        comp_row = df_rows.loc["Compression (N<0)"]
        cs_util[1] = comp_row.get("Utilization", "")
        cs_status[1] = comp_row.get("Status", "")

    # -------------------------------------------------
    # Helper to build one nice looking table
    # -------------------------------------------------
    def build_table_html(title, start_no, names, utils, statuses):
        card_open = """
<div style="
    border:1px solid #d7d7d7;
    border-radius:8px;
    padding:8px 10px 10px 10px;
    background-color:#fafafa;
    box-shadow:0 1px 2px rgba(0,0,0,0.05);
    margin-bottom:10px;">
"""
        card_close = "</div>"

        html = card_open
        html += f"<div style='font-weight:600; margin-bottom:6px;'>{title}</div>"

        html += """
<table style="width:100%; border-collapse:collapse; font-size:0.9rem;">
  <tr style="background-color:#f2f2f2;">
    <th style="width:8%;  border-bottom:1px solid #ccc; padding:6px; text-align:center;">#</th>
    <th style="width:54%; border-bottom:1px solid #ccc; padding:6px; text-align:left;">Check</th>
    <th style="width:19%; border-bottom:1px solid #ccc; padding:6px; text-align:center;">Utilization</th>
    <th style="width:19%; border-bottom:1px solid #ccc; padding:6px; text-align:center;">Status</th>
  </tr>
"""

        for i, (name, util, status) in enumerate(zip(names, utils, statuses)):
            number = start_no + i

            # background color based on status
            status_upper = (status or "").strip().upper()
            if status_upper == "OK":
                bg = "#e6f7e6"
            elif status_upper == "EXCEEDS":
                bg = "#fde6e6"
            else:
                # subtle striping for neutral rows
                bg = "#ffffff" if (i % 2 == 0) else "#f9f9f9"

            # bold if governing
            is_gov = False
            if gov_check is not None:
                if str(gov_check).strip() == str(number):
                    is_gov = True
                elif str(gov_check).strip().lower() == str(name).strip().lower():
                    is_gov = True
            fw = "700" if is_gov else "400"

            html += f"""
  <tr style="background-color:{bg};">
    <td style="border-top:1px solid #e0e0e0; padding:5px; text-align:center; font-weight:{fw};">{number}</td>
    <td style="border-top:1px solid #e0e0e0; padding:5px; text-align:left;   font-weight:{fw};">{name}</td>
    <td style="border-top:1px solid #e0e0e0; padding:5px; text-align:center; font-weight:{fw};">{util}</td>
    <td style="border-top:1px solid #e0e0e0; padding:5px; text-align:center; font-weight:{fw};">{status}</td>
  </tr>
"""
        html += "</table>" + card_close
        return html

    # -------------------------------------------------
    # TABLE 1: Cross-section strength
    # -------------------------------------------------
    cs_html = build_table_html(
        "Verification of cross-section strength (ULS, checks 1–14)",
        1,
        cs_checks,
        cs_util,
        cs_status,
    )
    st.markdown(cs_html, unsafe_allow_html=True)

    st.markdown("---")

    # -------------------------------------------------
    # TABLE 2: Buckling
    # -------------------------------------------------
    buck_html = build_table_html(
        "Verification of member stability (buckling, checks 15–20)",
        15,
        buck_checks,
        buck_util,
        buck_status,
    )
    st.markdown(buck_html, unsafe_allow_html=True)

    st.markdown("---")

    # -------------------------------------------------
    # Bottom hints
    # -------------------------------------------------
    st.caption("See **Report** tab for full formulas & Eurocode clause references.")
    st.caption("See **Diagrams** / ready cases tab for shear, moment and deflection graphs.")

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
    Use current ready-case inputs and the bending-axis radio
    to pick Iy or Iz for deflection. Store summary in
    st.session_state["diag_summary"].
    """
    selected_case = st.session_state.get("ready_selected_case")
    input_vals    = st.session_state.get("ready_input_vals")
    sr_display    = st.session_state.get("sr_display")

    if not selected_case or not input_vals:
        return

    diag_func = selected_case.get("diagram_func")
    if not diag_func:
        st.info("No diagrams yet for this case.")
        return

    # --------------------------------------
    # 1) Determine bending axis from radio
    # --------------------------------------
    case_key = st.session_state.get("ready_case_key")
    axis_choice = st.session_state.get(
        f"axis_choice_{case_key}", "Strong axis (y)"
    )

    if axis_choice.startswith("Weak"):
        bending_axis = "z"
    else:
        bending_axis = "y"

    # also store it for other places if needed
    st.session_state["bending_axis"] = bending_axis

    # --------------------------------------
    # 2) Section stiffness for deflection
    # --------------------------------------
    E = 210e9  # Pa

    I_y_m4 = float(sr_display.get("I_y_cm4", 0.0)) * 1e-8 if sr_display else 0.0
    I_z_m4 = float(sr_display.get("I_z_cm4", 0.0)) * 1e-8 if sr_display else 0.0

    if bending_axis == "z":
        I_m4 = I_z_m4
    else:
        I_m4 = I_y_m4

    if I_m4 <= 0:
        I_m4 = None  # allow V/M but disable deflection if no inertia

    # --------------------------------------
    # 3) Call diagram function
    # --------------------------------------
    args = [input_vals[k] for k in selected_case["inputs"].keys()]
    x, V, M, delta = diag_func(*args, E=E, I=I_m4)

    # span length
    L_val = float(input_vals.get("L", 0.0))
    if (not L_val or L_val <= 0.0) and x is not None and len(x) > 1:
        L_val = float(x[-1] - x[0])

    # --------------------------------------
    # 4) Summary for deflection & forces
    # --------------------------------------
    summary = get_beam_summary_for_diagrams(x, V, M, delta, L_val)
    summary["bending_axis"] = bending_axis
    st.session_state["diag_summary"] = summary
    # 4b) Export main numbers for Report tab & Loads 3.2
    try:
        # Vmax from the shear diagram
        if V is not None:
            st.session_state["diag_Vmax_kN"] = float(np.nanmax(np.abs(V)))
        else:
            st.session_state["diag_Vmax_kN"] = None

        # From summary dict
        M_max      = summary.get("M_max")
        x_M_max    = summary.get("x_M_max")
        V_at_Mmax  = summary.get("V_at_Mmax")
        R_left     = summary.get("R_left")
        R_right    = summary.get("R_right")
        w_max_mm   = summary.get("w_max_mm")

        st.session_state["diag_Mmax_kNm"]      = float(M_max)     if M_max     is not None else None
        st.session_state["diag_x_Mmax_m"]      = float(x_M_max)   if x_M_max   is not None else None
        st.session_state["diag_V_at_Mmax_kN"]  = float(V_at_Mmax) if V_at_Mmax is not None else None
        st.session_state["diag_R1_kN"]         = float(R_left)    if R_left    is not None else None
        st.session_state["diag_R2_kN"]         = float(R_right)   if R_right   is not None else None
        st.session_state["diag_delta_max_mm"]  = float(w_max_mm)  if w_max_mm  is not None else None
    except Exception:
        # Don't break diagrams if something is missing
        pass
    # --------------------------------------
    # 5) Diagrams (V & M, same style as before)
    # --------------------------------------
    colV, colM = st.columns(2)

    with colV:
        small_title("Shear force diagram V(x)")
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
        small_title("Bending moment diagram M(x)")
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

    # Deflection diagram δ(x) disabled; only δ_max is used in Results/Report.

# =========================================================
# REPORT TAB & PDF HELPERS — ENGISNAP FULL REPORT
# =========================================================

def render_loads_readonly_report(inputs: dict, torsion_supported: bool, key_prefix="rpt_load"):
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
        # support both old (6 items) and new (7 items with notes)
        if len(meta) == 7:
            doc_title, project_name, position, requested_by, revision, run_date, notes = meta
        else:
            doc_title, project_name, position, requested_by, revision, run_date = meta
            notes = "–"
    else:
        doc_title = "Beam check"
        project_name = position = requested_by = revision = ""
        run_date = date.today()
        notes = "–"

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
    story.append(Paragraph(f"Notes / comments: {notes}", N))
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
    ENGISNAP Report tab — structured as:

    1. Project information
    2. Material design values
    3. Member & section data
    4. Section classification
    5. Applied actions & internal forces (ULS)
    6. Verification of cross-section strength (ULS)
    7. Member stability (buckling)
    8. References
    """
    sr_display = st.session_state.get("sr_display")
    inputs = st.session_state.get("inputs")
    df_rows = st.session_state.get("df_rows")
    overall_ok = st.session_state.get("overall_ok", True)
    governing = st.session_state.get("governing", (None, None))
    extras = st.session_state.get("extras") or {}
    meta = st.session_state.get("meta")
    material = st.session_state.get("material", "S355")

    # from extras (computed in compute_checks)
    sigma_allow = extras.get("sigma_allow_MPa")
    sigma_eq = extras.get("sigma_eq_MPa")
    buck_results = extras.get("buck_results", [])

    if sr_display is None or inputs is None or df_rows is None or meta is None:
        st.info("To see the report: select a section, define loads, run the check, then return here.")
        return

    # --- support old 6-item meta and new 7-item meta with notes ---
    if len(meta) == 7:
        doc_title, project_name, position, requested_by, revision, run_date, notes = meta
    else:
        doc_title, project_name, position, requested_by, revision, run_date = meta
        notes = ""

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

    # ----------------------------------------------------
    # PDF download (top)
    # ----------------------------------------------------
    st.markdown("### Save report")
    if HAS_RL:
        pdf_buf = build_pdf_report(meta, material, sr_display, inputs, df_rows, overall_ok, governing, extras)
        if pdf_buf:
            st.download_button(
                "💾 Save as PDF",
                data=pdf_buf,
                file_name="EngiSnap_Beam_Report.pdf",
                mime="application/pdf",
                key="rpt_pdf_btn_top",
            )
    else:
        st.warning("PDF export not available (reportlab not installed).")

    st.markdown("---")

    # ----------------------------------------------------
    # 1. Project information
    # ----------------------------------------------------
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

    st.text_area(
        "Notes / comments",
        value=str(notes or ""),
        disabled=True,
        key="rpt_notes",
    )
    st.markdown("---")

    # ----------------------------------------------------
    # 2. Material design values (EN 1993-1-1)
    # ----------------------------------------------------
    st.markdown("## 2. Material design values (EN 1993-1-1)")

    eps = (235.0 / fy) ** 0.5 if fy > 0 else None
    fy_over_gM0 = fy / 1.0 if fy is not None else None
    fy_over_gM1 = fy / 1.0 if fy is not None else None

    m1, m2, m3 = st.columns(3)
    with m1:
        st.text_input("Steel grade", value=material, disabled=True, key="rpt_mat_grade")
        st.text_input("Yield strength f_y [MPa]", value=f"{fy:.1f}", disabled=True, key="rpt_mat_fy")
    with m2:
        st.text_input("Ultimate strength f_u [MPa]", value="(not specified)", disabled=True, key="rpt_mat_fu")
        st.text_input("Elastic modulus E [MPa]", value="210000", disabled=True, key="rpt_mat_E")
    with m3:
        st.text_input("γ_M0", value="1.00", disabled=True, key="rpt_mat_gM0")
        st.text_input("γ_M1", value="1.00", disabled=True, key="rpt_mat_gM1")

    m4, m5, m6 = st.columns(3)
    with m4:
        st.text_input(
            "ε = √(235 / f_y)",
            value=f"{eps:.3f}" if eps is not None else "n/a",
            disabled=True,
            key="rpt_mat_eps",
        )
    with m5:
        st.text_input(
            "f_y / γ_M0 [MPa]",
            value=f"{fy_over_gM0:.1f}" if fy_over_gM0 is not None else "n/a",
            disabled=True,
            key="rpt_mat_fy_gM0",
        )
    with m6:
        st.text_input(
            "f_y / γ_M1 [MPa]",
            value=f"{fy_over_gM1:.1f}" if fy_over_gM1 is not None else "n/a",
            disabled=True,
            key="rpt_mat_fy_gM1",
        )

    m7, m8, m9 = st.columns(3)
    with m7:
        st.text_input("Shear modulus G [MPa]", value="81000", disabled=True, key="rpt_mat_G")
    with m8:
        st.text_input(
            "Indicative σ_allow [MPa]",
            value=f"{sigma_allow:.3f}" if sigma_allow is not None else "n/a",
            disabled=True,
            key="rpt_mat_sigma_allow",
        )
    with m9:
        st.text_input(
            "Equivalent σ_eq [MPa]",
            value=f"{sigma_eq:.3f}" if sigma_eq is not None else "n/a",
            disabled=True,
            key="rpt_mat_sigma_eq",
        )

    st.markdown("---")

    # ----------------------------------------------------
    # 3. Member & section data
    # ----------------------------------------------------
    st.markdown("## 3. Member & section data")

    # 3.1 Member inputs
    st.markdown("### 3.1 Member definition")

    ready_case = st.session_state.get("ready_selected_case")
    member_type = "Standard member"
    support_txt = "User-defined"
    if ready_case:
        member_type = f"Beam case: {ready_case.get('key','')} — {ready_case.get('label','')}"
        support_txt = "From ready case"

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.text_input("Member type", value=member_type, disabled=True, key="rpt_mem_type")
        st.number_input("Span length L [m]", value=float(L), disabled=True, key="rpt_L")
    with mc2:
        st.text_input("Support conditions", value=support_txt, disabled=True, key="rpt_support")
        st.number_input("Effective length L_y [m]", value=float(Leff_y), disabled=True, key="rpt_Leff_y")
    with mc3:
        st.number_input("Effective length L_z [m]", value=float(Leff_z), disabled=True, key="rpt_Leff_z")
        st.number_input("LT buckling length L_LT [m]", value=float(Leff_LT), disabled=True, key="rpt_Leff_LT")

    # 3.2 Section data
    st.markdown("### 3.2 Section data")

    cs1, cs2 = st.columns(2)
    with cs1:
        st.text_input("Section family", value=fam, disabled=True, key="rpt_cs_family")
        st.text_input("Section size", value=name, disabled=True, key="rpt_cs_name")

        # Selected section summary — already nice (6 boxes per 2 rows)
        render_section_summary_like_props(material, sr_display, key_prefix="rpt_sum")

    with cs2:
        img_path = get_section_image(fam)
        if img_path:
            st.markdown("Cross-section schematic")
            st.image(img_path, width=260)
        else:
            st.markdown("Cross-section schematic")
            st.info("No image available for this family.")

    # full DB section properties
    with st.expander("3.3 Section properties from DB", expanded=False):
        render_section_properties_readonly(sr_display, key_prefix="rpt_props")

    st.markdown("---")

    # ----------------------------------------------------
    # 4. Section classification (EN 1993-1-1 §5.5)
    # ----------------------------------------------------
    st.markdown("## 4. Section classification (EN 1993-1-1 §5.5)")

    flange_class = sr_display.get("flange_class_db", "n/a")
    web_class_b = sr_display.get("web_class_bending_db", "n/a")
    web_class_c = sr_display.get("web_class_compression_db", "n/a")

    # Simple 4-row "table" using columns
    cfl, cwe, cwc = st.columns(3)
    with cfl:
        st.text_input("Flange – bending", value=str(flange_class), disabled=True, key="rpt_flange_class")
    with cwe:
        st.text_input("Web – bending", value=str(web_class_b), disabled=True, key="rpt_web_class_b")
    with cwc:
        st.text_input("Web – uniform compression", value=str(web_class_c), disabled=True, key="rpt_web_class_c")

    st.text_input(
        "Governing cross-section class",
        value="(not explicitly derived yet)",
        disabled=True,
        key="rpt_cs_class",
    )

    st.markdown("---")

    # ----------------------------------------------------
    # 5. Applied actions & internal forces (ULS)
    # ----------------------------------------------------
    st.markdown("## 5. Applied actions & internal forces (ULS)")

    # 5.1 Load input (as entered)
    st.markdown("### 5.1 Load input (ULS)")

    if ready_case:
        st.write(f"Load case type: **{ready_case.get('key','')} — {ready_case.get('label','')}**")
    else:
        st.write("Load case type: **User-defined**")

    st.write(
        f"N = {inputs.get('N_kN', 0.0):.3f} kN, "
        f"Vy = {inputs.get('Vy_kN', 0.0):.3f} kN, "
        f"Vz = {inputs.get('Vz_kN', 0.0):.3f} kN, "
        f"My = {inputs.get('My_kNm', 0.0):.3f} kNm, "
        f"Mz = {inputs.get('Mz_kNm', 0.0):.3f} kNm"
    )

    # 5.2 Internal forces results
    st.markdown("### 5.2 Internal forces & deflection summary")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.text_input(
            "V_max [kN]",
            value=f"{Vmax:.3f}" if Vmax is not None else "n/a",
            disabled=True,
            key="rpt_Vmax",
        )
    with c2:
        st.text_input(
            "M_max [kN·m]",
            value=f"{Mmax:.3f}" if Mmax is not None else "n/a",
            disabled=True,
            key="rpt_Mmax",
        )
    with c3:
        st.text_input(
            "R_A support [kN]",
            value=f"{R1:.3f}" if R1 is not None else "n/a",
            disabled=True,
            key="rpt_R1",
        )
    with c4:
        st.text_input(
            "R_B support [kN]",
            value=f"{R2:.3f}" if R2 is not None else "n/a",
            disabled=True,
            key="rpt_R2",
        )

    d1, d2, d3 = st.columns(3)
    if L > 0 and delta_max_mm not in (None, 0.0):
        L_over_w = (L * 1000.0) / abs(delta_max_mm) if delta_max_mm != 0 else None
    else:
        L_over_w = None

    with d1:
        st.text_input(
            "δ_max [mm]",
            value=f"{delta_max_mm:.3f}" if delta_max_mm is not None else "n/a",
            disabled=True,
            key="rpt_delta_max",
        )
    with d2:
        st.text_input(
            "Span L [mm]",
            value=f"{L*1000.0:.1f}" if L > 0 else "n/a",
            disabled=True,
            key="rpt_Lmm",
        )
    with d3:
        st.text_input(
            "Deflection ratio L / δ_max",
            value=f"{L_over_w:.1f}" if L_over_w is not None else "n/a",
            disabled=True,
            key="rpt_L_over_w",
        )

    st.markdown("---")

        # 6.1.a Detailed explanation for check (1) Tension
    st.markdown("#### (1) Tension – EN 1993-1-1 §6.2.3")

    # Get section area and material
    A_mm2 = float(sr_display.get("A_mm2", 0.0))  # from DB
    fy = material_to_fy(material)

    # Design axial force N (tension positive)
    N_kN = float(inputs.get("N_kN", 0.0))

    # Tension: N > 0
    if N_kN > 0.0:
        NEd_ten_kN = N_kN
    else:
        NEd_ten_kN = 0.0

    # Plastic tension resistance Npl,Rd = A * fy / γM0  (convert to kN)
    if A_mm2 > 0.0 and fy > 0.0:
        Npl_Rd_kN = A_mm2 * fy / (gamma_M0 * 1e3)  # N → kN
    else:
        Npl_Rd_kN = 0.0

    # Utilisation from our own numbers
    if Npl_Rd_kN > 0.0:
        u_ten = NEd_ten_kN / Npl_Rd_kN
        u_ten_str = f"{u_ten:.3f}"
    else:
        u_ten = None
        u_ten_str = "n/a"

    # Status from the checks table (row 'Tension (N≥0)')
    status_ten = "n/a"
    try:
        row_ten = df_rows[df_rows["Check"] == "Tension (N≥0)"]
        if not row_ten.empty:
            status_ten = str(row_ten.iloc[0]["Status"])
    except Exception:
        pass

    # --- Text + equations in math format ---

    st.markdown(
        "The critical cross-section is verified for tensile axial force "
        "in accordance with EN 1993-1-1 §6.2.3:"
    )
    st.latex(r"\frac{N_{Ed}}{N_{t,Rd}} \le 1.0")

    st.markdown("Design tensile axial force is taken as:")
    st.latex(rf"N_{{Ed}} = {NEd_ten_kN:.3f}\,\text{{kN}}")

    if Npl_Rd_kN > 0.0:
        st.markdown(
            "The plastic tension resistance of the gross cross-section is estimated as:"
        )
        st.latex(
            rf"N_{{pl,Rd}} = \frac{{A f_y}}{{\gamma_{{M0}}}}"
            rf" = {A_mm2:.1f}\,\text{{mm}}^2 \cdot {fy:.0f}\,\text{{MPa}}"
            rf" / {gamma_M0:.2f}"
            rf" = {Npl_Rd_kN:.3f}\,\text{{kN}}"
        )

        st.markdown("Hence the utilisation for the tension verification is:")
        st.latex(
            rf"u = \frac{{N_{{Ed}}}}{{N_{{pl,Rd}}}}"
            rf" = \frac{{{NEd_ten_kN:.3f}}}{{{Npl_Rd_kN:.3f}}}"
            rf" = {u_ten_str} \le 1.0"
            rf" \Rightarrow \mathbf{{{status_ten}}}."
        )
    else:
        st.markdown(
            "Tension resistance could not be evaluated because cross-section area or "
            "material data is missing in the DB."
        )

    st.caption(
        "Net-section tension per EN 1993-1-1 §6.2.3(2)b) (holes, openings) "
        "is not yet implemented in this prototype."
    )


        # 6.1.b Detailed explanation for check (2) Compression
    st.markdown("#### (2) Compression – EN 1993-1-1 §6.2.4")

    # design compressive axial force (take magnitude of N < 0)
    if N_kN < 0.0:
        NEd_comp_kN = -N_kN
    else:
        NEd_comp_kN = 0.0

    # compression resistance (class 1–3): Nc,Rd = A fy / γM0
    Nc_Rd_kN = Npl_Rd_kN  # same expression for gross section

    # utilisation u = NEd / Nc,Rd from our own numbers
    if Nc_Rd_kN > 0.0 and NEd_comp_kN > 0.0:
        u_comp = NEd_comp_kN / Nc_Rd_kN
        u_comp_str = f"{u_comp:.3f}"
    else:
        u_comp = None
        u_comp_str = "n/a"

    # status from results table row "Compression (N<0)"
    status_comp = "n/a"
    try:
        row_comp = df_rows[df_rows["Check"] == "Compression (N<0)"]
        if not row_comp.empty:
            status_comp = str(row_comp.iloc[0]["Status"])
            # if you want, you can also overwrite u_comp_str from the table
    except Exception:
        pass

    # --- text + equations in math format ---
    st.markdown(
        "The critical cross-section is verified for compressive axial force "
        "in accordance with EN 1993-1-1 §6.2.4:"
    )
    st.latex(r"\frac{N_{Ed}}{N_{c,Rd}} \le 1.0")

    st.markdown("Design compressive axial force (compression taken as negative in the global sign convention):")
    st.latex(rf"N_{{Ed}} = {NEd_comp_kN:.3f}\,\text{{kN}}")

    if Nc_Rd_kN > 0.0:
        st.markdown("Compression resistance of the gross cross-section (class 1–3):")
        st.latex(
            rf"N_{{c,Rd}} = \frac{{A f_y}}{{\gamma_{{M0}}}}"
            rf" = {A_mm2:.1f}\,\text{{mm}}^2 \cdot {fy:.0f}\,\text{{MPa}}"
            rf" / {gamma_M0:.2f}"
            rf" = {Nc_Rd_kN:.3f}\,\text{{kN}}"
        )

        if u_comp is not None:
            st.markdown("Therefore the utilisation for the compression verification is:")
            st.latex(
                rf"u = \frac{{N_{{Ed}}}}{{N_{{c,Rd}}}}"
                rf" = \frac{{{NEd_comp_kN:.3f}}}{{{Nc_Rd_kN:.3f}}}"
                rf" = {u_comp_str} \le 1.0"
                rf" \Rightarrow \mathbf{{{status_comp}}}."
            )
    else:
        st.markdown(
            "Compression resistance could not be evaluated because cross-section area "
            "or material data is missing in the DB."
        )

    st.markdown("### 6.2 Detailed checks table (1–14)")
    # Row-highlighting for detailed check table
    def _hl(row):
        # make sure we pull a scalar, not a Series
        s = row["Status"] if "Status" in row.index else ""
        if s == "OK":
            color = "background-color: #e6f7e6"
        elif s == "EXCEEDS":
            color = "background-color: #fde6e6"
        else:
            color = "background-color: #f0f0f0"
        return [color] * len(row)

    # Be defensive: if styling fails, fall back to plain table
    try:
        st.write(df_rows.style.apply(_hl, axis=1))
    except Exception as e:
        st.write(df_rows)
        st.caption(f"(Row colouring disabled due to an internal error: {e})")

    # 6.3 Explanations & formulas (expander, with LaTeX)
    with st.expander("6.3 Eurocode formulas for cross-section checks (1–14)", expanded=False):
        # (1) Tension
        st.markdown("### (1) Tension – EN 1993-1-1 §6.2.3")
        st.latex(r"\frac{N_{Ed}}{N_{t,Rd}} \leq 1.0")
        st.latex(r"N_{t,Rd} = N_{pl,Rd} = \frac{A\,f_y}{\gamma_{M0}}")
        
        # (2) Compression
        st.markdown("### (2) Compression – EN 1993-1-1 §6.2.4")
        st.latex(r"\frac{N_{Ed}}{N_{c,Rd}} \leq 1.0")
        st.latex(r"N_{c,Rd} = \frac{A\,f_y}{\gamma_{M0}}")

        st.markdown("### (3) Bending about major axis y – §6.2.5")
        st.latex(r"M_{y,Ed} \leq M_{y,Rd} = \frac{W_{pl,y} f_y}{\gamma_{M0}}")

        st.markdown("### (4) Bending about minor axis z – §6.2.5")
        st.latex(r"M_{z,Ed} \leq M_{z,Rd} = \frac{W_{pl,z} f_y}{\gamma_{M0}}")

        st.markdown("### (5–6) Shear – §6.2.6")
        st.latex(r"V_{Ed} \leq V_{pl,Rd} = \frac{A_v f_y}{\sqrt{3}\,\gamma_{M0}}")

        st.markdown("### (7–8) Bending with high shear – §6.2.8")
        st.latex(r"V_{Ed} > 0.5\,V_{pl,Rd} \Rightarrow M_{Rd,\text{red}} = \rho\, M_{pl}")

        st.markdown("### (9–11) Axial force + bending – §6.2.9")
        st.latex(r"\frac{N_{Ed}}{N_{c,Rd}} + \frac{M_{y,Ed}}{M_{y,Rd}} \leq 1.0")
        st.latex(r"\frac{N_{Ed}}{N_{c,Rd}} + \frac{M_{z,Ed}}{M_{z,Rd}} \leq 1.0")

        st.markdown("### (12–14) Axial + shear + bending – §6.2.10")
        st.latex(r" M_{Rd,\text{red}} = \rho\, M_{pl} ")
        st.latex(r" \frac{N_{Ed}}{N_{c,Rd}} + \frac{M_{Ed}}{M_{Rd,\text{red}}} \leq 1.0 ")

    st.markdown("---")

    # ----------------------------------------------------
    # 7. Member stability (buckling)
    # ----------------------------------------------------
    st.markdown("## 7. Member stability (buckling)")

    st.markdown("### 7.1 Buckling summary table")
    buck_data = []
    for axis_label, Ncr, lambda_bar, chi, N_b_Rd_N, status in buck_results or []:
        if N_b_Rd_N is None:
            continue
        buck_data.append(
            {
                "Axis / mode": f"Flexural buckling {axis_label}",
                "N_cr [kN]": f"{Ncr/1e3:.2f}" if Ncr is not None else "n/a",
                "λ̄": f"{lambda_bar:.3f}" if lambda_bar is not None else "n/a",
                "χ": f"{chi:.3f}" if chi is not None else "n/a",
                "N_b,Rd [kN]": f"{N_b_Rd_N/1e3:.2f}",
                "Status": status,
            }
        )

    if buck_data:
        df_buck = pd.DataFrame(buck_data)
        st.table(df_buck)
    else:
        st.info("No buckling results available.")

    st.markdown("### 7.2 Detailed buckling expressions (per axis)")
    if buck_data:
        for row in buck_data:
            st.markdown(f"**Axis / mode:** {row['Axis / mode']}")
            st.write(
                f"N_cr = {row['N_cr [kN]']} kN, "
                f"λ̄ = {row['λ̄']}, "
                f"χ = {row['χ']}, "
                f"N_b,Rd = {row['N_b,Rd [kN]']} kN → {row['Status']}"
            )
    else:
        st.write("No detailed values to display.")

    # 7.3 Buckling formulas expander
    with st.expander("7.3 Eurocode formulas for member buckling", expanded=False):
        st.markdown("### Flexural buckling – EN 1993-1-1 §6.3.1")

        st.latex(r"N_{cr} = \frac{\pi^2 E I}{L_{eff}^2}")
        st.latex(r"\bar{\lambda} = \sqrt{\frac{A f_y}{N_{cr}}}")

        st.latex(
            r"\phi = \frac{1}{2} \left[ 1 + \alpha \left( \bar{\lambda} - 0.2 \right) + \bar{\lambda}^2 \right]"
        )
        st.latex(
            r"\chi = \frac{1}{\phi + \sqrt{\phi^2 - \bar{\lambda}^2}} \leq 1.0"
        )
        st.latex(
            r"N_{b,Rd} = \frac{\chi A f_y}{\gamma_{M1}}"
        )

        st.markdown("### Lateral–torsional buckling (LTB) – simplified")
        st.latex(
            r"k_{yy}^* \frac{M_{y,Ed}}{\chi_{LT} \, M_{y,Rk} / \gamma_{M1}} \leq 1.0"
        )
        st.latex(
            r"M_{b,Rd} = \frac{\chi_{LT} \, M_{y,Rk}}{\gamma_{M1}}"
        )

    st.markdown("---")

    # ----------------------------------------------------
    # 8. References
    # ----------------------------------------------------
    st.markdown("## 8. References")

    st.markdown(
        """
- EN 1993-1-1: Eurocode 3 – Design of steel structures – Part 1-1  
- EN 1990: Basis of structural design  
- EN 1991 series: Actions on structures  
- National Annex to EN 1993-1-1 (where applicable)  
- EngiSnap – Standard steel beam design & selection (this prototype)
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

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["1) Project info", "2) Loads", "3) Section", "4) Results", "5) Report"]
)

with tab1:
    st.subheader("Project information")
    meta = render_project_data()
    st.session_state["meta"] = meta

with tab2:
    st.subheader("Loads settings")

    # --- 1) Design settings (ULS) ---
    with st.expander("Design settings (ULS)", expanded=True):
        gamma_choice = st.radio(
            "Load factor γ_F",
            ["1.35 (static)", "1.50 (dynamic)", "Custom"],
            key="gammaF_choice",
        )

        if gamma_choice == "Custom":
            gamma_F = st.number_input(
                "γ_F (custom)",
                min_value=0.0,
                value=float(st.session_state.get("gamma_F", 1.50)),
                key="gammaF_custom",
            )
        elif gamma_choice == "1.35 (static)":
            gamma_F = 1.35
        else:
            gamma_F = 1.50

        st.session_state["gamma_F"] = gamma_F

        manual_forces_type = st.radio(
            "Manual internal forces are",
            ["Characteristic", "Design values (N_Ed, M_Ed, …)"],
            key="manual_forces_type",
        )
        # no need to write back to session_state; Streamlit does that via the key

    # --- 2) Effective lengths for instability ---
        # --- 2) Instability length ratios (relative to span L) ---
    with st.expander("Instability length ratios (relative to span L)", expanded=False):
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            Lcr_y_over_L = st.number_input(
                "Lcr,y / L  (flexural y-y)",
                min_value=0.1,
                value=float(st.session_state.get("Lcr_y_over_L", 1.0)),
                step=0.05,
                key="Lcr_y_over_L",
            )

        with c2:
            Lcr_z_over_L = st.number_input(
                "Lcr,z / L  (flexural z-z)",
                min_value=0.1,
                value=float(st.session_state.get("Lcr_z_over_L", 1.0)),
                step=0.05,
                key="Lcr_z_over_L",
            )

        with c3:
            L_LT_over_L = st.number_input(
                "L_LT / L  (lateral–torsional)",
                min_value=0.1,
                value=float(st.session_state.get("L_LT_over_L", 1.0)),
                step=0.05,
                key="L_LT_over_L",
            )

        with c4:
            L_TF_over_L = st.number_input(
                "L_TF / L  (torsional / flexural–torsional)",
                min_value=0.1,
                value=float(st.session_state.get("L_TF_over_L", 1.0)),
                step=0.05,
                key="L_TF_over_L",
            )

    # --- 3) Determine section family for torsion (if already chosen in Section tab) ---
    sr_display_for_loads = st.session_state.get("sr_display")
    if isinstance(sr_display_for_loads, dict):
        family_for_torsion = sr_display_for_loads.get("family", "")
    else:
        family_for_torsion = ""

    # --- 4) Mode selector: ready beam case vs manual loads ---
    load_mode = st.radio(
        "How do you want to define loading for this member?",
        ["Use ready beam case", "Enter loads manually"],
        horizontal=True,
        key="load_mode_choice",
    )

    # If ready beam case: show the gallery
    if load_mode == "Use ready beam case":
        render_ready_cases_panel()
    else:
        st.info("Manual loads mode selected: use the form below to enter forces and moments directly.")

    # --- 5) Loads form (always shown) ---
    # When a ready case is applied, it just PREFILLS this form via session_state.
    # In ready-case mode the inputs are read-only; in manual mode they are editable.
    render_loads_form(family_for_torsion, read_only=False)

with tab3:
    material, family, selected_name, selected_row, detected_table = render_section_selection()
    st.session_state["material"] = material

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
                    width=320,
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

    else:
        st.info("Select a section to continue.")

with tab4:
    sr_display = st.session_state.get("sr_display", None)
    material = st.session_state.get("material", "S355")
    fy = material_to_fy(material)

    # Run button moved here from Loads tab
    run_col, _ = st.columns([1, 3])
    with run_col:
        if st.button("Run check", key="run_check_results"):
            try:
                store_design_forces_from_state()
            except Exception as e:
                st.error(f"Error reading Loads inputs: {e}")

    if not st.session_state.get("run_clicked", False):
        st.info("Set up Loads in Tab 2 and then press **Run check** here.")
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
with tab5:
    render_report_tab()






