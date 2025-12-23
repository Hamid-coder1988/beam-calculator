import streamlit as st
import pandas as pd
import math
import textwrap
from io import BytesIO
from datetime import datetime, date
import traceback
import re
import numbers
import numpy as np
import matplotlib.pyplot as plt
import io
from pathlib import Path

# -------------------------
# Asset path helpers
# -------------------------
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

def asset_path(rel: str) -> Path:
    """Return absolute path to a bundled asset (logo/images) relative to this file."""
    return (BASE_DIR / rel).resolve()

def safe_image(path_like, **kwargs) -> bool:
    """Show an image only if it exists. Returns True if shown, else False."""
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
    from reportlab.lib.utils import ImageReader  # for PDF images
    HAS_RL = True
except Exception:
    HAS_RL = False
    ImageReader = None  # so we can safely reference it later

# (BytesIO already imported at the top; no need to import again)
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

# Heading fonts
def report_h3(title):
    st.markdown(
        f"""
        <div style='
            font-weight:600;
            font-size:1.25rem;
            margin-top:25px;
            margin-bottom:12px;
        '>{title}</div>
        """,
        unsafe_allow_html=True
    )

def report_h4(title):
    st.markdown(
        f"""
        <div style='
            font-weight:550;
            font-size:1.05rem;
            margin-top:18px;
            margin-bottom:8px;
        '>{title}</div>
        """,
        unsafe_allow_html=True
    )

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

@st.cache_data(show_spinner=False)
def run_sql(sql, params=None):
    """
    Execute a SQL query using psycopg2 and return (df, err).

    - df  : pandas.DataFrame with the result (or None on error)
    - err : None if OK, or string with error message
    """
    if not HAS_PG:
        return None, "Postgres driver not available (HAS_PG = False)"

    conn = None
    cur = None
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(sql, params or ())
        # If the query returns rows
        if cur.description:
            cols = [c[0] for c in cur.description]
            rows = cur.fetchall()
            df = pd.DataFrame(rows, columns=cols)
        else:
            df = pd.DataFrame()
        return df, None
    except Exception as e:
        return None, str(e)
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

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
        p = asset_path("sections_img/IPE.png")
        return str(p) if p.exists() else None  # <-- FIXED PATH

    # Group 2: own image
    GROUP_OWN = ["SHS", "RHS", "CHS", "UPE", "UPN", "PFC", "L"]

    if family in GROUP_OWN:
        p = asset_path(f"sections_img/{family}.png")
        return str(p) if p.exists() else None  # <-- FIXED PATH

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
                    value=float(st.session_state.get("L_in", 6.0)),
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

def compute_checks(use_props, fy, inputs, torsion_supported):
    L = inputs["L"]
    N_kN = inputs["N_kN"]
    Vy_kN = inputs["Vy_kN"]
    Vz_kN = inputs["Vz_kN"]
    My_kNm = inputs["My_kNm"]
    Mz_kNm = inputs["Mz_kNm"]
    Tx_kNm = inputs.get("Tx_kNm", 0.0)
    K_y = inputs["K_y"]
    K_z = inputs["K_z"]

    # Aliases for design internal forces (Ed) used in results / report
    My_Ed_kNm = My_kNm
    Mz_Ed_kNm = Mz_kNm
    Vy_Ed_kN = Vy_kN
    Vz_Ed_kN = Vz_kN

    # Convert to base units
    N_N = N_kN * 1e3
    Vy_N = Vy_kN * 1e3
    Vz_N = Vz_kN * 1e3
    My_Nm = My_kNm * 1e3
    Mz_Nm = Mz_kNm * 1e3
    T_Nm = Tx_kNm * 1e3

    # Section properties from DB
    A_m2 = use_props.get("A_cm2", 0.0) / 1e4
    S_y_m3 = use_props.get("S_y_cm3", 0.0) * 1e-6
    S_z_m3 = use_props.get("S_z_cm3", 0.0) * 1e-6
    I_y_m4 = use_props.get("I_y_cm4", 0.0) * 1e-8
    I_z_m4 = use_props.get("I_z_cm4", 0.0) * 1e-8
    J_m4 = use_props.get("J_cm4", 0.0) * 1e-8
    c_max_m = use_props.get("c_max_mm", 0.0) / 1000.0

    Wpl_y_cm3 = use_props.get("Wpl_y_cm3", 0.0)
    Wpl_z_cm3 = use_props.get("Wpl_z_cm3", 0.0)
    Wel_y_cm3 = use_props.get("Wel_y_cm3", 0.0)
    Wel_z_cm3 = use_props.get("Wel_z_cm3", 0.0)
    # Shear areas: support both old and new keys
    Av_z_mm2 = use_props.get("Av_z_mm2", use_props.get("Avz_mm2", 0.0))
    Av_y_mm2 = use_props.get("Av_y_mm2", use_props.get("Avy_mm2", 0.0))

    # Imperfection factor (buckling curve) can come as:
    # - a numeric alpha (0.21, 0.34, 0.49, 0.76, ...)
    # - or a curve group string: a0 / a / b / c / d (from your DB)
    _ALPHA_FROM_CURVE = {"a0": 0.13, "a": 0.21, "b": 0.34, "c": 0.49, "d": 0.76}
    
    curve_or_alpha = (
        use_props.get("imperfection_group", None)
        or use_props.get("buckling_curve", None)
        or use_props.get("alpha_curve", None)
    )
    
    if isinstance(curve_or_alpha, str):
        alpha_curve_db = _ALPHA_FROM_CURVE.get(curve_or_alpha.strip().lower(), 0.49)
    else:
        alpha_curve_db = float(curve_or_alpha) if curve_or_alpha is not None else 0.49
    

    if A_m2 <= 0:
        raise ValueError("Section area not provided (A <= 0).")

    # Basic resistances (cross-section)
    N_Rd_N = A_m2 * fy * 1e6 / gamma_M0
    T_Rd_N = N_Rd_N
    Wpl_y_m3 = (Wpl_y_cm3 * 1e-6) if Wpl_y_cm3 > 0 else 1.1 * S_y_m3
    M_Rd_y_Nm = Wpl_y_m3 * fy * 1e6 / gamma_M0
    Av_m2 = 0.6 * A_m2
    V_Rd_N = Av_m2 * fy * 1e6 / (math.sqrt(3) * gamma_M0)

    # Allowable stress for indicative combined check
    sigma_allow_MPa = 0.6 * fy

    # Stresses
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
    # Cross-section checks: axial, bending, shear
    # -----------------------------------------------
    rows = []

    # Axial: sign convention N>0 tension, N<0 compression
    N_ten_N = max(N_N, 0.0)
    N_comp_N = max(-N_N, 0.0)

    # (1) Tension (N>0)
    if T_Rd_N > 0.0 and N_ten_N > 0.0:
        util_ten = N_ten_N / T_Rd_N
        status_ten = "OK" if util_ten <= 1.0 else "EXCEEDS"
    else:
        util_ten = 0.0
        status_ten = "OK"

    rows.append({
        "Check":      "Tension (N>0)",
        "Applied":    f"{N_ten_N/1e3:.3f} kN",
        "Resistance": f"{T_Rd_N/1e3:.3f} kN",
        "Utilization": f"{util_ten:.3f}",
        "Status":      status_ten,
    })

    # (2) Compression (N<0) – cross-section
    if N_Rd_N > 0.0 and N_comp_N > 0.0:
        util_comp = N_comp_N / N_Rd_N
        status_comp = "OK" if util_comp <= 1.0 else "EXCEEDS"
    else:
        util_comp = 0.0
        status_comp = "OK"

    rows.append({
        "Check":      "Compression (N<0)",
        "Applied":    f"{-N_comp_N/1e3:.3f} kN",
        "Resistance": f"{N_Rd_N/1e3:.3f} kN",
        "Utilization": f"{util_comp:.3f}",
        "Status":      status_comp,
    })

    # ---- Bending section modulus based on class ----
    section_class = use_props.get("class", 1)
    if section_class in (1, 2):
        W_y_cm3 = Wpl_y_cm3
        W_z_cm3 = Wpl_z_cm3
    else:
        W_y_cm3 = Wel_y_cm3
        W_z_cm3 = Wel_z_cm3

    W_y_mm3 = W_y_cm3 * 1e3
    W_z_mm3 = W_z_cm3 * 1e3

    # (3) Major-axis bending Mc,y,Rd
    if W_y_mm3 > 0 and fy > 0:
        Mc_y_Rd_kNm = (W_y_mm3 * fy / gamma_M0) / 1e6
    else:
        Mc_y_Rd_kNm = 0.0

    # (4) Minor-axis bending Mc,z,Rd
    if W_z_mm3 > 0 and fy > 0:
        Mc_z_Rd_kNm = (W_z_mm3 * fy / gamma_M0) / 1e6
    else:
        Mc_z_Rd_kNm = 0.0

    # Utilizations for bending
    if Mc_y_Rd_kNm > 0:
        util_My = My_Ed_kNm / Mc_y_Rd_kNm
        status_My = "OK" if util_My <= 1.0 else "EXCEEDS"
    else:
        util_My = None
        status_My = "n/a"

    if Mc_z_Rd_kNm > 0:
        util_Mz = Mz_Ed_kNm / Mc_z_Rd_kNm
        status_Mz = "OK" if util_Mz <= 1.0 else "EXCEEDS"
    else:
        util_Mz = None
        status_Mz = "n/a"

    rows.append({
        "Check": "(3) Bending My (major)",
        "Applied": f"{My_Ed_kNm:.3f} kNm",
        "Resistance": f"{Mc_y_Rd_kNm:.3f} kNm",
        "Utilization": f"{util_My:.3f}" if util_My is not None else "n/a",
        "Status": status_My,
    })

    rows.append({
        "Check": "(4) Bending Mz (minor)",
        "Applied": f"{Mz_Ed_kNm:.3f} kNm",
        "Resistance": f"{Mc_z_Rd_kNm:.3f} kNm",
        "Utilization": f"{util_Mz:.3f}" if util_Mz is not None else "n/a",
        "Status": status_Mz,
    })
    # ---- Shear resistances along z and y (always calculated) ----
    if Av_z_mm2 > 0 and fy > 0:
        Vc_z_Rd_kN = (Av_z_mm2 * (fy / math.sqrt(3)) / gamma_M0) / 1e3
    else:
        Vc_z_Rd_kN = 0.0
    
    if Av_y_mm2 > 0 and fy > 0:
        Vc_y_Rd_kN = (Av_y_mm2 * (fy / math.sqrt(3)) / gamma_M0) / 1e3
    else:
        Vc_y_Rd_kN = 0.0

    # Utilizations for shear
    if Vc_z_Rd_kN > 0:
        util_Vz = Vz_Ed_kN / Vc_z_Rd_kN
        status_Vz = "OK" if util_Vz <= 1.0 else "EXCEEDS"
    else:
        util_Vz = None
        status_Vz = "n/a"

    if Vc_y_Rd_kN > 0:
        util_Vy = Vy_Ed_kN / Vc_y_Rd_kN
        status_Vy = "OK" if util_Vy <= 1.0 else "EXCEEDS"
    else:
        util_Vy = None
        status_Vy = "n/a"

    rows.append({
        "Check": "(5) Shear Vz (z-axis)",
        "Applied": f"{Vz_Ed_kN:.3f} kN",
        "Resistance": f"{Vc_z_Rd_kN:.3f} kN",
        "Utilization": f"{util_Vz:.3f}" if util_Vz is not None else "n/a",
        "Status": status_Vz,
    })

    rows.append({
        "Check": "(6) Shear Vy (y-axis)",
        "Applied": f"{Vy_Ed_kN:.3f} kN",
        "Resistance": f"{Vc_y_Rd_kN:.3f} kN",
        "Utilization": f"{util_Vy:.3f}" if util_Vy is not None else "n/a",
        "Status": status_Vy,
    })

    # -------------------------------------------------
    # Combined effects placeholders required for summary tables (checks 7–14)
    # -------------------------------------------------
    # Shear influence on bending (EN 1993-1-1 §6.2.8): if VEd <= 0.5*Vpl,Rd => ignore
    shear_ratio_y = (Vy_Ed_kN / Vc_y_Rd_kN) if (Vc_y_Rd_kN and Vc_y_Rd_kN > 0) else None
    shear_ratio_z = (Vz_Ed_kN / Vc_z_Rd_kN) if (Vc_z_Rd_kN and Vc_z_Rd_kN > 0) else None

    shear_ok_y = (shear_ratio_y is not None) and (shear_ratio_y <= 0.50)
    shear_ok_z = (shear_ratio_z is not None) and (shear_ratio_z <= 0.50)

    # Axial influence on bending (EN 1993-1-1 §6.2.9): I/H doubly symmetric heuristic
    # Pull basic geometric dims if present (used for hw*tw criteria)
    h_mm = float(use_props.get("h_mm", use_props.get("h", 0.0)) or 0.0)
    b_mm = float(use_props.get("b_mm", use_props.get("b", 0.0)) or 0.0)
    tw_mm = float(use_props.get("tw_mm", use_props.get("tw", 0.0)) or 0.0)
    tf_mm = float(use_props.get("tf_mm", use_props.get("tf", 0.0)) or 0.0)
    r_mm  = float(use_props.get("r_mm",  use_props.get("r", 0.0))  or 0.0)

    A_mm2 = A_m2 * 1e6
    Npl_Rd_kN = (A_mm2 * fy / gamma_M0) / 1e3 if (A_mm2 > 0 and fy > 0) else 0.0

    hw_mm = 0.0
    if h_mm > 0 and tf_mm > 0:
        # reasonable web height approximation
        hw_mm = max(h_mm - 2.0 * tf_mm, 0.0)

    crit_y_25 = 0.25 * Npl_Rd_kN
    crit_y_web = (0.50 * hw_mm * tw_mm * fy / gamma_M0) / 1e3 if (hw_mm > 0 and tw_mm > 0 and fy > 0) else None
    crit_z_web = (hw_mm * tw_mm * fy / gamma_M0) / 1e3 if (hw_mm > 0 and tw_mm > 0 and fy > 0) else None

    NEd_kN = abs(N_kN)
    axial_ok_y = (NEd_kN <= crit_y_25) and (crit_y_web is None or NEd_kN <= crit_y_web)
    axial_ok_z = (crit_z_web is None) or (NEd_kN <= crit_z_web)

    # Prepare detail values for report
    cs_combo = dict(
        shear_ratio_y=shear_ratio_y,
        shear_ratio_z=shear_ratio_z,
        shear_ok_y=shear_ok_y,
        shear_ok_z=shear_ok_z,
        Npl_Rd_kN=Npl_Rd_kN,
        hw_mm=hw_mm,
        tw_mm=tw_mm,
        crit_y_25=crit_y_25,
        crit_y_web=crit_y_web,
        crit_z_web=crit_z_web,
        NEd_kN=NEd_kN,
        axial_ok_y=axial_ok_y,
        axial_ok_z=axial_ok_z,
        util_My=util_My,
        util_Mz=util_Mz,
    )

    # Indicative stress-based checks
    rows.append({
        "Check": "Bending y-y (σ_by)",
        "Applied": f"{sigma_by_Pa/1e6:.3f} MPa",
        "Resistance": f"{sigma_allow_MPa:.3f} MPa",
        "Utilization": f"{(abs(sigma_by_Pa/1e6)/sigma_allow_MPa):.3f}" if sigma_allow_MPa > 0 else "n/a",
        "Status": "OK" if abs(sigma_by_Pa/1e6)/sigma_allow_MPa <= 1.0 else "EXCEEDS",
    })

    rows.append({
        "Check": "Bending z-z (σ_bz)",
        "Applied": f"{sigma_bz_Pa/1e6:.3f} MPa",
        "Resistance": f"{sigma_allow_MPa:.3f} MPa",
        "Utilization": f"{(abs(sigma_bz_Pa/1e6)/sigma_allow_MPa):.3f}" if sigma_allow_MPa > 0 else "n/a",
        "Status": "OK" if abs(sigma_bz_Pa/1e6)/sigma_allow_MPa <= 1.0 else "EXCEEDS",
    })

    rows.append({
        "Check": "Biaxial bending + axial + shear (indicative)",
        "Applied": f"σ_eq={sigma_eq_MPa:.3f} MPa",
        "Resistance": f"{sigma_allow_MPa:.3f} MPa",
        "Utilization": f"{(sigma_eq_MPa/sigma_allow_MPa):.3f}" if sigma_allow_MPa > 0 else "n/a",
        "Status": "OK" if sigma_eq_MPa/sigma_allow_MPa <= 1.0 else "EXCEEDS",
    })

    # -------------------------
    # Buckling checks (EN 1993-1-1 §6.3)
    # -------------------------
    E = 210e9
    nu = 0.30
    G = E / (2.0 * (1.0 + nu))

    # Pull basic geometric dims if present (used in curve selection and hw*tw criteria)
    h_mm = float(use_props.get("h_mm", use_props.get("h", 0.0)) or 0.0)
    b_mm = float(use_props.get("b_mm", use_props.get("b", 0.0)) or 0.0)
    tw_mm = float(use_props.get("tw_mm", use_props.get("tw", 0.0)) or 0.0)
    tf_mm = float(use_props.get("tf_mm", use_props.get("tf", 0.0)) or 0.0)
    r_mm  = float(use_props.get("r_mm",  use_props.get("r", 0.0))  or 0.0)

    # Warping constant (if available in DB)
    Iw_m6 = use_props.get("Iw_mm6", None)
    if Iw_m6 is None:
        Iw_cm6 = use_props.get("Iw_cm6", None)
        if Iw_cm6 is None:
            Iw_m6 = 0.0
        else:
            Iw_m6 = float(Iw_cm6) * 1e-12  # cm^6 -> m^6
    else:
        Iw_m6 = float(Iw_m6) * 1e-18  # mm^6 -> m^6

    # Plastic section moduli in m^3 (prefer plastic)
    Wpl_y_m3 = (Wpl_y_cm3 or 0.0) * 1e-6
    Wpl_z_m3 = (Wpl_z_cm3 or 0.0) * 1e-6
    Wel_y_m3 = (Wel_y_cm3 or 0.0) * 1e-6
    Wel_z_m3 = (Wel_z_cm3 or 0.0) * 1e-6

    Wy_m3 = Wpl_y_m3 if Wpl_y_m3 > 0 else S_y_m3
    Wz_m3 = Wpl_z_m3 if Wpl_z_m3 > 0 else S_z_m3

    # Characteristic resistances (no material factors) for interaction checks
    NRk_N = A_m2 * fy * 1e6
    My_Rk_Nm = Wy_m3 * fy * 1e6
    Mz_Rk_Nm = Wz_m3 * fy * 1e6

    # Determine imperfection factors (rolled I/H heuristic; fallback to DB alpha)
    alpha_y = float(alpha_curve_db) if alpha_curve_db is not None else 0.49
    alpha_z = float(alpha_curve_db) if alpha_curve_db is not None else 0.49
    alpha_LT = 0.34  # rolled I-sections curve b (common default)

    if b_mm > 0 and h_mm > 0 and (h_mm / b_mm) > 1.2 and (tf_mm <= 40.0):
        alpha_y = 0.21  # curve a
        alpha_z = 0.34  # curve b

    def chi_reduction(lambda_bar: float, alpha: float) -> float:
        # EN 1993-1-1 §6.3.1.2
        phi = 0.5 * (1.0 + alpha * (lambda_bar - 0.20) + lambda_bar**2)
        sqrt_term = max(phi**2 - lambda_bar**2, 0.0)
        denom = phi + math.sqrt(sqrt_term)
        return min(1.0, 1.0 / denom) if denom > 0 else 0.0

    # Flexural buckling about y and z
    buck_results = []
    buck_map = {}  # for report/results mapping
    for axis_label, I_axis, K_axis, alpha_use in [
        ("y", I_y_m4, K_y, alpha_y),
        ("z", I_z_m4, K_z, alpha_z),
    ]:
        if I_axis <= 0:
            buck_results.append((axis_label, None, None, None, None, "No I"))
            buck_map[f"Ncr_{axis_label}"] = None
            continue

        Leff_axis = float(K_axis) * float(L)
        Ncr = (math.pi**2 * E * I_axis) / (Leff_axis**2) if Leff_axis > 0 else 0.0
        lambda_bar = math.sqrt(NRk_N / Ncr) if Ncr > 0 else float("inf")
        chi = chi_reduction(lambda_bar, alpha_use)
        Nb_Rd_N = chi * NRk_N / gamma_M1

        util = (abs(N_N) / Nb_Rd_N) if Nb_Rd_N > 0 else float("inf")
        status = "OK" if util <= 1.0 else "EXCEEDS"

        buck_results.append((axis_label, Ncr, lambda_bar, chi, Nb_Rd_N, status))
        buck_map[f"Ncr_{axis_label}"] = Ncr
        buck_map[f"lambda_{axis_label}"] = lambda_bar
        buck_map[f"chi_{axis_label}"] = chi
        buck_map[f"Nb_Rd_{axis_label}"] = Nb_Rd_N
        buck_map[f"util_buck_{axis_label}"] = util
        buck_map[f"status_buck_{axis_label}"] = status

        rows.append({
            "Check": f"Flexural buckling {axis_label}",
            "Applied": f"{abs(N_N)/1e3:.3f} kN",
            "Resistance": f"{Nb_Rd_N/1e3:.3f} kN",
            "Utilization": f"{util:.3f}",
            "Status": status,
        })

    # (17) Torsional / torsional-flexural buckling (approx, doubly symmetric)
    # Only meaningful if Iw is available.
    i_y_m = math.sqrt(I_y_m4 / A_m2) if (A_m2 > 0 and I_y_m4 > 0) else 0.0
    i_z_m = math.sqrt(I_z_m4 / A_m2) if (A_m2 > 0 and I_z_m4 > 0) else 0.0
    i0_m = math.sqrt(i_y_m**2 + i_z_m**2)

    K_T = float(inputs.get("K_T", 1.0))
    Leff_T = K_T * L

    Ncr_T = None
    chi_T = None
    Nb_Rd_T_N = None
    util_T = None
    status_T = "n/a"

    if i0_m > 0 and J_m4 > 0 and Iw_m6 > 0 and Leff_T > 0:
        Ncr_T = (1.0 / (i0_m**2)) * (G * J_m4 + (math.pi**2) * E * Iw_m6 / (Leff_T**2))
        lambda_T = math.sqrt(NRk_N / Ncr_T) if Ncr_T > 0 else float("inf")
        chi_T = chi_reduction(lambda_T, alpha_z)  # use minor-axis curve
        Nb_Rd_T_N = chi_T * NRk_N / gamma_M1
        util_T = abs(N_N) / Nb_Rd_T_N if Nb_Rd_T_N and Nb_Rd_T_N > 0 else float("inf")
        status_T = "OK" if util_T <= 1.0 else "EXCEEDS"

        rows.append({
            "Check": "Torsional / torsional-flexural buckling",
            "Applied": f"{abs(N_N)/1e3:.3f} kN",
            "Resistance": f"{Nb_Rd_T_N/1e3:.3f} kN",
            "Utilization": f"{util_T:.3f}",
            "Status": status_T,
        })

    buck_map["i0_m"] = i0_m
    buck_map["Ncr_T"] = Ncr_T
    buck_map["chi_T"] = chi_T
    buck_map["Nb_Rd_T"] = Nb_Rd_T_N
    buck_map["util_T"] = util_T
    buck_map["status_T"] = status_T

    # (18) Lateral-torsional buckling (uniform moment, zg=0, k=kw=1; NCCI-style)
    K_LT = float(inputs.get("K_LT", 1.0))
    Leff_LT = K_LT * L

    Mcr = None
    lambda_LT = None
    chi_LT = None
    Mb_Rd_Nm = None
    util_LT = None
    status_LT = "n/a"

    if Leff_LT > 0 and I_z_m4 > 0 and J_m4 > 0 and Iw_m6 > 0 and Wy_m3 > 0:
        term = (Iw_m6 / I_z_m4) + (Leff_LT**2) * G * J_m4 / ((math.pi**2) * E * I_z_m4)
        Mcr = (math.pi**2) * E * I_z_m4 / (Leff_LT**2) * math.sqrt(max(term, 0.0))
        lambda_LT = math.sqrt((Wy_m3 * fy * 1e6) / Mcr) if Mcr > 0 else float("inf")

        lambda_LT0 = 0.40
        beta = 0.75
        phi_LT = 0.5 * (1.0 + alpha_LT * (lambda_LT - lambda_LT0) + beta * lambda_LT**2)
        sqrt_term = max(phi_LT**2 - beta * lambda_LT**2, 0.0)
        chi_LT = min(
            1.0,
            (1.0 / (lambda_LT**2)) if lambda_LT > 0 else 1.0,
            (1.0 / (phi_LT + math.sqrt(sqrt_term))) if (phi_LT + math.sqrt(sqrt_term)) > 0 else 0.0,
        )

        Mb_Rd_Nm = chi_LT * Wy_m3 * fy * 1e6 / gamma_M1
        util_LT = abs(My_Ed_kNm * 1e3) / Mb_Rd_Nm if Mb_Rd_Nm and Mb_Rd_Nm > 0 else float("inf")
        status_LT = "OK" if util_LT <= 1.0 else "EXCEEDS"

        rows.append({
            "Check": "Lateral-torsional buckling",
            "Applied": f"{abs(My_Ed_kNm):.3f} kNm",
            "Resistance": f"{Mb_Rd_Nm/1e3:.3f} kNm",
            "Utilization": f"{util_LT:.3f}",
            "Status": status_LT,
        })

    buck_map["Leff_LT"] = Leff_LT
    buck_map["Mcr"] = Mcr
    buck_map["lambda_LT"] = lambda_LT
    buck_map["chi_LT"] = chi_LT
    buck_map["Mb_Rd"] = Mb_Rd_Nm
    buck_map["util_LT"] = util_LT
    buck_map["status_LT"] = status_LT

    # (19)/(20) Buckling interaction for bending + axial compression (Annex A / Annex B style)
    # NOTE: We assume uniform moment diagram (ψ=1) since the app takes single My, Mz.
    psi_y = 1.0
    psi_z = 1.0
    psi_LT = 1.0

    # Pull flexural chi factors from buck_results (fallbacks)
    chi_y = buck_map.get("chi_y", 1.0) or 1.0
    chi_z = buck_map.get("chi_z", 1.0) or 1.0
    chiLT = chi_LT if chi_LT is not None else 1.0

    Ncr_y = buck_map.get("Ncr_y", None) or 0.0
    Ncr_z = buck_map.get("Ncr_z", None) or 0.0
    Ncr_T_use = Ncr_T if (Ncr_T is not None) else 0.0

    lam_y = buck_map.get("lambda_y", None) or 0.0
    lam_z = buck_map.get("lambda_z", None) or 0.0

    # interaction utilities
    util_int_A = None
    util_int_B = None

    if NRk_N > 0 and My_Rk_Nm > 0 and Mz_Rk_Nm > 0 and (Ncr_y > 0) and (Ncr_z > 0):
        # --- Method 1 (Annex A inspired) ---
        Cmy0 = 0.79 + 0.21 * psi_y + 0.36 * (psi_y - 0.33) * (abs(N_N) / Ncr_y) if Ncr_y > 0 else 1.0
        Cmz0 = 0.79 + 0.21 * psi_z + 0.36 * (psi_z - 0.33) * (abs(N_N) / Ncr_z) if Ncr_z > 0 else 1.0

        npl = abs(N_N) / (NRk_N / gamma_M0) if (NRk_N > 0) else 0.0

        # epsilon_y (use elastic modulus if available)
        eps_y = None
        if Wel_y_m3 > 0 and abs(N_N) > 1e-9:
            eps_y = (abs(My_Ed_kNm) * 1e3 / Wel_y_m3) / (abs(N_N) / A_m2) if A_m2 > 0 else None
        elif Wel_y_m3 > 0:
            eps_y = 1e9
        else:
            eps_y = 1.0

        wy = min(1.5, (Wpl_y_m3 / Wel_y_m3)) if (Wpl_y_m3 > 0 and Wel_y_m3 > 0) else 1.0
        wz = min(1.5, (Wpl_z_m3 / Wel_z_m3)) if (Wpl_z_m3 > 0 and Wel_z_m3 > 0) else 1.0

        mu_y = (1 - abs(N_N)/Ncr_y) / (1 - chi_y*abs(N_N)/Ncr_y) if (Ncr_y > 0 and (1 - chi_y*abs(N_N)/Ncr_y) != 0) else 1.0
        mu_z = (1 - abs(N_N)/Ncr_z) / (1 - chi_z*abs(N_N)/Ncr_z) if (Ncr_z > 0 and (1 - chi_z*abs(N_N)/Ncr_z) != 0) else 1.0

        aLT = max(0.0, 1.0 - (J_m4 / I_y_m4)) if (I_y_m4 > 0 and J_m4 > 0) else 0.0

        # Apply simple Cmy correction (as in your text)
        if eps_y is None:
            eps_y = 1.0
        Cmy = Cmy0 + (1.0 - Cmy0) * (math.sqrt(max(eps_y, 0.0)) * aLT) / (1.0 + math.sqrt(max(eps_y, 0.0)) * aLT)
        Cmz = Cmz0

        denom_CM = math.sqrt(max((1 - abs(N_N)/Ncr_z) * (1 - abs(N_N)/max(Ncr_T_use, 1e-9)), 1e-9))
        CmLT = max(1.0, (Cmy**2) * aLT / denom_CM)

        # Table A.1 constants (class 1/2 I-sections, common values)
        Cyy, Cyz, Czy, Czz = 0.973, 0.657, 0.939, 0.968

        kyy = Cmy * CmLT * mu_y / max((1 - abs(N_N)/Ncr_y), 1e-9) * (1.0 / Cyy)
        kyz = Cmz * mu_y / max((1 - abs(N_N)/Ncr_z), 1e-9) * (1.0 / Cyz) * 0.6 * math.sqrt(max(wz / max(wy, 1e-9), 0.0))
        kzy = Cmy * CmLT * mu_z / max((1 - abs(N_N)/Ncr_y), 1e-9) * (1.0 / Czy) * 0.6 * math.sqrt(max(wy / max(wz, 1e-9), 0.0))
        kzz = Cmz * mu_z / max((1 - abs(N_N)/Ncr_z), 1e-9) * (1.0 / Czz)

        # Utilizations (6.61/6.62 simplified)
        Ny = abs(N_N) / (chi_y * NRk_N / gamma_M1) if (chi_y * NRk_N) > 0 else float("inf")
        Nz = abs(N_N) / (chi_z * NRk_N / gamma_M1) if (chi_z * NRk_N) > 0 else float("inf")

        My_term = kyy * (abs(My_Ed_kNm) * 1e3) / (chiLT * My_Rk_Nm / gamma_M1) if (chiLT * My_Rk_Nm) > 0 else float("inf")
        Mz_term = kyz * (abs(Mz_Ed_kNm) * 1e3) / (Mz_Rk_Nm / gamma_M1) if Mz_Rk_Nm > 0 else float("inf")
        util_61 = Ny + My_term + Mz_term

        My_term2 = kzy * (abs(My_Ed_kNm) * 1e3) / (chiLT * My_Rk_Nm / gamma_M1) if (chiLT * My_Rk_Nm) > 0 else float("inf")
        Mz_term2 = kzz * (abs(Mz_Ed_kNm) * 1e3) / (Mz_Rk_Nm / gamma_M1) if Mz_Rk_Nm > 0 else float("inf")
        util_62 = Nz + My_term2 + Mz_term2

        util_int_A = max(util_61, util_62)


        # Store Method 1 (Annex A) intermediate values for the Report tab
        buck_map.update({
            "Cmy0_A": Cmy0,
            "Cmz0_A": Cmz0,
            "npl_A": npl,
            "eps_y_A": eps_y,
            "wy_A": wy,
            "wz_A": wz,
            "mu_y_A": mu_y,
            "mu_z_A": mu_z,
            "aLT_A": aLT,
            "Cmy_A": Cmy,
            "Cmz_A": Cmz,
            "CmLT_A": CmLT,
            "kyy_A": kyy,
            "kyz_A": kyz,
            "kzy_A": kzy,
            "kzz_A": kzz,
            "util_61_A": util_61,
            "util_62_A": util_62,
            "util_int_A": util_int_A,
            "util_int_A_y": util_61,
            "util_int_A_z": util_62,
            "status_int_A_y": "OK" if util_61 <= 1.0 else "EXCEEDS",
            "status_int_A_z": "OK" if util_62 <= 1.0 else "EXCEEDS",
        })

        # (19) / (20) — Method 1 (Annex A): two interaction expressions
        rows.append({
            "Check": "Buckling interaction (Method 1, Annex A) — Eq. (y)",
            "Applied": "Interaction",
            "Resistance": "≤ 1.0",
            "Utilization": f"{util_61:.3f}",
            "Status": "OK" if util_61 <= 1.0 else "EXCEEDS",
        })
        rows.append({
            "Check": "Buckling interaction (Method 1, Annex A) — Eq. (z)",
            "Applied": "Interaction",
            "Resistance": "≤ 1.0",
            "Utilization": f"{util_62:.3f}",
            "Status": "OK" if util_62 <= 1.0 else "EXCEEDS",
        })

        # --- Method 2 (Annex B inspired) ---
        Cmy_B = max(0.4, 0.60 + 0.40 * psi_y)
        Cmz_B = max(0.4, 0.60 + 0.40 * psi_z)
        CmLT_B = max(0.4, 0.60 + 0.40 * psi_LT)

        kyy_B = Cmy_B * (1.0 + (min(lam_y, 1.0) - 0.2) * abs(N_N) / (chi_y * NRk_N / gamma_M1)) if (chi_y * NRk_N) > 0 else 1.0
        kzz_B = Cmz_B * (1.0 + (2.0 * min(lam_z, 1.0) - 0.6) * abs(N_N) / (chi_z * NRk_N / gamma_M1)) if (chi_z * NRk_N) > 0 else 1.0
        kyz_B = 0.6 * kzz_B

        if lam_z >= 0.4:
            kzy_B = 1.0 - 0.1 * min(lam_z, 1.0) / max((CmLT_B - 0.25), 1e-9) * abs(N_N) / (chi_z * NRk_N / gamma_M1)
        else:
            kzy_B = 1.0

        util_61_B = Ny + kyy_B * (abs(My_Ed_kNm) * 1e3) / (chiLT * My_Rk_Nm / gamma_M1) + kyz_B * (abs(Mz_Ed_kNm) * 1e3) / (Mz_Rk_Nm / gamma_M1)
        util_62_B = Nz + kzy_B * (abs(My_Ed_kNm) * 1e3) / (chiLT * My_Rk_Nm / gamma_M1) + kzz_B * (abs(Mz_Ed_kNm) * 1e3) / (Mz_Rk_Nm / gamma_M1)

        util_int_B = max(util_61_B, util_62_B)


        # Store Method 2 (Annex B) intermediate values for the Report tab
        buck_map.update({
            "Cmy_B": Cmy_B,
            "Cmz_B": Cmz_B,
            "CmLT_B": CmLT_B,
            "kyy_B": kyy_B,
            "kzz_B": kzz_B,
            "kyz_B": kyz_B,
            "kzy_B": kzy_B,
            "util_61_B": util_61_B,
            "util_62_B": util_62_B,
            "util_int_B": util_int_B,
            "util_int_B_y": util_61_B,
            "util_int_B_z": util_62_B,
            "status_int_B_y": "OK" if util_61_B <= 1.0 else "EXCEEDS",
            "status_int_B_z": "OK" if util_62_B <= 1.0 else "EXCEEDS",
        })

        # (21) / (22) — Method 2 (Annex B): two interaction expressions
        rows.append({
            "Check": "Buckling interaction (Method 2, Annex B) — Eq. (y)",
            "Applied": "Interaction",
            "Resistance": "≤ 1.0",
            "Utilization": f"{util_61_B:.3f}",
            "Status": "OK" if util_61_B <= 1.0 else "EXCEEDS",
        })
        rows.append({
            "Check": "Buckling interaction (Method 2, Annex B) — Eq. (z)",
            "Applied": "Interaction",
            "Resistance": "≤ 1.0",
            "Utilization": f"{util_62_B:.3f}",
            "Status": "OK" if util_62_B <= 1.0 else "EXCEEDS",
        })

        buck_map.update({
            "Cmy0": Cmy0, "Cmz0": Cmz0, "Cmy": Cmy, "Cmz": Cmz, "CmLT": CmLT,
            "kyy": kyy, "kyz": kyz, "kzy": kzy, "kzz": kzz,
            "util_int_method1": util_int_A,
            "util_int_method2": util_int_B,
            "status_int_method1": "OK" if util_int_A <= 1.0 else "EXCEEDS",
            "status_int_method2": "OK" if util_int_B <= 1.0 else "EXCEEDS",
        })

    # Store detailed buckling numbers for report
    
    extras_buck_map = buck_map

    # Build DataFrame and summary
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
        buck_map=extras_buck_map,
        cs_combo=cs_combo,
        N_Rd_N=N_Rd_N,
        M_Rd_y_Nm=M_Rd_y_Nm,
        V_Rd_N=V_Rd_N,
    )
    return df_rows, overall_ok, governing, extras

def render_results(df_rows, overall_ok, governing,
                   show_footer=True,
                   include_deflection=True,
                   key_prefix="res_"):
    """
    Results summary (can be used in Results tab and Report tab).
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
    # Deflection summary (serviceability)
    # -------------------------------------------------
    if include_deflection:
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

            # keys depend on context (Results vs Report)
            k_delta = f"{key_prefix}delta_max_mm"
            k_L300  = f"{key_prefix}L300_mm"
            k_L600  = f"{key_prefix}L600_mm"
            k_L900  = f"{key_prefix}L900_mm"

            # update state so inputs show latest values
            st.session_state[k_delta] = val_delta
            st.session_state[k_L300]  = val_L300
            st.session_state[k_L600]  = val_L600
            st.session_state[k_L900]  = val_L900

            small_title("Deflection (serviceability)")

            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.text_input("δ_max [mm]", value=val_delta, key=k_delta, disabled=True)
            with d2:
                st.text_input("Limit L/300 [mm]", value=val_L300, key=k_L300, disabled=True)
            with d3:
                st.text_input("Limit L/600 [mm]", value=val_L600, key=k_L600, disabled=True)
            with d4:
                st.text_input("Limit L/900 [mm]", value=val_L900, key=k_L900, disabled=True)

        else:
            st.caption(
                "Deflection summary (δ_max, L/300, L/600, L/900) will appear here "
                "after running the diagrams in the Loads tab."
            )

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
        "Flexural buckling y–y",                                              # 15
        "Flexural buckling z–z",                                              # 16
        "Torsional / torsional-flexural buckling z",                          # 17
        "Lateral-torsional buckling",                                         # 18
        "Buckling interaction (Method 1, Annex A) — Eq. (y)",                 # 19
        "Buckling interaction (Method 1, Annex A) — Eq. (z)",                 # 20
        "Buckling interaction (Method 2, Annex B) — Eq. (y)",                 # 21
        "Buckling interaction (Method 2, Annex B) — Eq. (z)",                 # 22
    ]

    # For now: placeholders (you'll later replace with real values)
    cs_util = ["" for _ in cs_checks]
    cs_status = ["" for _ in cs_checks]
    buck_util = ["" for _ in buck_checks]
    buck_status = ["" for _ in buck_checks]
    # -----------------------------
    # -----------------------------
    # Map detailed check results into the summary table
    # -----------------------------
    def fill_cs_from_df(idx_out, must_contain, must_not_contain=None):
        """
        Find the first df_rows row whose "Check" text contains all strings in must_contain
        and none of the strings in must_not_contain.

        NOTE: We normalize the text (lowercase + remove non-alphanumerics except '+')
        so that labels like "Tension (N>0)" still match queries like ["tension","n"].
        """
        if df_rows is None:
            return
        if must_not_contain is None:
            must_not_contain = []

        def _norm(x: str) -> str:
            x = (x or "").lower()
            # keep "+" because we use it as a filter; strip everything else non-alnum
            return re.sub(r"[^a-z0-9+]+", "", x)

        must_cont = [_norm(s) for s in must_contain]
        must_not = [_norm(s) for s in must_not_contain]

        for _idx, row in df_rows.iterrows():
            s_raw = str(row.get("Check", ""))
            if not s_raw and hasattr(row, "name"):
                s_raw = str(row.name)
            if not s_raw:
                s_raw = str(_idx)
            s = _norm(s_raw)

            if all(mc in s for mc in must_cont) and all(mn not in s for mn in must_not):
                cs_util[idx_out] = row.get("Utilization", "")
                cs_status[idx_out] = row.get("Status", "")
                break

        # Fallbacks for the first 6 checks (some apps use short labels like "My", "Vz", etc.)
        if (cs_util[idx_out] == "" or cs_util[idx_out] is None) and idx_out in (0,1,2,3,4,5):
            for _idx, row in df_rows.iterrows():
                s_raw = str(row.get("Check", ""))
                if not s_raw and hasattr(row, "name"):
                    s_raw = str(row.name)
                if not s_raw:
                    s_raw = str(_idx)
                s = _norm(s_raw)
                # Map by common Eurocode wording / your check titles
                if idx_out == 0 and ("tension" in s):
                    cs_util[idx_out] = row.get("Utilization", "")
                    cs_status[idx_out] = row.get("Status", "")
                    break
                if idx_out == 1 and ("compression" in s) and ("buckling" not in s):
                    cs_util[idx_out] = row.get("Utilization", "")
                    cs_status[idx_out] = row.get("Status", "")
                    break
                if idx_out == 2 and ("my" in s) and ("+" not in s) and ("vy" not in s) and ("vz" not in s):
                    cs_util[idx_out] = row.get("Utilization", "")
                    cs_status[idx_out] = row.get("Status", "")
                    break
                if idx_out == 3 and ("mz" in s) and ("+" not in s) and ("vy" not in s) and ("vz" not in s):
                    cs_util[idx_out] = row.get("Utilization", "")
                    cs_status[idx_out] = row.get("Status", "")
                    break
                if idx_out == 4 and ("vy" in s) and ("+" not in s) and ("shear" in s):
                    cs_util[idx_out] = row.get("Utilization", "")
                    cs_status[idx_out] = row.get("Status", "")
                    break
                if idx_out == 5 and ("vz" in s) and ("+" not in s) and ("shear" in s):
                    cs_util[idx_out] = row.get("Utilization", "")
                    cs_status[idx_out] = row.get("Status", "")
                    break

        # ---- Fallback for checks 1–6 (robust) ----
        # If matching fails for any reason (label variations), map directly using the first occurrences
        # in df_rows by looking for key words.
        if df_rows is not None:
            def _pick_first(predicate):
                for __i, __r in df_rows.iterrows():
                    __s = str(__r.get("Check","")).lower()
                    if predicate(__s):
                        return __r
                return None

            # 1) tension
            if cs_util[0] == "" and cs_status[0] == "":
                r = _pick_first(lambda s: "tension" in s)
                if r is not None:
                    cs_util[0] = r.get("Utilization","")
                    cs_status[0] = r.get("Status","")

            # 2) compression
            if cs_util[1] == "" and cs_status[1] == "":
                r = _pick_first(lambda s: "compression" in s)
                if r is not None:
                    cs_util[1] = r.get("Utilization","")
                    cs_status[1] = r.get("Status","")

            # 3) My (pure bending)
            if cs_util[2] == "" and cs_status[2] == "":
                r = _pick_first(lambda s: ("my" in s) and ("+" not in s) and ("vy" not in s) and ("vz" not in s))
                if r is not None:
                    cs_util[2] = r.get("Utilization","")
                    cs_status[2] = r.get("Status","")

            # 4) Mz (pure bending)
            if cs_util[3] == "" and cs_status[3] == "":
                r = _pick_first(lambda s: ("mz" in s) and ("+" not in s) and ("vy" not in s) and ("vz" not in s))
                if r is not None:
                    cs_util[3] = r.get("Utilization","")
                    cs_status[3] = r.get("Status","")

            # 5) Vy (pure shear)
            if cs_util[4] == "" and cs_status[4] == "":
                r = _pick_first(lambda s: ("vy" in s) and ("+" not in s))
                if r is not None:
                    cs_util[4] = r.get("Utilization","")
                    cs_status[4] = r.get("Status","")

            # 6) Vz (pure shear)
            if cs_util[5] == "" and cs_status[5] == "":
                r = _pick_first(lambda s: ("vz" in s) and ("+" not in s))
                if r is not None:
                    cs_util[5] = r.get("Utilization","")
                    cs_status[5] = r.get("Status","")

    # 1) N (tension)
    fill_cs_from_df(
        idx_out=0,
        must_contain=["Tension", "N"],
    )

    # 2) N (compression)
    fill_cs_from_df(
        idx_out=1,
        must_contain=["Compression", "N"],
    )

    # 3) My  – pure bending My only (no +, no shear)
    fill_cs_from_df(
        idx_out=2,
        must_contain=["My"],
        must_not_contain=["+", "Vy", "Vz"],
    )
    
    # 4) Mz  – pure bending Mz only (no +, no shear)
    fill_cs_from_df(
        idx_out=3,
        must_contain=["Mz"],
        must_not_contain=["+", "Vy", "Vz"],
    )

    # 5) Vy – pure shear in y direction
    fill_cs_from_df(
        idx_out=4,
        must_contain=["Vy"],
        must_not_contain=["+"],
    )
    
    # 6) Vz – pure shear in z direction
    fill_cs_from_df(
        idx_out=5,
        must_contain=["Vz"],
        must_not_contain=["+"],
    )
# Helper to build one nice looking table

    # -------------------------------------------------
    # Fill combined checks (7–14) using stored combo flags
    # -------------------------------------------------
    extras = st.session_state.get("extras") or {}
    cs_combo = extras.get("cs_combo") or {}

    def _fmt_util(x):
        if x is None:
            return ""
        try:
            return f"{float(x):.3f}"
        except Exception:
            return str(x)

    def _ok(u):
        try:
            return "OK" if float(u) <= 1.0 else "EXCEEDS"
        except Exception:
            return ""

    util_My = cs_combo.get("util_My", None)
    util_Mz = cs_combo.get("util_Mz", None)

    # (7) My + Vy : if Vy <= 0.5 Vpl,Rd,y => same as My
    if cs_combo.get("shear_ratio_y", None) is not None and cs_combo.get("shear_ok_y", False):
        cs_util[6] = _fmt_util(util_My)
        cs_status[6] = _ok(util_My)
    else:
        cs_util[6] = _fmt_util(util_My)
        cs_status[6] = _ok(util_My)

    # (8) Mz + Vz : if Vz <= 0.5 Vpl,Rd,z => same as Mz
    if cs_combo.get("shear_ratio_z", None) is not None and cs_combo.get("shear_ok_z", False):
        cs_util[7] = _fmt_util(util_Mz)
        cs_status[7] = _ok(util_Mz)
    else:
        cs_util[7] = _fmt_util(util_Mz)
        cs_status[7] = _ok(util_Mz)

    # (9)-(11) Bending + axial force : if axial criteria met => same as bending
    if cs_combo.get("axial_ok_y", False):
        cs_util[8] = _fmt_util(util_My)
        cs_status[8] = _ok(util_My)
    else:
        cs_util[8] = _fmt_util(util_My)
        cs_status[8] = _ok(util_My)

    if cs_combo.get("axial_ok_z", False):
        cs_util[9] = _fmt_util(util_Mz)
        cs_status[9] = _ok(util_Mz)
    else:
        cs_util[9] = _fmt_util(util_Mz)
        cs_status[9] = _ok(util_Mz)

    cs_util[10] = _fmt_util(max([u for u in [util_My, util_Mz] if u is not None], default=None))
    cs_status[10] = _ok(max([u for u in [util_My, util_Mz] if u is not None], default=0.0))

    # (12)-(14) Bending + shear + axial : if shear <=0.5 => same as (9)-(11)
    cs_util[11] = cs_util[8]
    cs_status[11] = cs_status[8]
    cs_util[12] = cs_util[9]
    cs_status[12] = cs_status[9]
    cs_util[13] = cs_util[10]
    cs_status[13] = cs_status[10]

    # -------------------------------------------------
    # Fill member stability checks (15–20) from buckling results in df_rows / extras
    # -------------------------------------------------
    buck_map = extras.get("buck_map") or {}

    # 15/16
    buck_util[0] = _fmt_util(buck_map.get("util_buck_y", None))
    buck_status[0] = buck_map.get("status_buck_y", "") or ""
    buck_util[1] = _fmt_util(buck_map.get("util_buck_z", None))
    buck_status[1] = buck_map.get("status_buck_z", "") or ""

    # 17
    buck_util[2] = _fmt_util(buck_map.get("util_T", None))
    buck_status[2] = buck_map.get("status_T", "") or ""

    # 18
    buck_util[3] = _fmt_util(buck_map.get("util_LT", None))
    buck_status[3] = buck_map.get("status_LT", "") or ""

    # 19–22
    buck_util[4] = _fmt_util(buck_map.get("util_int_A_y", None))
    buck_status[4] = buck_map.get("status_int_A_y", "") or ""
    buck_util[5] = _fmt_util(buck_map.get("util_int_A_z", None))
    buck_status[5] = buck_map.get("status_int_A_z", "") or ""
    buck_util[6] = _fmt_util(buck_map.get("util_int_B_y", None))
    buck_status[6] = buck_map.get("status_int_B_y", "") or ""
    buck_util[7] = _fmt_util(buck_map.get("util_int_B_z", None))
    buck_status[7] = buck_map.get("status_int_B_z", "") or ""
    # -------------------------------------------------
    # Helper    # -------------------------------------------------
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

        for i, name in enumerate(names):
            util = utils[i] if i < len(utils) else ""
            status = statuses[i] if i < len(statuses) else ""
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
        "Verification of member stability (buckling, checks 15–22)",
        15,
        buck_checks,
        buck_util,
        buck_status,
    )
    st.markdown(buck_html, unsafe_allow_html=True)

    if not show_footer:
        return

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
    sr_display = st.session_state.get("sr_display")

    # Alias: section properties used throughout the report
    use_props = sr_display or {}
    inputs = st.session_state.get("inputs")
    df_rows = st.session_state.get("df_rows")
    overall_ok = st.session_state.get("overall_ok", True)
    governing = st.session_state.get("governing", (None, None))
    extras = st.session_state.get("extras") or {}
    meta = st.session_state.get("meta")
    material = st.session_state.get("material", "S355")
    # Safety factors (use session state defaults)
    gamma_M0 = float(st.session_state.get("gamma_M0", 1.00))
    gamma_M1 = float(st.session_state.get("gamma_M1", 1.00))

    # always rebuild project meta from Project tab widgets
    doc_name     = st.session_state.get("doc_title_in",   "Beam check")
    project_name = st.session_state.get("project_name_in","")
    position     = st.session_state.get("position_in",    "")
    requested_by = st.session_state.get("requested_by_in","")
    revision     = st.session_state.get("revision_in",    "A")
    run_date     = st.session_state.get("run_date_in",    date.today())
    notes        = st.session_state.get("notes_in",       "")

    meta = (doc_name, project_name, position, requested_by, revision, run_date, notes)
    st.session_state["meta"] = meta

    sigma_allow   = extras.get("sigma_allow_MPa")
    sigma_eq      = extras.get("sigma_eq_MPa")
    buck_results  = extras.get("buck_results", [])

    if sr_display is None or inputs is None or df_rows is None:
        st.info("To see the report: select a section, define loads, run the check, then return here.")
        return
    # from extras (computed in compute_checks)
    sigma_allow = extras.get("sigma_allow_MPa")
    sigma_eq = extras.get("sigma_eq_MPa")
    buck_results = extras.get("buck_results", [])

    if sr_display is None or inputs is None or df_rows is None or meta is None:
        st.info("To see the report: select a section, define loads, run the check, then return here.")
        return

    # Design internal forces for report equations (same as Loads tab)
    N_kN = inputs.get("N_kN", 0.0)
    Vy_Ed_kN = inputs.get("Vy_kN", 0.0)
    Vz_Ed_kN = inputs.get("Vz_kN", 0.0)
    My_Ed_kNm = inputs.get("My_kNm", 0.0)
    Mz_Ed_kNm = inputs.get("Mz_kNm", 0.0)

    fam = sr_display.get("family", "")
    name = sr_display.get("name", "")
    fy = material_to_fy(material)
    gov_check, gov_util = governing
    status_txt = "OK" if overall_ok else "NOT OK"

    # 🔽🔽🔽 PUT THE SECTION-CLASS / Wpl–Wel BLOCK HERE 🔽🔽🔽
    # Plastic + elastic section moduli from DB (values in cm³)
    Wpl_y_cm3 = sr_display.get("Wpl_y_cm3", 0.0)
    Wpl_z_cm3 = sr_display.get("Wpl_z_cm3", 0.0)
    Wel_y_cm3 = sr_display.get("Wel_y_cm3", 0.0)
    Wel_z_cm3 = sr_display.get("Wel_z_cm3", 0.0)

    # Section class (1, 2, 3, 4)
    section_class = sr_display.get("class", 1)

    # Select correct modulus based on class:
    #   Class 1–2 → plastic modulus Wpl
    #   Class 3   → elastic modulus Wel
    if section_class in (1, 2):
        W_y_cm3 = Wpl_y_cm3
        W_z_cm3 = Wpl_z_cm3
        W_text = "plastic"
    else:
        W_y_cm3 = Wel_y_cm3
        W_z_cm3 = Wel_z_cm3
        W_text = "elastic"

    # Convert to mm³ for equations (1 cm³ = 1000 mm³)
    W_y_mm3 = W_y_cm3 * 1e3
    W_z_mm3 = W_z_cm3 * 1e3
    # Aliases for older code that still uses Wpl_y_mm3 / Wpl_z_mm3
    Wpl_y_mm3 = W_y_mm3
    Wpl_z_mm3 = W_z_mm3
    
    # --- Shear areas for shear resistance (from DB, mm²) ---
    Av_z_mm2 = float(sr_display.get("Avz_mm2") or sr_display.get("Av_z_mm2") or 0.0)
    Av_y_mm2 = float(sr_display.get("Avy_mm2") or sr_display.get("Av_y_mm2") or 0.0)
    
    # 🔼🔼🔼 END OF BLOCK 🔼🔼🔼

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
    report_h4("Save report")
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
    # 1. Project data (from Project tab)
    # ----------------------------------------------------
    c1, c2, c3 = st.columns([1, 1, 1])
    
    with c1:
        st.text_input("Document title", value=str(doc_name), disabled=True)
        st.text_input("Project name", value=str(project_name), disabled=True)
    
    with c2:
        st.text_input("Position / Location (Beam ID)", value=str(position), disabled=True)
        st.text_input("Requested by", value=str(requested_by), disabled=True)
    
    with c3:
        st.text_input("Revision", value=str(revision), disabled=True)
        st.text_input("Date", value=str(run_date), disabled=True)
    
    # Notes: keep editable in Report tab (optional but usually useful)
    if "rpt_notes" not in st.session_state:
        st.session_state["rpt_notes"] = str(notes)
    
    st.text_area("Notes / comments", value=str(notes), disabled=True, height=120)
    
    st.markdown("---")

    # ----------------------------------------------------
    # 2. Material design values (EN 1993-1-1)
    # ----------------------------------------------------
    report_h3("2. Material design values (EN 1993-1-1)")

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
    report_h3("3. Member & section data")

    # 3.1 Member inputs
    report_h4("3.1 Member definition")

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
    report_h4("3.2 Section data")

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
    report_h3("4. Section classification (EN 1993-1-1 §5.5)")

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
    report_h3("5. Applied actions & internal forces (ULS)")

    if ready_case:
        st.write(f"Load case type: **{ready_case.get('key','')} — {ready_case.get('label','')}**")
    else:
        st.write("Load case type: **User-defined**")

    # --- Row 1: N, Vy, Vz ---
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        st.text_input("N_Ed [kN]", f"{inputs.get('N_kN',0.0):.3f}", disabled=True)
    with r1c2:
        st.text_input("Vy_Ed [kN]", f"{inputs.get('Vy_kN',0.0):.3f}", disabled=True)
    with r1c3:
        st.text_input("Vz_Ed [kN]", f"{inputs.get('Vz_kN',0.0):.3f}", disabled=True)

    # --- Row 2: My, Mz ---
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        st.text_input("My_Ed [kNm]", f"{inputs.get('My_kNm',0.0):.3f}", disabled=True)
    with r2c2:
        st.text_input("Mz_Ed [kNm]", f"{inputs.get('Mz_kNm',0.0):.3f}", disabled=True)
    with r2c3:
        st.empty()   # clean layout (no empty grey box)

    # ----------------------------------------------------
    # Result summary (same tables as in Results tab)
    # ----------------------------------------------------
    report_h3("6. Result summary (ULS, buckling & deflection)")
    render_results(
        df_rows,
        overall_ok,
        governing,
        show_footer=False,      # no bottom hints in Report
        include_deflection=True,
        key_prefix="rep_",      # different keys than Results tab
    )
    # Status from the checks table (row 'Tension (N≥0)')
    status_ten = "n/a"
    
    # ----------------------------------------------------
    # 6. Detailed calculations
    # ----------------------------------------------------
    with st.expander("Detailed calculations", expanded=False):
        report_h3("6. Detailed calculations")
        report_h4("6.1 Verification of cross-section strength (ULS, checks 1–14)")
        # 6.1.a Detailed explanation for check (1) Tension
        report_h4("(1) Tension – EN 1993-1-1 §6.2.3")

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
        report_h4("(2) Compression – EN 1993-1-1 §6.2.4")

        # design compressive axial force (take magnitude of N < 0)
        if N_kN < 0.0:
            NEd_comp_kN = -N_kN
        else:
            NEd_comp_kN = 0.0

        # compression resistance (class 1–3): Nc,Rd = A fy / γM0
        Nc_Rd_kN = Npl_Rd_kN  # same expression for gross section

        # utilisation u = NEd / Nc,Rd from our own numbers
        if Nc_Rd_kN > 0.0:
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
            # ---- Bending resistances for report (using selected Wpl/Wel) ----
        if Wpl_y_mm3 > 0 and fy > 0:
            Mc_y_Rd_kNm = (Wpl_y_mm3 * fy / gamma_M0) / 1e6  # mm³*MPa → Nmm → kNm
        else:
            Mc_y_Rd_kNm = 0.0

        if Wpl_z_mm3 > 0 and fy > 0:
            Mc_z_Rd_kNm = (Wpl_z_mm3 * fy / gamma_M0) / 1e6
        else:
            Mc_z_Rd_kNm = 0.0

        # Utilizations in bending for the equations
        if Mc_y_Rd_kNm > 0:
            util_My = My_Ed_kNm / Mc_y_Rd_kNm
            status_My = "OK" if util_My <= 1.0 else "EXCEEDS"
        else:
            util_My = 0.0
            status_My = "n/a"

        if Mc_z_Rd_kNm > 0:
            util_Mz = Mz_Ed_kNm / Mc_z_Rd_kNm
            status_Mz = "OK" if util_Mz <= 1.0 else "EXCEEDS"
        else:
            util_Mz = 0.0
            status_Mz = "n/a"
    
        report_h4("(3), (4) Bending moment resistance (EN 1993-1-1 §6.2.5)")

        st.markdown("The design bending resistance is checked using:")
        st.latex(r"\frac{M_{Ed}}{M_{c,Rd}} \le 1.0")

        st.markdown(
            f"For a **Class {section_class}** cross-section the "
            f"**{W_text} section modulus** is used in accordance with EN 1993-1-1 §6.2.5:"
        )
        st.latex(r"M_{c,Rd} = W \, \frac{f_y}{\gamma_{M0}}")

        # Show computed resistances with the selected modulus (Wpl or Wel)
        st.latex(
            rf"M_{{c,y,Rd}} = W_{{y}} \frac{{f_y}}{{\gamma_{{M0}}}}"
            rf" = {W_y_mm3:.0f} \; mm^3 \cdot {fy:.0f} \, MPa / {gamma_M0}"
            rf" = {Mc_y_Rd_kNm:.2f} \; kNm"
        )

        st.latex(
            rf"M_{{c,z,Rd}} = W_{{z}} \frac{{f_y}}{{\gamma_{{M0}}}}"
            rf" = {W_z_mm3:.0f} \; mm^3 \cdot {fy:.0f} \, MPa / {gamma_M0}"
            rf" = {Mc_z_Rd_kNm:.2f} \; kNm"
        )

        # Utilization results
        report_h4("Utilization for bending resistance")

        st.latex(
            rf"u_y = \frac{{M_{{y,Ed}}}}{{M_{{c,y,Rd}}}}"
            rf" = \frac{{{My_Ed_kNm:.2f}}}{{{Mc_y_Rd_kNm:.2f}}}"
            rf" = {util_My:.3f} \Rightarrow \textbf{{{status_My}}}"
        )

        st.latex(
            rf"u_z = \frac{{M_{{z,Ed}}}}{{M_{{c,z,Rd}}}}"
            rf" = \frac{{{Mz_Ed_kNm:.2f}}}{{{Mc_z_Rd_kNm:.2f}}}"
            rf" = {util_Mz:.3f} \Rightarrow \textbf{{{status_Mz}}}"
        )

        st.markdown(
            """
    According to EN 1993-1-1 §6.2.5(4–6), holes may be neglected in bending resistance
    provided the tensile and compression areas satisfy Eq. (6.16) and the holes in compression
    zones are filled with fasteners.
    """
        )

            # ---- Shear resistances and utilizations for report ----
        # Design shear forces from inputs
        Vz_Ed_kN = Vz_Ed_kN  # already set at top, just documenting
        Vy_Ed_kN = Vy_Ed_kN

        # Shear resistances
        if Av_z_mm2 > 0 and fy > 0:
            Vc_z_Rd_kN = (Av_z_mm2 * (fy / math.sqrt(3)) / gamma_M0) / 1000.0
        else:
            Vc_z_Rd_kN = 0.0

        if Av_y_mm2 > 0 and fy > 0:
            Vc_y_Rd_kN = (Av_y_mm2 * (fy / math.sqrt(3)) / gamma_M0) / 1000.0
        else:
            Vc_y_Rd_kN = 0.0

        # Utilizations
        if Vc_z_Rd_kN > 0:
            util_Vz = Vz_Ed_kN / Vc_z_Rd_kN
            status_Vz = "OK" if util_Vz <= 1.0 else "EXCEEDS"
        else:
            util_Vz = 0.0
            status_Vz = "n/a"

        if Vc_y_Rd_kN > 0:
            util_Vy = Vy_Ed_kN / Vc_y_Rd_kN
            status_Vy = "OK" if util_Vy <= 1.0 else "EXCEEDS"
        else:
            util_Vy = 0.0
            status_Vy = "n/a"

        report_h4("(5), (6) Shear resistance (EN 1993-1-1 §6.2.6)")

        st.markdown("The shear resistance check uses:")
        st.latex(r"\frac{V_{Ed}}{V_{c,Rd}} \le 1.0")

        st.markdown("Plastic shear resistance:")
        st.latex(r"V_{c,Rd} = A_v \, \frac{f_y}{\sqrt{3}\,\gamma_{M0}}")
    
        # Shear resistances
        st.latex(
            rf"V_{{c,z,Rd}} = A_{{v,z}} \frac{{f_y}}{{\sqrt 3 \gamma_{{M0}}}}"
            rf" = {Av_z_mm2:.0f} \, mm^2 \cdot ({fy:.0f} / \sqrt 3) / {gamma_M0}"
            rf" = {Vc_z_Rd_kN:.2f} \, kN"
        )
    
        st.latex(
            rf"V_{{c,y,Rd}} = A_{{v,y}} \frac{{f_y}}{{\sqrt 3 \gamma_{{M0}}}}"
            rf" = {Av_y_mm2:.0f} \, mm^2 \cdot ({fy:.0f} / \sqrt 3) / {gamma_M0}"
            rf" = {Vc_y_Rd_kN:.2f} \, kN"
        )
    
        # Utilization
        report_h4("Utilization checks")
    
        st.latex(
            rf"u_z = \frac{{V_{{z,Ed}}}}{{V_{{c,z,Rd}}}}"
            rf" = \frac{{{Vz_Ed_kN:.2f}}}{{{Vc_z_Rd_kN:.2f}}}"
            rf" = {util_Vz:.3f} \Rightarrow \textbf{{{status_Vz}}}"
        )
    
        st.latex(
            rf"u_y = \frac{{V_{{y,Ed}}}}{{V_{{c,y,Rd}}}}"
            rf" = \frac{{{Vy_Ed_kN:.2f}}}{{{Vc_y_Rd_kN:.2f}}}"
            rf" = {util_Vy:.3f} \Rightarrow \textbf{{{status_Vy}}}"
        )
    
        st.markdown("""
        Per EN 1993-1-1 §6.2.6(7), shear resistance does not need to account for fastener holes
        except at joints where EN 1993-1-8 applies.
        """)
    

        # ----------------------------------------------------
        # (6.2.7) Torsion
        # ----------------------------------------------------
        st.markdown("**Torsion (EN 1993-1-1 §6.2.7)**")
        st.markdown("""
    The verifications for torsional moment **T<sub>Ed</sub>** are **not examined** in this calculation.

    For open cross-sections without any directly applied torsional load the torsional moment occurs primarily as **warping torsion** due to:
    1. rotation compatibility due to bending of other transversely connected members, and  
    2. eccentricity of loading applied directly on the examined member.

    For this typical case the effects of warping torsion are small and can be ignored. However, if the steel member directly supports significant torsional loads then a **closed cross-section is recommended**.
    """, unsafe_allow_html=True)

        # ----------------------------------------------------
        # (7),(8) Bending and shear (EN 1993-1-1 §6.2.8)
        # ----------------------------------------------------
        st.markdown("**(7), (8) Bending and shear (EN 1993-1-1 §6.2.8)**")
        
        st.markdown("""
        Shear can influence the available bending resistance of a cross-section.
        In line with **EN 1993-1-1 §6.2.8**, a reduction is only relevant when the applied shear force
        exceeds **50%** of the corresponding plastic shear resistance.
        """)
        
        cs_combo = (extras.get("cs_combo") or {})
        shear_ratio_z = cs_combo.get("shear_ratio_z", None)
        shear_ratio_y = cs_combo.get("shear_ratio_y", None)
        
        st.markdown("For the examined case:")
        
        def _line(label_html: str, latex_expr: str):
            c_label, c_eq, c_right = st.columns([3, 4, 3])
            with c_label:
                st.markdown(label_html, unsafe_allow_html=True)
            with c_eq:
                st.latex(latex_expr)

        
        # --- z-z ---
        if Vc_z_Rd_kN and Vc_z_Rd_kN > 0:
            _line(
                "<u>Shear force along axis z-z:</u>",
                rf"\frac{{V_{{z,Ed}}}}{{V_{{pl,Rd,z}}}}"
                rf"=\frac{{{Vz_Ed_kN:.1f}\,\mathrm{{kN}}}}{{{Vc_z_Rd_kN:.1f}\,\mathrm{{kN}}}}"
                rf"={(Vz_Ed_kN / Vc_z_Rd_kN):.3f}\le 0.50"
            )
        else:
            _line("<u>Shear force along axis z-z:</u>", r"\text{Not available (missing }V_{pl,Rd,z}\text{)}")
        
        # --- y-y ---
        if Vc_y_Rd_kN and Vc_y_Rd_kN > 0:
            _line(
                "<u>Shear force along axis y-y:</u>",
                rf"\frac{{V_{{y,Ed}}}}{{V_{{pl,Rd,y}}}}"
                rf"=\frac{{{Vy_Ed_kN:.1f}\,\mathrm{{kN}}}}{{{Vc_y_Rd_kN:.1f}\,\mathrm{{kN}}}}"
                rf"={(Vy_Ed_kN / Vc_y_Rd_kN):.3f}\le 0.50"
            )
        else:
            _line("<u>Shear force along axis y-y:</u>", r"\text{Not available (missing }V_{pl,Rd,y}\text{)}")
        
        # --- conclusion (use cs_combo flags, as you already computed them in compute_checks) ---
        if cs_combo.get("shear_ok_y", False) and cs_combo.get("shear_ok_z", False):
            st.markdown("""
        Both shear ratios are below **0.50**, so shear does not govern the bending resistance here.
        Therefore, the bending utilization factors remain the same as in Sections **(3)** and **(4)**.
        """)
        else:
            st.markdown("""
        At least one shear ratio is above **0.50**. In that case, bending resistance may need to be reduced
        according to **EN 1993-1-1 §6.2.8**.
        """)
        # ----------------------------------------------------
        # (6.2.9) Bending and axial force
        # ----------------------------------------------------
        report_h4("(9), (10), (11) Bending and axial force (EN 1993-1-1 §6.2.9)")
        
        # Helpers: centered equation column (matches your new shear layout)
        def _eq_line(label_html: str, latex_expr: str):
            cL, cM, cR = st.columns([3, 4, 3])
            with cL:
                st.markdown(label_html, unsafe_allow_html=True)
            with cM:
                st.latex(latex_expr)
        
        # Inputs / section resistances (from DB)
        family = (sr_display.get("family") or "").upper()
        A_mm2  = float(sr_display.get("A_mm2") or 0.0)
        h_mm   = float(sr_display.get("h_mm") or 0.0)
        b_mm   = float(sr_display.get("b_mm") or 0.0)
        tw_mm  = float(sr_display.get("tw_mm") or 0.0)
        tf_mm  = float(sr_display.get("tf_mm") or 0.0)
        
        Npl_Rd_kN   = float(sr_display.get("Npl_Rd_kN") or 0.0)
        Mpl_y_Rd_kNm = float(sr_display.get("Mpl_Rd_y_kNm") or 0.0)
        Mpl_z_Rd_kNm = float(sr_display.get("Mpl_Rd_z_kNm") or 0.0)
        
        NEd_kN = float(cs_combo.get("NEd_kN") or inputs.get("N_kN") or 0.0)
        My_Ed_kNm = float(inputs.get("My_kNm") or 0.0)
        Mz_Ed_kNm = float(inputs.get("Mz_kNm") or 0.0)
        
        st.markdown(
            "This section evaluates the influence of axial force on the bending resistance "
            "for Class 1–2 cross-sections in accordance with **EN 1993-1-1 §6.2.9**."
        )
        
        # Normalized axial force
        n = (abs(NEd_kN) / Npl_Rd_kN) if Npl_Rd_kN > 0 else None
        if n is not None:
            _eq_line("Design axial force:", rf"N_{{Ed}} = {NEd_kN:.2f}\,\mathrm{{kN}}")
            _eq_line("Normalized axial force:", rf"n=\frac{{N_{{Ed}}}}{{N_{{pl,Rd}}}}=\frac{{{abs(NEd_kN):.2f}}}{{{Npl_Rd_kN:.2f}}}={n:.3f}")
        
        # Decide “shape family” (practical classification)
        is_rhs = any(k in family for k in ["RHS", "SHS", "HSS", "BOX"])
        is_ih  = any(k in family for k in ["IPE", "HE", "HEA", "HEB", "HEM", "IPN", "UB", "UC", "W", "I", "H"])
        
        st.markdown("For the examined case:")
        
        # --------------------------
        # A) Doubly-symmetric I / H
        # --------------------------
        if is_ih and (Npl_Rd_kN > 0) and (fy > 0) and (gamma_M0 > 0):
        
            # Web height approximation (kept consistent with your existing approach)
            hw_mm = max(h_mm - 2.0 * tf_mm, 0.0)
        
            # “May ignore” criteria (as you already computed in cs_combo)
            crit_y_25  = cs_combo.get("crit_y_25", None)
            crit_y_web = cs_combo.get("crit_y_web", None)
            crit_z_web = cs_combo.get("crit_z_web", None)
        
            # Show the limit values (if available)
            if crit_y_25 is not None:
                _eq_line("Major axis criterion 1:", rf"N_{{Ed}}\le 0.25\,N_{{pl,Rd}} = {crit_y_25:.1f}\,\mathrm{{kN}}")
            if crit_y_web is not None:
                _eq_line("Major axis criterion 2:", rf"N_{{Ed}}\le 0.50\,h_w t_w f_y/\gamma_{{M0}} = {crit_y_web:.1f}\,\mathrm{{kN}}")
            if crit_z_web is not None:
                _eq_line("Minor axis criterion:", rf"N_{{Ed}}\le h_w t_w f_y/\gamma_{{M0}} = {crit_z_web:.1f}\,\mathrm{{kN}}")
        
            if cs_combo.get("axial_ok_y", False) and cs_combo.get("axial_ok_z", False):
                st.markdown(
                    "The axial force is sufficiently small to neglect its influence on the plastic bending resistance "
                    "for the current I/H section. Therefore the bending utilizations remain as in **(3)** and **(4)**."
                )
            else:
                # Apply I/H reduction rules (EN 1993-1-1 §6.2.9.1(5), formulas 8.48–8.50)
                if n is None:
                    st.warning("Cannot evaluate interaction because Npl,Rd is not available.")
                else:
                    a = min(0.5, (A_mm2 - 2.0 * b_mm * tf_mm) / A_mm2) if A_mm2 > 0 else 0.5
        
                    # Reduced bending resistances
                    MN_y_Rd = min(Mpl_y_Rd_kNm, Mpl_y_Rd_kNm * (1.0 - n) / (1.0 - 0.5 * a)) if (Mpl_y_Rd_kNm > 0) else 0.0
        
                    if n <= a:
                        MN_z_Rd = Mpl_z_Rd_kNm
                    else:
                        MN_z_Rd = Mpl_z_Rd_kNm * (1.0 - ((n - a) / (1.0 - a)) ** 2) if (Mpl_z_Rd_kNm > 0) else 0.0
        
                    _eq_line("Section ratio:", rf"a=\min\left[0.5,\frac{{A-2b_f t_f}}{{A}}\right]={a:.3f}")
                    _eq_line("Reduced resistance (y):", rf"M_{{N,y,Rd}} = {MN_y_Rd:.3f}\,\mathrm{{kNm}}")
                    _eq_line("Reduced resistance (z):", rf"M_{{N,z,Rd}} = {MN_z_Rd:.3f}\,\mathrm{{kNm}}")
        
                    # Biaxial interaction (EN 1993-1-1 §6.2.9.1(6), formula 8.56)
                    alpha_y = 2.0
                    alpha_z = max(1.0, 5.0 * n)
        
                    uy = (abs(My_Ed_kNm) / MN_y_Rd) if MN_y_Rd > 0 else None
                    uz = (abs(Mz_Ed_kNm) / MN_z_Rd) if MN_z_Rd > 0 else None
        
                    if (uy is not None) and (uz is not None):
                        u_biax = (uy ** alpha_y) + (uz ** alpha_z)
                        _eq_line("Interaction:", rf"\left(\frac{{M_{{y,Ed}}}}{{M_{{N,y,Rd}}}}\right)^{{{alpha_y:.2f}}}+\left(\frac{{M_{{z,Ed}}}}{{M_{{N,z,Rd}}}}\right)^{{{alpha_z:.2f}}}={u_biax:.3f}\le 1.0")
                        st.markdown("Result: **OK**" if u_biax <= 1.0 else "Result: **NOT OK**")
        
        # --------------------------
        # B) Rectangular hollow (RHS/SHS/HSS)
        # --------------------------
        elif is_rhs and (Npl_Rd_kN > 0) and (A_mm2 > 0):
        
            if n is None:
                st.warning("Cannot evaluate interaction because Npl,Rd is not available.")
            else:
                # aw, af (uniform thickness assumption: use tf for flange thickness, tw for web thickness)
                aw = min(0.5, (A_mm2 - 2.0 * b_mm * tf_mm) / A_mm2) if (A_mm2 > 0) else 0.5
                af = min(0.5, (A_mm2 - 2.0 * h_mm * tw_mm) / A_mm2) if (A_mm2 > 0) else 0.5
        
                _eq_line("Ratio (web part):",  rf"a_w=\min\left[0.5,\frac{{A-2b\,t_f}}{{A}}\right]={aw:.3f}")
                _eq_line("Ratio (flange part):", rf"a_f=\min\left[0.5,\frac{{A-2h\,t_w}}{{A}}\right]={af:.3f}")
        
                # Reduced bending resistances (EN 1993-1-1 §6.2.9.1(6), formulas 8.51–8.52)
                MN_y_Rd = min(Mpl_y_Rd_kNm, Mpl_y_Rd_kNm * (1.0 - n) / (1.0 - 0.5 * aw)) if (Mpl_y_Rd_kNm > 0) else 0.0
                MN_z_Rd = min(Mpl_z_Rd_kNm, Mpl_z_Rd_kNm * (1.0 - n) / (1.0 - 0.5 * af)) if (Mpl_z_Rd_kNm > 0) else 0.0
        
                _eq_line("Reduced resistance (y):", rf"M_{{N,y,Rd}} = {MN_y_Rd:.3f}\,\mathrm{{kNm}}")
                _eq_line("Reduced resistance (z):", rf"M_{{N,z,Rd}} = {MN_z_Rd:.3f}\,\mathrm{{kNm}}")
        
                uy = (abs(My_Ed_kNm) / MN_y_Rd) if MN_y_Rd > 0 else None
                uz = (abs(Mz_Ed_kNm) / MN_z_Rd) if MN_z_Rd > 0 else None
        
                if (uy is not None) and (uz is not None):
                    # Exponents for RHS (from EN 1993-1-1 guidance)
                    if n <= 0.8:
                        alpha = min(6.0, 1.66 / (1.0 - 1.13 * (n ** 2)))
                    else:
                        alpha = 6.0
                    beta = alpha
        
                    u_biax = (uy ** alpha) + (uz ** beta)
        
                    _eq_line("Exponents:", rf"\alpha=\beta={alpha:.3f}")
                    _eq_line("Interaction:", rf"\left(\frac{{M_{{y,Ed}}}}{{M_{{N,y,Rd}}}}\right)^{{{alpha:.3f}}}+\left(\frac{{M_{{z,Ed}}}}{{M_{{N,z,Rd}}}}\right)^{{{beta:.3f}}}={u_biax:.3f}\le 1.0")
        
                    st.markdown("Result: **OK**" if u_biax <= 1.0 else "Result: **NOT OK**")
        
        # --------------------------
        # C) Fallback (unknown family)
        # --------------------------
        else:
            st.info(
                "Section family is not recognized as I/H or rectangular hollow. "
                "For now, no special axial–bending interaction model is applied here."
            )

        # ----------------------------------------------------
        # (6.2.10) Combined bending, shear and axial force
        # ----------------------------------------------------
        report_h4("(12), (13), (14) Bending, shear and axial force (EN 1993-1-1 §6.2.10)")
        
        st.markdown(
            "This section considers the combined influence of **shear** and **axial force** on bending resistance "
            "in line with **EN 1993-1-1 §6.2.10**. If the applied shear exceeds **0.5·Vpl,Rd**, a reduction is applied."
        )
        
        # Use your already computed shear ratios (from §6.2.8 section)
        cs_combo = (extras.get("cs_combo") or {})
        shear_ratio_z = cs_combo.get("shear_ratio_z", None)  # Vz,Ed / Vpl,Rd,z
        shear_ratio_y = cs_combo.get("shear_ratio_y", None)  # Vy,Ed / Vpl,Rd,y
        
        # Simple centered equation helper (same style you use in §6.2.8)
        def _eq_line(label_html: str, latex_expr: str):
            cL, cM, cR = st.columns([3, 4, 3])
            with cL:
                st.markdown(label_html, unsafe_allow_html=True)
            with cM:
                st.latex(latex_expr)
        
        st.markdown("For the examined case:")
        
        # Show shear ratios (from earlier section)
        if (shear_ratio_z is not None) and (Vc_z_Rd_kN > 0):
            _eq_line(
                "<u>Axis z-z:</u>",
                rf"\frac{{V_{{z,Ed}}}}{{V_{{pl,Rd,z}}}}"
                rf"=\frac{{{Vz_Ed_kN:.1f}\,\mathrm{{kN}}}}{{{Vc_z_Rd_kN:.1f}\,\mathrm{{kN}}}}"
                rf"={shear_ratio_z:.3f}"
            )
        
        if (shear_ratio_y is not None) and (Vc_y_Rd_kN > 0):
            _eq_line(
                "<u>Axis y-y:</u>",
                rf"\frac{{V_{{y,Ed}}}}{{V_{{pl,Rd,y}}}}"
                rf"=\frac{{{Vy_Ed_kN:.1f}\,\mathrm{{kN}}}}{{{Vc_y_Rd_kN:.1f}\,\mathrm{{kN}}}}"
                rf"={shear_ratio_y:.3f}"
            )
        
        # Case check: can we ignore shear effect?
        ok_z = (shear_ratio_z is not None) and (shear_ratio_z <= 0.50)
        ok_y = (shear_ratio_y is not None) and (shear_ratio_y <= 0.50)
        
        if ok_z and ok_y:
            st.markdown(
                "Both shear ratios are **≤ 0.50**, therefore shear does not reduce the bending+axial resistances here. "
                "The interaction check remains as in **(9)–(11)**."
            )
        else:
            st.markdown(
                "At least one shear ratio is **> 0.50**, therefore reduction is applied using the factor "
                r"$\rho=(2V_{Ed}/V_{pl,Rd}-1)^2$."
            )
        
            # ρ factors per axis (only where ratio > 0.5). Clamp to [0,1] for numerical safety.
            rho_z = 0.0
            rho_y = 0.0
        
            if (shear_ratio_z is not None) and (shear_ratio_z > 0.50) and (Vc_z_Rd_kN > 0):
                rho_z = (2.0 * Vz_Ed_kN / Vc_z_Rd_kN - 1.0) ** 2
                rho_z = max(0.0, min(1.0, rho_z))
                _eq_line("Reduction factor (z-z):", rf"\rho_z=\left(2\frac{{V_{{z,Ed}}}}{{V_{{pl,Rd,z}}}}-1\right)^2={rho_z:.3f}")
        
            if (shear_ratio_y is not None) and (shear_ratio_y > 0.50) and (Vc_y_Rd_kN > 0):
                rho_y = (2.0 * Vy_Ed_kN / Vc_y_Rd_kN - 1.0) ** 2
                rho_y = max(0.0, min(1.0, rho_y))
                _eq_line("Reduction factor (y-y):", rf"\rho_y=\left(2\frac{{V_{{y,Ed}}}}{{V_{{pl,Rd,y}}}}-1\right)^2={rho_y:.3f}")
        
            # Reduction multipliers
            k_z = 1.0 - rho_z
            k_y = 1.0 - rho_y
        
            _eq_line("Shear reduction multiplier (z-z):", rf"k_z=1-\rho_z={k_z:.3f}")
            _eq_line("Shear reduction multiplier (y-y):", rf"k_y=1-\rho_y={k_y:.3f}")
        
            # ---- IMPORTANT NOTE (temporary model) ----
            st.caption(
                "Note: until reduced section properties (A′, Wpl′, etc.) are stored in the database, "
                "a conservative resistance reduction is applied by scaling resistances with k. "
                "When you add reduced properties, we will replace this scaling with exact A′/W′ based values."
            )
        
            # Conservative scaling for resistances (temporary)
            # axial uses the whole section -> use the worst reduction
            k_ax = min(k_y, k_z)
        
            Npl_Rd_kN_red   = Npl_Rd_kN * k_ax
            Mpl_y_Rd_kNm_red = Mpl_y_Rd_kNm * k_y  # major axis mostly flange-related -> use k_y
            Mpl_z_Rd_kNm_red = Mpl_z_Rd_kNm * k_ax # conservative for minor axis
        
            _eq_line("Reduced axial resistance:", rf"N_{{V,pl,Rd}} = {Npl_Rd_kN_red:.1f}\,\mathrm{{kN}}")
            _eq_line("Reduced plastic moment (y):", rf"M_{{V,pl,y,Rd}} = {Mpl_y_Rd_kNm_red:.2f}\,\mathrm{{kNm}}")
            _eq_line("Reduced plastic moment (z):", rf"M_{{V,pl,z,Rd}} = {Mpl_z_Rd_kNm_red:.2f}\,\mathrm{{kNm}}")
        
            # Now repeat the §6.2.9 interaction using the reduced resistances
            NEd_kN = float(inputs.get("N_kN", 0.0))
            My_Ed_kNm = float(inputs.get("My_kNm", 0.0))
            Mz_Ed_kNm = float(inputs.get("Mz_kNm", 0.0))
        
            nV = (abs(NEd_kN) / Npl_Rd_kN_red) if (Npl_Rd_kN_red > 0) else None
            if nV is not None:
                _eq_line("Normalized axial force (reduced):", rf"n=\frac{{N_{{Ed}}}}{{N_{{V,pl,Rd}}}}=\frac{{{abs(NEd_kN):.2f}}}{{{Npl_Rd_kN_red:.2f}}}={nV:.3f}")
        
            uy = (abs(My_Ed_kNm) / Mpl_y_Rd_kNm_red) if (Mpl_y_Rd_kNm_red > 0) else None
            uz = (abs(Mz_Ed_kNm) / Mpl_z_Rd_kNm_red) if (Mpl_z_Rd_kNm_red > 0) else None
        
            if (uy is not None) and (uz is not None):
                # Use same exponent rules as your §6.2.9 implementation:
                # For I/H: alpha_y=2, alpha_z=max(1,5n). If you already detect family elsewhere, reuse that.
                alpha_y = 2.0
                alpha_z = max(1.0, 5.0 * nV) if nV is not None else 1.0
        
                u_biax = (uy ** alpha_y) + (uz ** alpha_z)
        
                _eq_line(
                    "Biaxial interaction (reduced):",
                    rf"\left(\frac{{M_{{y,Ed}}}}{{M_{{V,pl,y,Rd}}}}\right)^{{{alpha_y:.2f}}}"
                    rf"+\left(\frac{{M_{{z,Ed}}}}{{M_{{V,pl,z,Rd}}}}\right)^{{{alpha_z:.2f}}}"
                    rf"={u_biax:.3f}\le 1.0"
                )
        
                st.markdown("Result: **OK**" if u_biax <= 1.0 else "Result: **NOT OK**")

        # ----------------------------------------------------
        # (6.3) Member stability summary (checks 15–22)
        # ----------------------------------------------------
            # ----------------------------------------------------
        # (6.3) Verification of member stability (buckling, checks 15–22)
        # ----------------------------------------------------
        buck_map = (extras.get("buck_map") or {})

        # Basic inputs
        L = float(inputs.get("L", 0.0))
        K_y = float(inputs.get("K_y", 1.0))
        K_z = float(inputs.get("K_z", 1.0))
        K_T = float(inputs.get("K_T", 1.0))
        K_LT = float(inputs.get("K_LT", 1.0))

        NEd_kN = float(inputs.get("N_kN", 0.0))
        MyEd_kNm = float(inputs.get("My_kNm", 0.0))
        MzEd_kNm = float(inputs.get("Mz_kNm", 0.0))

        # Section & material
        A_mm2 = float(use_props.get("A_mm2", 0.0))
        Iy_mm4 = float(use_props.get("Iy_mm4", use_props.get("Iy_cm4", 0.0) * 1e4))
        Iz_mm4 = float(use_props.get("Iz_mm4", use_props.get("Iz_cm4", 0.0) * 1e4))
        iy_mm = float(use_props.get("iy_mm", 0.0))
        iz_mm = float(use_props.get("iz_mm", 0.0))
        It_mm4 = float(use_props.get("It_mm4", use_props.get("It_cm4", 0.0) * 1e4))
        Iw_mm6 = float(use_props.get("Iw_mm6", use_props.get("Iw_cm6", 0.0) * 1e6))
        Wel_y_mm3 = float(use_props.get("Wel_y_mm3", use_props.get("Wel_y_cm3", 0.0) * 1e3))
        Wpl_y_mm3 = float(use_props.get("Wpl_y_mm3", use_props.get("Wpl_y_cm3", 0.0) * 1e3))
        Wel_z_mm3 = float(use_props.get("Wel_z_mm3", use_props.get("Wel_z_cm3", 0.0) * 1e3))
        Wpl_z_mm3 = float(use_props.get("Wpl_z_mm3", use_props.get("Wpl_z_cm3", 0.0) * 1e3))

        # Factors
        # (gamma_M0, gamma_M1 already defined at top of render_report_tab)

        # Buckling curve letters inferred from alpha (if available)
        def _curve_from_alpha(a):
            if a is None:
                return "n/a"
            a = float(a)
            if abs(a - 0.21) < 1e-3: return "a"
            if abs(a - 0.34) < 1e-3: return "b"
            if abs(a - 0.49) < 1e-3: return "c"
            if abs(a - 0.76) < 1e-3: return "d"
            return "n/a"

        alpha_y = float(buck_map.get("alpha_y", st.session_state.get("alpha_y", 0.21)))
        alpha_z = float(buck_map.get("alpha_z", st.session_state.get("alpha_z", 0.34)))

        curve_y = _curve_from_alpha(alpha_y)
        curve_z = _curve_from_alpha(alpha_z)

        # Extract buckling results
        Ncr_y = buck_map.get("Ncr_y")
        Ncr_z = buck_map.get("Ncr_z")
        lam_y = buck_map.get("lambda_y")
        lam_z = buck_map.get("lambda_z")
        chi_y = buck_map.get("chi_y")
        chi_z = buck_map.get("chi_z")
        Nb_Rd_y = buck_map.get("Nb_Rd_y")
        Nb_Rd_z = buck_map.get("Nb_Rd_z")
        util_y = buck_map.get("util_y")
        util_z = buck_map.get("util_z")

        i0_m = buck_map.get("i0_m")
        Ncr_T = buck_map.get("Ncr_T")
        chi_T = buck_map.get("chi_T")
        Nb_Rd_T = buck_map.get("Nb_Rd_T")
        util_T = buck_map.get("util_T")

        Mcr = buck_map.get("Mcr")
        lam_LT = buck_map.get("lambda_LT")
        chi_LT = buck_map.get("chi_LT")
        Mb_Rd = buck_map.get("Mb_Rd")
        util_LT = buck_map.get("util_LT")

        # Interaction (methods)
        util_int_A = buck_map.get("util_int_A")
        util_int_B = buck_map.get("util_int_B")

        # ----------------------------
        # (15),(16) Flexural buckling
        # ----------------------------
    
        report_h4("6.2 Verification of member stability (buckling, checks 15–22)")
        report_h4("(15), (16) Flexural buckling – EN 1993-1-1 §6.3.1.3")
        st.markdown(f"""
    The compression member is verified against flexural buckling in accordance with EN1993-1-1 §6.3.1 as follows:

    NEd / Nb,Rd ≤ 1.0

    where Nb,Rd is the design buckling resistance of the compression member given in EN1993-1-1 §6.3.1.1(3) for class 1, 2 and 3 cross-sections:

    Nb,Rd = χ⋅A⋅fy / γM1

    The reduction factor χ due to flexural buckling is calculated for the major and the minor bending axes.

    Flexural buckling about major axis y-y  
    The appropriate buckling curve is determined from EN1993-1-1 Table 6.2. The corresponding buckling curve for the selected member is taken as curve "{curve_y}".

    The imperfection factor α corresponding to the buckling curve "{curve_y}" is determined from EN1993-1-1 Table 6.1 as α = {alpha_y:.2f}.

    The critical buckling length Lcr,y for flexural buckling about the major axis y-y is considered as Lcr,y = {K_y:.3f}⋅L = {K_y:.3f}⋅{L:.3f} m = {K_y*L:.3f} m.

    According to the theory of elasticity the elastic critical buckling load for flexural buckling is:

    Ncr,y = π2⋅E⋅Iy / Lcr,y2 = π2⋅210000 MPa⋅{Iy_mm4:,.0f} mm4 / ({K_y*L:.3f} m)2 = {(Ncr_y/1e3 if Ncr_y else float('nan')):.1f} kN

    The ratio of the compression load to the elastic critical buckling load is NEd/Ncr,y = {abs(NEd_kN):.1f} kN / {(Ncr_y/1e3 if Ncr_y else float('nan')):.1f} kN = {(abs(NEd_kN)/(Ncr_y/1e3) if Ncr_y else float('nan')):.3f}

    For class 1, 2 and 3 cross-section the non-dimensional slenderness λy for flexural buckling is given in EN1993-1-1 §6.3.1.3(1):

    λy = (A⋅fy / Ncr,y)0.5 = ({A_mm2:,.0f} mm2⋅{fy:.0f} MPa / {(Ncr_y/1e3 if Ncr_y else float('nan')):.1f} kN)0.5 = {(lam_y if lam_y is not None else float('nan')):.3f}

    According to EN1993-1-1 §6.3.1.2(4) flexural buckling effects may be ignored when NEd/Ncr,y ≤ 0.04 or λy ≤ 0.20.

    The factors Φ and χy are calculated in accordance with EN1993-1-1 §6.3.1.2:

    Φ = 0.5⋅[1 + α⋅(λy - 0.20) + λy2]  
    χy = min[1.0, 1 / (Φ + [Φ2 - λy2]0.5)]

    The design buckling resistance of the compression member for flexural buckling about the major axis y-y is calculated as:

    Nb,Rd,y = χy ⋅ A ⋅ fy / γM1 = {(chi_y if chi_y is not None else float('nan')):.3f} ⋅ {A_mm2:,.0f} mm2 ⋅ {fy:.0f} MPa / {gamma_M1:.2f} = {(Nb_Rd_y/1e3 if Nb_Rd_y else float('nan')):.1f} kN

    Therefore the utilization for the flexural buckling resistance about major axis y-y is:

    u = NEd / Nb,Rd,y = {abs(NEd_kN):.1f} kN / {(Nb_Rd_y/1e3 if Nb_Rd_y else float('nan')):.1f} kN = {(util_y if util_y is not None else float('nan')):.3f} ≤ 1.0 ⇒ {"ok" if (util_y is not None and util_y <= 1.0) else "exceeds"}

    Flexural buckling about minor axis z-z  
    The appropriate buckling curve is determined from EN1993-1-1 Table 6.2. The corresponding buckling curve for the selected member is taken as curve "{curve_z}".

    The imperfection factor α corresponding to the buckling curve "{curve_z}" is determined from EN1993-1-1 Table 6.1 as α = {alpha_z:.2f}.

    The critical buckling length Lcr,z for flexural buckling about the minor axis z-z is considered as Lcr,z = {K_z:.3f}⋅L = {K_z:.3f}⋅{L:.3f} m = {K_z*L:.3f} m.

    According to the theory of elasticity the elastic critical buckling load for flexural buckling is:

    Ncr,z = π2⋅E⋅Iz / Lcr,z2 = π2⋅210000 MPa⋅{Iz_mm4:,.0f} mm4 / ({K_z*L:.3f} m)2 = {(Ncr_z/1e3 if Ncr_z else float('nan')):.1f} kN

    The ratio of the compression load to the elastic critical buckling load is NEd/Ncr,z = {abs(NEd_kN):.1f} kN / {(Ncr_z/1e3 if Ncr_z else float('nan')):.1f} kN = {(abs(NEd_kN)/(Ncr_z/1e3) if Ncr_z else float('nan')):.3f}

    For class 1, 2 and 3 cross-section the non-dimensional slenderness λz for flexural buckling is given in EN1993-1-1 §6.3.1.3(1):

    λz = (A⋅fy / Ncr,z)0.5 = ({A_mm2:,.0f} mm2⋅{fy:.0f} MPa / {(Ncr_z/1e3 if Ncr_z else float('nan')):.1f} kN)0.5 = {(lam_z if lam_z is not None else float('nan')):.3f}

    The factors Φ and χz are calculated in accordance with EN1993-1-1 §6.3.1.2:

    Φ = 0.5⋅[1 + α⋅(λz - 0.20) + λz2]  
    χz = min[1.0, 1 / (Φ + [Φ2 - λz2]0.5)]

    The design buckling resistance of the compression member for flexural buckling about the minor axis z-z is calculated as:

    Nb,Rd,z = χz ⋅ A ⋅ fy / γM1 = {(chi_z if chi_z is not None else float('nan')):.3f} ⋅ {A_mm2:,.0f} mm2 ⋅ {fy:.0f} MPa / {gamma_M1:.2f} = {(Nb_Rd_z/1e3 if Nb_Rd_z else float('nan')):.1f} kN

    Therefore the utilization for the flexural buckling resistance about minor axis z-z is:

    u = NEd / Nb,Rd,z = {abs(NEd_kN):.1f} kN / {(Nb_Rd_z/1e3 if Nb_Rd_z else float('nan')):.1f} kN = {(util_z if util_z is not None else float('nan')):.3f} ≤ 1.0 ⇒ {"ok" if (util_z is not None and util_z <= 1.0) else "exceeds"}

    According to EN1993-1-1 §6.3.1.1(4) the calculated flexural buckling resistance is also valid for members with holes for fasteners at the member ends.""")

        # ----------------------------
        # (17) Torsional & torsional-flexural buckling
        # ----------------------------
        report_h4("(17) Torsional and torsional-flexural buckling – EN 1993-1-1 §6.3.1.4")
        st.markdown(f"""
    Typically for standard I- and H-sections the torsional and torsional-flexural buckling verifications are not critical as compared to the flexural buckling verification. For completeness of the calculation the torsional and torsional-flexural buckling loads are estimated below.

    The polar radius of gyration of the cross-section i0 is equal to:

    i0 = [iy2 + iz2 + y02 + z02]0.5

    For doubly symmetrical cross-sections the shear center and the centroid coincide, therefore y0 = 0 and z0 = 0 and:

    i0 = [iy2 + iz2]0.5 = [({iy_mm:.1f} mm)2 + ({iz_mm:.1f} mm)2]0.5 = {(i0_m*1e3 if i0_m else float('nan')):.1f} mm

    The critical buckling length Lcr,T for torsional buckling is considered as Lcr,T = {K_T:.3f}⋅L = {K_T*L:.3f} m.

    The elastic critical force Ncr,T for torsional buckling is estimated as:

    Ncr,T = (1 / i02)⋅(G⋅IT + π2⋅E⋅Iw / Lcr,T2) = {(Ncr_T/1e3 if Ncr_T else float('nan')):.1f} kN

    The design buckling resistance is:

    Nb,Rd,T = χ ⋅ A ⋅ fy / γM1 = {(Nb_Rd_T/1e3 if Nb_Rd_T else float('nan')):.1f} kN

    Therefore the utilization is:

    u = NEd / Nb,Rd,T = {(util_T if util_T is not None else float('nan')):.3f} ≤ 1.0 ⇒ {"ok" if (util_T is not None and util_T <= 1.0) else "exceeds"}""")

        # ----------------------------
        # (18) Lateral-torsional buckling
        # ----------------------------
        report_h4("(18) Lateral-torsional buckling – EN 1993-1-1 §6.3.2")
        st.markdown(f"""
    Members with laterally unrestrained compression flange subject to bending about major axis y-y should be verified against lateral-torsional buckling in accordance with EN1993-1-1 §6.3.2 as follows:

    My,Ed / Mb,Rd ≤ 1.0

    where Mb,Rd is the design buckling resistance moment:

    Mb,Rd = χLT⋅Wy⋅fy / γM1

    For class 1 or 2 cross-sections: Wy = Wpl,y = {Wpl_y_mm3:,.0f} mm3

    Elastic critical moment (uniform moment, k=kw=1, zg=0) is calculated as:

    Mcr = {(Mcr/1e3 if Mcr else float('nan')):.1f} kNm

    The non-dimensional slenderness is:

    λLT = (Wy⋅fy / Mcr)0.5 = {(lam_LT if lam_LT is not None else float('nan')):.3f}

    The reduction factor is:

    χLT = {(chi_LT if chi_LT is not None else float('nan')):.3f}

    The design buckling resistance moment is:

    Mb,Rd = {(Mb_Rd/1e3 if Mb_Rd else float('nan')):.1f} kNm

    Therefore the utilization for the lateral-torsional buckling resistance is:

    u = My,Ed / Mb,Rd = {abs(MyEd_kNm):.1f} kNm / {(Mb_Rd/1e3 if Mb_Rd else float('nan')):.1f} kNm = {(util_LT if util_LT is not None else float('nan')):.3f} ≤ 1.0 ⇒ {"ok" if (util_LT is not None and util_LT <= 1.0) else "exceeds"}
    """)

        # ----------------------------

        # ----------------------------
        # (19)–(22) Buckling interaction for bending and axial compression
        # ----------------------------
        util_61_A = buck_map.get("util_61_A")
        util_62_A = buck_map.get("util_62_A")
        util_61_B = buck_map.get("util_61_B")
        util_62_B = buck_map.get("util_62_B")

        # ---- Method 1 (Annex A) ----
        report_h4("(19),(20) Buckling interaction for bending and axial compression - Method 1 (EN1993-1-1 Annex A)")

        st.markdown("Equivalent uniform moment factors for flexural buckling.")
        st.markdown("The equivalent uniform moment factors **Cmi,0** are obtained from **EN 1993-1-1 Table A.2**.")

        st.markdown("- For flexural buckling about major axis **y–y**, the moment diagram **My,Ed** is considered between points braced along **z–z** direction.")
        st.markdown("For uniform or linearly varying bending moment diagram:")
        st.latex(rf"C_{{my,0}} = 0.79 + 0.21\,\psi_y + 0.36\,(\psi_y-0.33)\,\frac{{N_{{Ed}}}}{{N_{{cr,y}}}} = {buck_map.get('Cmy0_A', float('nan')):.3f}")

        st.markdown("- For flexural buckling about minor axis **z–z**, the moment diagram **Mz,Ed** is considered between points braced along **y–y** direction.")
        st.markdown("For uniform or linearly varying bending moment diagram:")
        st.latex(rf"C_{{mz,0}} = 0.79 + 0.21\,\psi_z + 0.36\,(\psi_z-0.33)\,\frac{{N_{{Ed}}}}{{N_{{cr,z}}}} = {buck_map.get('Cmz0_A', float('nan')):.3f}")

        st.markdown("Intermediate factors and coefficients:")
        st.markdown(textwrap.dedent(f"""        - Normalized axial force: **npl = {buck_map.get('npl_A', float('nan')):.3f}**
        - λLT (from Section 18): **{buck_map.get('lambdaLT_A', float('nan')):.3f}**
        - λmax = max(λy, λz): **{buck_map.get('lambda_max_A', float('nan')):.3f}**
        - εy: **{buck_map.get('eps_y_A', float('nan')):.3f}**
        - wy: **{buck_map.get('wy_A', float('nan')):.3f}**, wz: **{buck_map.get('wz_A', float('nan')):.3f}**
        """))

        st.markdown("Interaction factors (EN 1993-1-1 Table A.1):")
        st.markdown(textwrap.dedent(f"""        - kyy = **{buck_map.get('kyy_A', float('nan')):.3f}**
        - kyz = **{buck_map.get('kyz_A', float('nan')):.3f}**
        - kzy = **{buck_map.get('kzy_A', float('nan')):.3f}**
        - kzz = **{buck_map.get('kzz_A', float('nan')):.3f}**
        """))


        st.markdown(f"""Verification of member resistance — Equation (y)  
    u = NEd/(χy⋅NRk/γM1) + kyy⋅My,Ed/(χLT⋅My,Rk/γM1) + kyz⋅Mz,Ed/(Mz,Rk/γM1)  
    u = **{(util_61_A if util_61_A is not None else float('nan')):.3f}** {'≤ 1.0 ⇒ OK' if (util_61_A is not None and util_61_A<=1.0) else '> 1.0 ⇒ NOT OK'}""")
        st.markdown(f"""Verification of member resistance — Equation (z)  
    u = NEd/(χz⋅NRk/γM1) + kzy⋅My,Ed/(χLT⋅My,Rk/γM1) + kzz⋅Mz,Ed/(Mz,Rk/γM1)  
    u = **{(util_62_A if util_62_A is not None else float('nan')):.3f}** {'≤ 1.0 ⇒ OK' if (util_62_A is not None and util_62_A<=1.0) else '> 1.0 ⇒ NOT OK'}""")

        # ---- Method 2 (Annex B) ----
        report_h4("(21),(22) Buckling interaction for bending and axial compression - Method 2 (EN1993-1-1 Annex B)")

        st.markdown("Equivalent uniform moment factors for flexural buckling.")
        st.markdown("The equivalent uniform moment factors **Cmi** are obtained from **EN 1993-1-1 Table B.3**.")

        st.latex(rf"C_{{my}} = \max(0.4,\,0.60 + 0.40\,\psi_y) = {buck_map.get('Cmy_B', float('nan')):.3f}")
        st.latex(rf"C_{{mz}} = \max(0.4,\,0.60 + 0.40\,\psi_z) = {buck_map.get('Cmz_B', float('nan')):.3f}")
        st.latex(rf"C_{{mLT}} = \max(0.4,\,0.60 + 0.40\,\psi_{{LT}}) = {buck_map.get('CmLT_B', float('nan')):.3f}")

        st.markdown("Interaction factors (EN 1993-1-1 Annex B):")
        st.markdown(textwrap.dedent(f"""        - kyy = **{buck_map.get('kyy_B', float('nan')):.3f}**, kyz = **{buck_map.get('kyz_B', float('nan')):.3f}**
        - kzy = **{buck_map.get('kzy_B', float('nan')):.3f}**, kzz = **{buck_map.get('kzz_B', float('nan')):.3f}**
        """))


        st.markdown(f"""Verification of member resistance — Equation (y)  
    u = NEd/(χy⋅NRk/γM1) + kyy⋅My,Ed/(χLT⋅My,Rk/γM1) + kyz⋅Mz,Ed/(Mz,Rk/γM1)  
    u = **{(util_61_B if util_61_B is not None else float('nan')):.3f}** {'≤ 1.0 ⇒ OK' if (util_61_B is not None and util_61_B<=1.0) else '> 1.0 ⇒ NOT OK'}""")
        st.markdown(f"""Verification of member resistance — Equation (z)  
    u = NEd/(χz⋅NRk/γM1) + kzy⋅My,Ed/(χLT⋅My,Rk/γM1) + kzz⋅Mz,Ed/(Mz,Rk/γM1)  
    u = **{(util_62_B if util_62_B is not None else float('nan')):.3f}** {'≤ 1.0 ⇒ OK' if (util_62_B is not None and util_62_B<=1.0) else '> 1.0 ⇒ NOT OK'}""")


    # ----------------------------------------------------
        # 8. References
        # ----------------------------------------------------
    report_h3("8. References")

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
    page_icon=str(asset_path("EngiSnap-Logo.png")) if asset_path("EngiSnap-Logo.png").exists() else "🧰",
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
    safe_image("EngiSnap-Logo.png", width=140)

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
    meta = render_project_data()          # your existing function
    st.session_state["meta"] = meta       # <-- this line is IMPORTANT

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
            if img_path and Path(img_path).exists():
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

            render_results(df_rows, overall_ok, governing, show_footer=True)

        except Exception as e:
            st.error(f"Computation error: {e}")
with tab5:
    render_report_tab()

















































