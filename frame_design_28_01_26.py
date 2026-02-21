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
import streamlit.components.v1 as components
from pathlib import Path

# -----------------------------
# Frame diagram plot styling
# -----------------------------
FRAME_COLOR = "#1f77b4"   # blue-ish, you can change later
LW = 2.0                 # line width for frame sketch

# ----------------------------------------------------
# Material properties (EN 10025-2) – simplified table values
# NOTE: In reality, fy depends on thickness. Here we use the table values you showed.
# ----------------------------------------------------
MATERIAL_PROPS = {
    "S235": {"fy": 235.0, "fu": 360.0},
    "S275": {"fy": 275.0, "fu": 430.0},
    "S355": {"fy": 355.0, "fu": 490.0},
    "S450": {"fy": 440.0, "fu": 550.0},  # per your table screenshot
}

def _pick(sr: dict, keys, default=None):
    """Return the first existing, non-empty key from sr_display."""
    for k in keys:
        if k in sr and sr.get(k) not in (None, "", "n/a"):
            return sr.get(k)
    return default

def _to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

def get_material_props(grade: str):
    d = MATERIAL_PROPS.get(str(grade), MATERIAL_PROPS["S355"])
    return float(d["fy"]), float(d["fu"])
# =========================================================
# EC3 Section class (material-dependent) — based on ε = sqrt(235/fy)
#
# OUTPUT (ONLY 3 classes):
#   - flange_comp      : flange class in pure compression
#   - web_bend_comp    : web class in bending + axial compression (Table 7.3, uses αc + ψ)
#   - governing        : max(flange_comp, web_bend_comp)
#
# IMPLEMENTATION NOTES:
# - Web bending+axial uses Table 7.3:
#     Class 1 & 2 limits via αc
#     Class 3 limit via ψ
# - ψ is computed from extreme-fibre stresses due to N + My:
#     σ = N/A ± My/Wel,y
# - N is assumed POSITIVE in compression (Eurocode convention).
#   If your app uses compression negative, flip N once where marked.
# =========================================================
import math

def ec3_epsilon(fy_MPa: float) -> float:
    fy = max(float(fy_MPa), 1e-9)
    return math.sqrt(235.0 / fy)

def _class_from_limits(ct: float, lim1: float, lim2: float, lim3: float) -> int:
    r = float(ct)
    if r <= lim1: return 1
    if r <= lim2: return 2
    if r <= lim3: return 3
    return 4


# ------------------------
# Table 7.4 — outstand flange in PURE compression
# ------------------------
def ec3_outstand_flange_class_pure_comp(ct: float, fy_MPa: float) -> int:
    eps = ec3_epsilon(fy_MPa)
    return _class_from_limits(ct, 9.0*eps, 10.0*eps, 14.0*eps)


# ------------------------
# Table 7.3 — internal part in PURE uniform compression
# (used for rectangular flange in pure compression)
# ------------------------
def ec3_internal_part_class_uniform_comp(ct: float, fy_MPa: float) -> int:
    eps = ec3_epsilon(fy_MPa)
    return _class_from_limits(ct, 28.0*eps, 34.0*eps, 38.0*eps)


# ------------------------
# αc from Table 7.3 note form (general algebra)
# denom = c * tw * fy  (N)
# ------------------------
def ec3_alpha_c_from_denom(NEd_N: float, denom_N: float) -> float:
    if denom_N <= 0.0:
        return 0.5
    if NEd_N >= denom_N:
        return 1.0
    if NEd_N <= -denom_N:
        return 0.0
    return 0.5 * (1.0 + (NEd_N / denom_N))


# ------------------------
# Table 7.3 — Class 3 limit for bending + axial force uses ψ
# returns LIMIT (not a class)
# ------------------------
def ec3_internal_part_class3_limit_bending_axial_psi(fy_MPa: float, psi: float) -> float:
    eps = ec3_epsilon(fy_MPa)
    psi = float(psi)

    if psi > -1.0:
        denom = 0.608 + 0.343 * psi + 0.049 * psi * psi
        return (38.0 * eps) / max(denom, 1e-9)
    else:
        # ψ <= -1
        return 60.5 * eps * (1.0 - psi)


# ------------------------
# Table 7.3 — web class in bending + axial force
# - Class 1 & 2 via αc
# - Class 3 via ψ
# ------------------------
def ec3_internal_part_class_bending_axial(ct: float, fy_MPa: float, alpha_c: float, psi: float) -> int:
    eps = ec3_epsilon(fy_MPa)
    a = max(float(alpha_c), 1e-9)

    # Class 1 limits
    if a > 0.5:
        lim1 = (126.0 * eps) / max(5.5 * a - 1.0, 1e-9)
    else:
        lim1 = (36.0 * eps) / a

    # Class 2 limits
    if a > 0.5:
        lim2 = (188.0 * eps) / max(6.53 * a - 1.0, 1e-9)
    else:
        lim2 = (41.5 * eps) / a

    # Class 3 limit via ψ
    lim3 = ec3_internal_part_class3_limit_bending_axial_psi(fy_MPa, psi)

    return _class_from_limits(ct, lim1, lim2, lim3)


# ------------------------
# ψ from N + My using extreme fibre stresses
# σ = N/A ± My/Wel,y
# compression positive convention
# ------------------------
def ec3_psi_from_N_My(NEd_kN: float, My_Ed_kNm: float, A_mm2: float, Wel_y_cm3: float) -> float | None:
    if A_mm2 <= 0.0 or Wel_y_cm3 <= 0.0:
        return None

    # Units
    NEd_N = float(NEd_kN) * 1e3
    NEd_N = -NEd_N   # YOUR APP: compression is negative -> EC3 needs compression positive
    My_Nmm = float(My_Ed_kNm) * 1e6
    Wel_y_mm3 = float(Wel_y_cm3) * 1e3

    # If your app uses compression NEGATIVE, uncomment next line:
    NEd_N *= -1.0

    sigma_N = NEd_N / A_mm2
    sigma_M = My_Nmm / Wel_y_mm3

    s_top = sigma_N + sigma_M
    s_bot = sigma_N - sigma_M

    # compression positive: take the more compressed as σ1
    s1 = max(s_top, s_bot)
    s2 = min(s_top, s_bot)

    if abs(s1) <= 1e-9:
        return None

    psi = s2 / s1
    # clamp for stability
    psi = max(-1.0, min(1.0, float(psi)))
    return psi


def calc_section_classes_ec3(sr_display: dict, fy_MPa: float,
                            NEd_kN: float = 0.0,
                            My_Ed_kNm: float = 0.0) -> dict:

    fam = str(sr_display.get("family", sr_display.get("Type", "")) or "").upper()

    # geometry (mm)
    h  = float(sr_display.get("h_mm", 0.0) or 0.0)
    b  = float(sr_display.get("b_mm", 0.0) or 0.0)
    tw = float(sr_display.get("tw_mm", 0.0) or 0.0)
    tf = float(sr_display.get("tf_mm", 0.0) or 0.0)

    # props for ψ
    A_mm2 = float(sr_display.get("A_mm2", 0.0) or 0.0)
    Wel_y_cm3 = float(sr_display.get("Wel_y_cm3", 0.0) or 0.0)

    res = {
        "flange_comp": None,
        "web_bend_comp": None,
        "governing": None,
        # optional debug (keep if you want)
        "alpha_c": None,
        "psi": None,
        "ct_flange": None,
        "ct_web": None,
        "notes": ""
    }

    # --------------------------
    # I/H sections
    # --------------------------
    is_IH = any(k in fam for k in ["IPE", "HEA", "HEB", "HEM", "IPN", "I ", "H "])
    if is_IH:
        if not (h > 0 and b > 0 and tw > 0 and tf > 0):
            res["notes"] = "Missing geometry for I/H (need h,b,tw,tf)."
            sr_display["class_map_calc"] = res
            return res

        # flange outstand: c = (b - tw)/2 ; t = tf
        c_f = max((b - tw) / 2.0, 0.0)
        ct_f = c_f / tf if tf > 0 else 1e9

        # web internal: c = h - 2tf ; t = tw
        c_w = max(h - 2.0 * tf, 0.0)
        ct_w = c_w / tw if tw > 0 else 1e9

        # ψ from N + My
        psi = ec3_psi_from_N_My(NEd_kN, My_Ed_kNm, A_mm2, Wel_y_cm3)
        if psi is None:
            psi = 1.0  # safe fallback

        # αc from Table 7.3 note
        # denom = c * tw * fy   (N)
        NEd_N = float(NEd_kN) * 1e3
        # If your app uses compression NEGATIVE, uncomment next line:
        NEd_N *= -1.0
        denom = c_w * tw * float(fy_MPa)
        alpha_c = ec3_alpha_c_from_denom(NEd_N, denom)

        flange_cls = ec3_outstand_flange_class_pure_comp(ct_f, fy_MPa)
        web_cls    = ec3_internal_part_class_bending_axial(ct_w, fy_MPa, alpha_c, psi)

        res["flange_comp"] = flange_cls
        res["web_bend_comp"] = web_cls
        res["governing"] = max(flange_cls, web_cls)

        res["alpha_c"] = alpha_c
        res["psi"] = psi
        res["ct_flange"] = ct_f
        res["ct_web"] = ct_w
        res["notes"] = "I/H: flange Table 7.4 pure comp; web Table 7.3 bending+axial (αc+ψ)."

    # --------------------------
    # RHS / SHS / rectangular box
    # --------------------------
    elif any(k in fam for k in ["RHS", "SHS", "BOX", "RECT", "SQUARE"]):
        if not (h > 0 and b > 0 and tw > 0 and tf > 0):
            res["notes"] = "Missing geometry for RHS/SHS (need h,b,tw,tf)."
            sr_display["class_map_calc"] = res
            return res

        # conservative flat widths for non-uniform thickness box:
        # web (vertical wall): clear depth between corners
        c_w = max(h - 2.0*tf - tw, 0.0)
        ct_w = c_w / tw if tw > 0 else 1e9

        # flange (horizontal wall): clear width between corners
        c_f = max(b - 2.0*tw - tf, 0.0)
        ct_f = c_f / tf if tf > 0 else 1e9

        # flange pure compression: internal part uniform compression (Table 7.3)
        flange_cls = ec3_internal_part_class_uniform_comp(ct_f, fy_MPa)

        # ψ from N + My (same stress-based approach)
        psi = ec3_psi_from_N_My(NEd_kN, My_Ed_kNm, A_mm2, Wel_y_cm3)
        if psi is None:
            psi = 1.0

        # αc approximation using same normalized force idea
        NEd_N = float(NEd_kN) * 1e3
        # If your app uses compression NEGATIVE, uncomment next line:
        NEd_N *= -1.0
        denom = c_w * tw * float(fy_MPa)
        alpha_c = ec3_alpha_c_from_denom(NEd_N, denom)

        web_cls = ec3_internal_part_class_bending_axial(ct_w, fy_MPa, alpha_c, psi)

        res["flange_comp"] = flange_cls
        res["web_bend_comp"] = web_cls
        res["governing"] = max(flange_cls, web_cls)

        res["alpha_c"] = alpha_c
        res["psi"] = psi
        res["ct_flange"] = ct_f
        res["ct_web"] = ct_w
        res["notes"] = "RHS/SHS: flange Table 7.3 uniform comp; web Table 7.3 bending+axial (αc+ψ)."

    else:
        res["notes"] = "No mapping for this family (CHS/angles etc.)."
        sr_display["class_map_calc"] = res
        return res

    # writeback for the rest of your app (Wpl/Wel switch)
    if res["governing"] is not None:
        sr_display["class"] = int(res["governing"])
    sr_display["class_map_calc"] = res
    return res

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


def report_status_badge(status, show_icon: bool = True):
    import math

    # Case 1: numeric utilization (e.g. 0.83)
    if isinstance(status, (int, float)):
        if not math.isfinite(float(status)):
            label = "n/a"
            is_ok = False
        else:
            is_ok = float(status) <= 1.0
            label = "OK" if is_ok else "NOT OK"

    # Case 2: string status ("OK", "NOT OK", "EXCEEDS", ...)
    else:
        s = str(status or "").strip().upper()
        is_ok = s.startswith("OK") or s in {"PASS", "SAFE", "SATISFIED"}
        if "EXCEED" in s or "NOT" in s or "FAIL" in s:
            is_ok = False
        label = "OK" if is_ok else "NOT OK"

    icon = "✅" if is_ok else "❌"
    color = "#1b8f2a" if is_ok else "#c62828"
    if not show_icon:
        icon = ""

    st.markdown(
        f"""<div style="margin-top:6px;margin-bottom:14px;font-weight:650;color:{color};">
        {icon} {label}
        </div>""",
        unsafe_allow_html=True,
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
    
# ============================================================
# Simply Supported Beams (SSB) — ALL LENGTHS IN mm
# Units:
#   L_mm, a, b, c, a1, a2 in mm
#   w, w1, w2 in kN/m
#   F, F1, F2, P in kN
#   M1, M2 in kN·m
#   E in Pa, I in m^4
# ============================================================

import numpy as np

def _mm_to_m(x_mm: float) -> float:
    return float(x_mm) / 1000.0


# ----------------------------
# CASE 1: SSB-C1 (UDL full span)
# ----------------------------
def ss_udl_case(L_mm, w):
    x, V, M, delta = ss_udl_diagram(L_mm, w, E=None, I=None, n=801)
    Vmax = float(np.nanmax(np.abs(V))) if V is not None else 0.0
    Mmax = float(np.nanmax(np.abs(M))) if M is not None else 0.0
    return (0.0, Mmax, 0.0, Vmax, 0.0)

def ss_udl_diagram(L_mm, w, E=None, I=None, n=801):
    L = _mm_to_m(L_mm)
    w = float(w)

    if L <= 0:
        x = np.array([0.0, 0.0])
        return x, np.zeros_like(x), np.zeros_like(x), None

    x = np.linspace(0.0, L, n)

    # Reactions
    R1 = w * L / 2.0
    R2 = w * L / 2.0

    # Shear and moment
    V = R1 - w * x
    M = R1 * x - w * x**2 / 2.0

    # Deflection (classic)
    delta = None
    if E and I and I > 0:
        w_Nm = w * 1000.0  # kN/m -> N/m
        delta = (w_Nm * x * (L**3 - 2*L*x**2 + x**3)) / (24.0 * E * I)

    return x, V, M, delta

def ss_udl_deflection_max(L_mm, w, E, I):
    L = _mm_to_m(L_mm)
    if not E or not I or I <= 0 or L <= 0:
        return None
    # max at midspan: 5 w L^4 / (384 EI)
    w_Nm = float(w) * 1000.0
    return (5.0 * w_Nm * L**4) / (384.0 * E * I)  # m


# --------------------------------
# CASE 2: SSB-C2 (central point load)
# --------------------------------
def ss_central_point_case(L_mm, F):
    x, V, M, delta = ss_central_point_diagram(L_mm, F, E=None, I=None, n=801)
    Vmax = float(np.nanmax(np.abs(V))) if V is not None else 0.0
    Mmax = float(np.nanmax(np.abs(M))) if M is not None else 0.0
    return (0.0, Mmax, 0.0, Vmax, 0.0)

def ss_central_point_diagram(L_mm, F, E=None, I=None, n=801):
    L = _mm_to_m(L_mm)
    F = float(F)

    if L <= 0:
        x = np.array([0.0, 0.0])
        return x, np.zeros_like(x), np.zeros_like(x), None

    x = np.linspace(0.0, L, n)
    xP = L / 2.0

    # Reactions
    R1 = F / 2.0
    R2 = F / 2.0

    # Heaviside for point load
    H = (x >= xP).astype(float)

    V = R1 - F * H
    M = R1 * x - F * (x - xP) * H

    delta = None
    if E and I and I > 0:
        # standard piecewise deflection, implemented numerically via M/EI integration
        M_Nm = M * 1000.0  # kN·m -> N·m
        kappa = M_Nm / (E * I)

        dx = x[1] - x[0]
        theta = np.cumsum(kappa) * dx
        y = np.cumsum(theta) * dx

        # enforce y(0)=0 and y(L)=0 via linear correction
        y = y - (y[-1] / L) * x
        delta = y

    return x, V, M, delta

def ss_central_point_deflection_max(L_mm, F, E, I):
    L = _mm_to_m(L_mm)
    if not E or not I or I <= 0 or L <= 0:
        return None
    # max at midspan: F L^3 / (48 EI)
    F_N = float(F) * 1000.0
    return (F_N * L**3) / (48.0 * E * I)  # m


# -------------------------------------------------------------------------
# CASE 3: SSB-C3 (Two point loads + partial UDL at any point)
#
# Your inputs:
#   L_mm : span in mm
#   F1   : kN, at distance a1 (mm) from LEFT support
#   F2   : kN, at distance a2 (mm) from RIGHT support  -> x2 = L - a2
#   w    : kN/m, partial UDL intensity
#   a    : mm, distance from LEFT support to start of partial UDL
#   b    : mm, length of partial UDL
# -------------------------------------------------------------------------
def ssb_c3_case(L_mm, F1, a1, F2, a2, w, a, b):
    x, V, M, delta = ssb_c3_diagram(L_mm, F1, a1, F2, a2, w, a, b, E=None, I=None, n=1001)
    Vmax = float(np.nanmax(np.abs(V))) if V is not None else 0.0
    Mmax = float(np.nanmax(np.abs(M))) if M is not None else 0.0
    return (0.0, Mmax, 0.0, Vmax, 0.0)

def ssb_c3_diagram(L_mm, F1, a1, F2, a2, w, a, b, E=None, I=None, n=1001):
    L = _mm_to_m(L_mm)
    F1 = float(F1)
    F2 = float(F2)
    w = float(w)

    x1 = _mm_to_m(a1)                 # from left
    x2 = L - _mm_to_m(a2)             # from left (because a2 is from right)

    udl_start = _mm_to_m(a)
    udl_len = _mm_to_m(b)
    udl_end = udl_start + udl_len

    if L <= 0:
        x = np.array([0.0, 0.0])
        return x, np.zeros_like(x), np.zeros_like(x), None

    # clamp positions into [0, L]
    x1 = max(0.0, min(x1, L))
    x2 = max(0.0, min(x2, L))
    udl_start = max(0.0, min(udl_start, L))
    udl_len = max(0.0, udl_len)
    udl_end = max(udl_start, min(udl_end, L))
    b_eff = max(0.0, udl_end - udl_start)

    x = np.linspace(0.0, L, n)

    # Resultant of partial UDL
    W = w * b_eff
    xW = udl_start + b_eff / 2.0 if b_eff > 0 else 0.0

    # Reactions (classic simply supported)
    # moment about left: R2*L = F1*x1 + F2*x2 + W*xW
    Mtot = F1 * x1 + F2 * x2 + W * xW
    R2 = Mtot / L
    R1 = (F1 + F2 + W) - R2

    # Shear using step functions
    H1 = (x >= x1).astype(float)
    H2 = (x >= x2).astype(float)

    V = R1 - F1 * H1 - F2 * H2

    # subtract partial UDL shear contribution
    if b_eff > 0:
        # within UDL: subtract w*(x-udl_start)
        in_udl = (x >= udl_start) & (x <= udl_end)
        V[in_udl] -= w * (x[in_udl] - udl_start)
        # right of UDL: subtract full resultant W
        right_udl = (x > udl_end)
        V[right_udl] -= W

    # Moment = integrate loads in closed form
    M = R1 * x - F1 * (x - x1) * H1 - F2 * (x - x2) * H2

    if b_eff > 0:
        M_udl = np.zeros_like(x)
        in_udl = (x >= udl_start) & (x <= udl_end)
        M_udl[in_udl] = w * (x[in_udl] - udl_start)**2 / 2.0

        right_udl = (x > udl_end)
        M_udl[right_udl] = W * (x[right_udl] - xW)

        M -= M_udl

    # Deflection by numeric integration of curvature M/EI
    delta = None
    if E and I and I > 0:
        M_Nm = M * 1000.0  # kN·m -> N·m
        kappa = M_Nm / (E * I)

        dx = x[1] - x[0]
        theta = np.cumsum(kappa) * dx
        y = np.cumsum(theta) * dx

        # enforce y(0)=0 and y(L)=0 via linear correction
        y = y - (y[-1] / L) * x
        delta = y

    return x, V, M, delta


# -----------------------------------------------
# CASE 4: SSB-C4 (two partial UDLs) — lengths in mm
# -----------------------------------------------
def ssb_c4_case(L_mm, a, b, c, w1, w2):
    x, V, M, delta = ssb_c4_diagram(L_mm, a, b, c, w1, w2, E=None, I=None, n=801)
    Vmax = float(np.nanmax(np.abs(V))) if V is not None else 0.0
    Mmax = float(np.nanmax(np.abs(M))) if M is not None else 0.0
    return (0.0, Mmax, 0.0, Vmax, 0.0)

def ssb_c4_diagram(L_mm, a, b, c, w1, w2, E=None, I=None, n=801):
    L = _mm_to_m(L_mm)
    a = _mm_to_m(a)
    b = _mm_to_m(b)
    c = _mm_to_m(c)
    w1 = float(w1)
    w2 = float(w2)

    if L <= 0:
        x = np.array([0.0, 0.0])
        return x, np.zeros_like(x), np.zeros_like(x), None

    # spans sanity
    a = max(0.0, min(a, L))
    b = max(0.0, b)
    c = max(0.0, min(c, max(0.0, L - a)))

    x0_1 = 0.0
    x1_1 = a
    x0_2 = a + b
    x1_2 = min(L, x0_2 + c)

    W1 = w1 * max(0.0, x1_1 - x0_1)
    xW1 = (x0_1 + x1_1) / 2.0 if W1 > 0 else 0.0

    W2 = w2 * max(0.0, x1_2 - x0_2)
    xW2 = (x0_2 + x1_2) / 2.0 if W2 > 0 else 0.0

    # reactions
    Mtot = W1 * xW1 + W2 * xW2
    R2 = Mtot / L
    R1 = (W1 + W2) - R2

    x = np.linspace(0.0, L, n)

    V = np.full_like(x, R1)

    # subtract distributed shears
    # UDL 1 on [0, a]
    m1 = (x >= x0_1) & (x <= x1_1)
    V[m1] -= w1 * (x[m1] - x0_1)
    V[x > x1_1] -= W1

    # UDL 2 on [x0_2, x1_2]
    m2 = (x >= x0_2) & (x <= x1_2)
    V[m2] -= w2 * (x[m2] - x0_2)
    V[x > x1_2] -= W2

    # moment
    M = R1 * x

    # subtract moment of UDLs
    # UDL1
    M1 = np.zeros_like(x)
    m1 = (x >= x0_1) & (x <= x1_1)
    M1[m1] = w1 * (x[m1] - x0_1)**2 / 2.0
    M1[x > x1_1] = W1 * (x[x > x1_1] - xW1)
    M -= M1

    # UDL2
    M2 = np.zeros_like(x)
    m2 = (x >= x0_2) & (x <= x1_2)
    M2[m2] = w2 * (x[m2] - x0_2)**2 / 2.0
    M2[x > x1_2] = W2 * (x[x > x1_2] - xW2)
    M -= M2

    delta = None
    if E and I and I > 0:
        M_Nm = M * 1000.0
        kappa = M_Nm / (E * I)
        dx = x[1] - x[0]
        theta = np.cumsum(kappa) * dx
        y = np.cumsum(theta) * dx
        y = y - (y[-1] / L) * x
        delta = y

    return x, V, M, delta


# --------------------------------------------
# CASE 5: SSB-C5 (UDL + mid-point load + end moments) — L in mm
# --------------------------------------------
def ssb_c5_case(L_mm, w, P, M1, M2):
    x, V, M, delta = ssb_c5_diagram(L_mm, w, P, M1, M2, E=None, I=None, n=1001)
    Vmax = float(np.nanmax(np.abs(V))) if V is not None else 0.0
    Mmax = float(np.nanmax(np.abs(M))) if M is not None else 0.0
    return (0.0, Mmax, 0.0, Vmax, 0.0)

def ssb_c5_diagram(L_mm, w, P, M1, M2, E=None, I=None, n=1001):
    L = _mm_to_m(L_mm)
    w = float(w)
    P = float(P)
    M1 = float(M1)
    M2 = float(M2)

    if L <= 0.0:
        x = np.array([0.0, 0.0])
        return x, np.zeros_like(x), np.zeros_like(x), None

    x = np.linspace(0.0, L, n)
    xP = L / 2.0

    # Reactions from equilibrium:
    # SumV: R1 + R2 = wL + P
    # SumM about left: R2*L = wL*(L/2) + P*(L/2) + M1 + M2  (sign convention as before)
    # Keep the SAME sign convention you used earlier: end moments directly add to bending diagram.
    R2 = (w * L * (L / 2.0) + P * (L / 2.0) + M1 + M2) / L
    R1 = (w * L + P) - R2

    H = (x >= xP).astype(float)

    V = R1 - w * x - P * H
    M = R1 * x - w * x**2 / 2.0 - P * (x - xP) * H

    # add end moments linearly (same as your older approach)
    M = M + M1 * (1.0 - x / L) + M2 * (x / L)

    delta = None
    if E and I and I > 0:
        M_Nm = M * 1000.0
        kappa = M_Nm / (E * I)
        dx = x[1] - x[0]
        theta = np.cumsum(kappa) * dx
        y = np.cumsum(theta) * dx
        y = y - (y[-1] / L) * x
        delta = y

    return x, V, M, delta


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


def feb_c1_case(L_mm, w, a, F):
    """
    FEB-C1: Propped cantilever (pin at x=0, fixed at x=L)
    Loads: full-span UDL w + point load F at x=a
    Inputs: L_mm (mm), w (kN/m), a (m), F (kN)

    Returns (N, My, Mz, Vy, Vz) maxima for prefill.
    """
    L = float(L_mm) / 1000.0  # m
    a = float(a)
    w = float(w)
    F = float(F)

    if L <= 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    # clamp a into [0, L]
    a = max(0.0, min(a, L))
    b = L - a

    # Reactions (kN)
    R1_w = 3.0 * w * L / 8.0
    R1_F = (F * b**2 * (a + 2.0 * L)) / (2.0 * L**3) if L > 0 else 0.0
    R1 = R1_w + R1_F

    # Use diagram arrays to get maxima (robust)
    x, V, M, _ = feb_c1_diagram(L_mm, w, a, F, E=None, I=None, n=1001)

    Mmax = float(np.max(np.abs(M))) if M is not None else 0.0
    Vmax = float(np.max(np.abs(V))) if V is not None else 0.0

    return (0.0, Mmax, 0.0, Vmax, 0.0)


def feb_c1_diagram(L_mm, w, a, F, E=None, I=None, n=801):
    """
    Returns x (m), V (kN), M (kN·m), delta (m) for FEB-C1:
    Propped cantilever: pin at x=0, fixed at x=L
    Loads: UDL w over [0,L] + point load F at x=a
    Inputs: L_mm (mm), w (kN/m), a (m), F (kN)
    """
    L = float(L_mm) / 1000.0  # m
    a = float(a)
    w = float(w)
    F = float(F)

    if L <= 0:
        x = np.array([0.0, 0.0])
        return x, np.zeros_like(x), np.zeros_like(x), None

    # clamp a
    a = max(0.0, min(a, L))
    b = L - a

    # Reactions at x=0 (kN)
    R1_w = 3.0 * w * L / 8.0
    R1_F = (F * b**2 * (a + 2.0 * L)) / (2.0 * L**3)
    R1 = R1_w + R1_F

    x = np.linspace(0.0, L, n)

    H = (x >= a).astype(float)

    # Shear (kN): V = R1 - w x - F H(x-a)
    V = R1 - w * x - F * H

    # Moment (kN·m): M = R1 x - w x^2/2 - F (x-a) H(x-a)
    M = R1 * x - (w * x**2) / 2.0 - F * (x - a) * H

    # Deflection (m) if E and I provided
    delta = None
    if E and I and I > 0:
        w_Nm = w * 1000.0  # kN/m -> N/m
        F_N = F * 1000.0   # kN -> N

        # UDL deflection
        delta_w = (w_Nm * x / (48.0 * E * I)) * (L**3 - 3.0*L*x**2 + 2.0*x**3)

        # Point load deflection (piecewise)
        delta_F = np.zeros_like(x)

        mask1 = x < a
        mask2 = ~mask1

        # x < a
        if np.any(mask1):
            xx = x[mask1]
            delta_F[mask1] = (F_N * b**2 * xx / (12.0 * E * I * L**3)) * (
                3.0*a*L**2 - 2.0*L*xx**2 - a*xx**2
            )

        # x >= a
        if np.any(mask2):
            xx = x[mask2]
            delta_F[mask2] = (F_N * a / (12.0 * E * I * L**3)) * (L - xx)**2 * (
                3.0*L**2*xx - a**2*xx - 2.0*a**2*L
            )

        delta = delta_w + delta_F

    return x, V, M, delta
def fbb_c1_case(L_mm, w, a, F):
    """
    FBB-C1: Fixed-Fixed beam (both ends fixed)
    Loads: full-span UDL w + point load F at x=a (from left)
    Inputs: L_mm (mm), w (kN/m), a (m), F (kN)

    Returns (N, My, Mz, Vy, Vz) based on max |M| and |V|.
    """
    x, V, M, _ = fbb_c1_diagram(L_mm, w, a, F, E=None, I=None, n=1001)
    Vmax = float(np.nanmax(np.abs(V))) if V is not None else 0.0
    Mmax = float(np.nanmax(np.abs(M))) if M is not None else 0.0
    return (0.0, Mmax, 0.0, Vmax, 0.0)


def fbb_c1_diagram(L_mm, w, a, F, E=None, I=None, n=801):
    """
    Fixed-Fixed beam (both ends fixed)
    Loads: full-span UDL w + point load F at x=a (from left)
    Returns x (m), V (kN), M (kN·m), delta (m or None)
    """
    L = float(L_mm) / 1000.0  # m
    w = float(w)              # kN/m
    F = float(F)              # kN
    a = float(a)              # m

    if L <= 0:
        x = np.array([0.0, 0.0])
        return x, np.zeros_like(x), np.zeros_like(x), None

    # clamp a into [0, L]
    a = max(0.0, min(a, L))
    b = L - a

    x = np.linspace(0.0, L, n)

    # -------------------------
    # (1) UDL part (fixed-fixed)
    # -------------------------
    # Reactions: R = wL/2 ; Shear: V = w(L/2 - x)
    V_w = w * (L / 2.0 - x)

    # Moment: M(x) = (w/12) * (6 L x - L^2 - 6 x^2)
    M_w = (w / 12.0) * (6.0 * L * x - L**2 - 6.0 * x**2)

    # Deflection: delta(x) = w x^2 (L-x)^2 / (24 E I)
    delta_w = None
    if E and I and I > 0:
        w_Nm = w * 1000.0
        delta_w = (w_Nm * x**2 * (L - x)**2) / (24.0 * E * I)

    # ------------------------------------------
    # (2) Point load part (fixed-fixed, at x=a)
    # ------------------------------------------
    # Reactions (kN)
    # R1 = P b^2/L^3 (3a + b), R2 = P a^2/L^3 (a + 3b)
    if L > 0:
        R1_F = (F * b**2 * (3.0 * a + b)) / (L**3)
        # R2_F not needed for V/M construction but shown for completeness:
        # R2_F = (F * a**2 * (a + 3.0 * b)) / (L**3)
    else:
        R1_F = 0.0

    # End moments magnitudes (kN·m), hogging at ends (negative in our sign)
    M1 = (F * a * b**2) / (L**2) if L > 0 else 0.0  # left end magnitude
    # M2 = (F * a**2 * b) / (L**2)                   # right end magnitude (not required below)

    H = (x >= a).astype(float)

    # Shear for point load: V = R1 for x<a ; V = R1 - F for x>=a
    V_F = R1_F - F * H

    # Moment for point load (from your sheet):
    # for x<a:  M = R1*x - (P a b^2 / L^2)
    # for x>=a: M = R1*x - P(x-a) - (P a b^2 / L^2)
    M_F = R1_F * x - M1 - F * (x - a) * H

    # Deflection (piecewise) from standard fixed-fixed point-load expressions
    delta_F = None
    if E and I and I > 0:
        F_N = F * 1000.0
        delta_F = np.zeros_like(x)

        mask1 = x < a
        mask2 = ~mask1

        # x < a
        if np.any(mask1):
            xx = x[mask1]
            delta_F[mask1] = (F_N * b**2 * xx**2 / (6.0 * E * I * L**3)) * (
                3.0 * a * L - xx * (3.0 * a + b)
            )

        # x >= a
        if np.any(mask2):
            xx = x[mask2]
            uu = (L - xx)
            delta_F[mask2] = (F_N * a**2 * uu**2 / (6.0 * E * I * L**3)) * (
                3.0 * b * L - uu * (3.0 * b + a)
            )

    # -------------------------
    # (3) Superposition
    # -------------------------
    V = V_w + V_F
    M = M_w + M_F

    delta = None
    if E and I and I > 0:
        # delta_w or delta_F could be None if something odd; guard anyway
        delta = 0.0
        if delta_w is not None:
            delta = delta + delta_w
        if delta_F is not None:
            delta = delta + delta_F

    return x, V, M, delta
def cant_c1_case(L_mm, w, a, F, M):
    """
    Cantilever general case (x measured from FREE end):
    - Free end at x=0, Fixed end at x=L
    - UDL w over [0,L]
    - Point load F at x=a (measured from FREE end)
    - End moment M (kN·m) applied at FREE end

    Inputs: L_mm (mm), w (kN/m), a (m from FREE end), F (kN), M (kN·m)
    Returns (N, My, Mz, Vy, Vz) based on max |M| and |V|.
    Also stores max deflection (meters) into: st.session_state["ready_case_delta_max_m"]
    """
    x, V, Mx, _ = cant_c1_diagram(L_mm, w, a, F, M, E=None, I=None, n=1001)
    Vmax = float(np.nanmax(np.abs(V))) if V is not None else 0.0
    Mmax = float(np.nanmax(np.abs(Mx))) if Mx is not None else 0.0

    # Try to compute Δmax using closed-form (only if E & I are available in session_state)
    # Adjust these keys if your app stores E/I under different names.
    try:
        E = float(st.session_state.get("E", 0.0))          # Pa
        I_m4 = float(st.session_state.get("I_m4", 0.0))    # m^4
    except Exception:
        E, I_m4 = 0.0, 0.0

    if E > 0 and I_m4 > 0:
        L = float(L_mm) / 1000.0
        w = float(w)
        F = float(F)
        M_end = float(M)

        if L > 0:
            a = max(0.0, min(float(a), L))   # from FREE end
            s_a = L - a                      # distance from FIXED end

            w_Nm = w * 1000.0         # N/m
            F_N = F * 1000.0          # N
            M_Nm = M_end * 1000.0     # N·m

            # Superposition of maximum deflection magnitudes (at FREE end in classic form)
            # Using s_a (distance from fixed):
            delta_w = (w_Nm * L**4) / (8.0 * E * I_m4)
            delta_F = (F_N * s_a**2 * (3.0*L - s_a)) / (6.0 * E * I_m4)
            delta_M = (M_Nm * L**2) / (2.0 * E * I_m4)

            st.session_state["ready_case_delta_max_m"] = float(delta_w + delta_F + delta_M)

    return (0.0, Mmax, 0.0, Vmax, 0.0)


def cant_c1_diagram(L_mm, w, a, F, M, E=None, I=None, n=801):
    """
    Cantilever general case (x measured from FREE end):
    - Free end at x=0, Fixed end at x=L
    - UDL w (kN/m) over full span
    - Point load F (kN) at x=a (m from FREE end)
    - End moment M (kN·m) applied at FREE end

    Returns:
      x (m), V (kN), Mx (kN·m), delta=None  (no deflection curve)
    """
    L = float(L_mm) / 1000.0  # m
    w = float(w)              # kN/m
    F = float(F)              # kN
    M_end = float(M)          # kN·m
    a = float(a)              # m (from FREE end)

    if L <= 0:
        x = np.array([0.0, 0.0])
        return x, np.zeros_like(x), np.zeros_like(x), None

    # clamp a into [0, L] (from FREE end)
    a = max(0.0, min(a, L))

    # x from FREE end
    x = np.linspace(0.0, L, n)

    # -------------------------
    # (1) UDL part (cantilever, x from FREE end)
    # -------------------------
    # V(x) = -w*x
    # M(x) = -w*x^2/2
    V_w = -w * x
    M_w = -w * x**2 / 2.0

    # -------------------------
    # (2) Point load part at x=a (from FREE end)
    # -------------------------
    # For sections 0 <= x <= a (between FREE end and load): affected
    # V = -F ; M = -F (a - x)
    # For x > a: V = 0 ; M = 0
    mask = (x <= a).astype(float)
    V_F = -F * mask
    M_F = -F * (a - x) * mask

    # -------------------------
    # (3) End moment applied at FREE end
    # -------------------------
    # Constant internal moment along beam
    M_M = -M_end * np.ones_like(x)

    V = V_w + V_F
    Mx = M_w + M_F + M_M

    # We do NOT return deflection curve (no plot)
    return x, V, Mx, None
def cant_c1_delta_max(L_mm, w, a, F, M, E, I):
    """
    Max deflection magnitude for cantilever (meters), by closed-form superposition.
    Convention: x from FREE end, a from FREE end.
    """
    L = float(L_mm) / 1000.0
    if L <= 0 or not E or not I or I <= 0:
        return None

    w = float(w)   # kN/m
    F = float(F)   # kN
    M = float(M)   # kN·m

    a = max(0.0, min(float(a), L))   # from FREE end
    s_a = L - a                      # from FIXED end

    w_Nm = w * 1000.0
    F_N = F * 1000.0
    M_Nm = M * 1000.0

    # Superposition (magnitudes):
    delta_w = (w_Nm * L**4) / (8.0 * E * I)
    delta_F = (F_N * s_a**2 * (3.0 * L - s_a)) / (6.0 * E * I)
    delta_M = (M_Nm * L**2) / (2.0 * E * I)

    return float(delta_w + delta_F + delta_M)
def oh_c1_case(L_mm, a, w1, w2):
    """
    OH - C1: Overhanging beam (right overhang)
    Supports at x=0 and x=L, overhang length a (to x=L+a)
    UDL w1 on [0,L], UDL w2 on [L, L+a]

    Inputs:
      L_mm (mm), a (m), w1 (kN/m), w2 (kN/m)

    Returns (N, My, Mz, Vy, Vz) from max |M| and |V|.
    """
    x, V, M, _ = oh_c1_diagram(L_mm, a, w1, w2, E=None, I=None, n=1201)
    Vmax = float(np.nanmax(np.abs(V))) if V is not None else 0.0
    Mmax = float(np.nanmax(np.abs(M))) if M is not None else 0.0
    return (0.0, Mmax, 0.0, Vmax, 0.0)


def oh_c1_diagram(L_mm, a, w1, w2, E=None, I=None, n=1001):
    """
    OH - C1: Overhanging beam (right overhang)
    Supports: x=0 and x=L. Overhang to x=L+a.
    Loads:
      w1 (kN/m) on [0,L]
      w2 (kN/m) on [L,L+a]

    Returns: x (m), V (kN), M (kN·m), delta (m or None)
    Deflection computed numerically from curvature with y(0)=0 and y(L)=0.
    """
    L = float(L_mm) / 1000.0  # m
    a = float(a)              # m
    w1 = float(w1)            # kN/m
    w2 = float(w2)            # kN/m

    if L <= 0:
        x = np.array([0.0, 0.0])
        return x, np.zeros_like(x), np.zeros_like(x), None

    a = max(0.0, a)
    Lt = L + a  # total length

    x = np.linspace(0.0, Lt, n)

    # ---- Reactions by statics (supports at 0 and L) ----
    # W1 = w1*L at x=L/2, W2 = w2*a at x=L + a/2
    W1 = w1 * L
    W2 = w2 * a

    # Moments about x=0: R2*L = W1*(L/2) + W2*(L + a/2)
    R2 = 0.0
    if L > 0:
        R2 = (W1 * (L / 2.0) + W2 * (L + a / 2.0)) / L
    R1 = (W1 + W2) - R2

    # ---- Shear and moment ----
    V = np.zeros_like(x)
    M = np.zeros_like(x)

    # region 1: 0 <= x <= L (UDL w1)
    m1 = x <= L
    xx = x[m1]
    V[m1] = R1 - w1 * xx
    M[m1] = R1 * xx - (w1 * xx**2) / 2.0

    # values just to the left of support at x=L
    V_L_minus = R1 - w1 * L
    M_L_minus = R1 * L - (w1 * L**2) / 2.0

    # region 2: L <= x <= L+a (overhang, UDL w2)
    m2 = x >= L
    x2 = x[m2]
    x1 = x2 - L  # local coordinate from support at x=L into overhang

    V_L_plus = V_L_minus + R2
    V[m2] = V_L_plus - w2 * x1
    M[m2] = M_L_minus + V_L_plus * x1 - (w2 * x1**2) / 2.0

    # ---- Deflection (numerical), enforce y(0)=0 and y(L)=0 ----
    delta = None
    if E and I and I > 0:
        # curvature k = M/(E I) with consistent units
        M_Nm = M * 1000.0  # kN·m -> N·m
        kappa = M_Nm / (E * I)

        dx = x[1] - x[0]

        # integrate curvature -> slope (theta) with theta(0)=0
        theta = np.zeros_like(x)
        theta[1:] = np.cumsum((kappa[:-1] + kappa[1:]) * 0.5 * dx)

        # integrate slope -> deflection y with y(0)=0
        y = np.zeros_like(x)
        y[1:] = np.cumsum((theta[:-1] + theta[1:]) * 0.5 * dx)

        # enforce y(L)=0 by adding a linear correction: y_corr = y + C1*x
        # find index closest to L
        iL = int(np.argmin(np.abs(x - L)))
        yL = y[iL]
        C1 = -yL / x[iL] if x[iL] != 0 else 0.0
        y = y + C1 * x

        delta = y

    return x, V, M, delta

def oh_c2_case(L_mm, a, F):
    """
    OH - C2: Overhanging beam, point load at free end of overhang.
    Supports at x=0 and x=L, overhang length a to x=L+a, load F at x=L+a (down).

    Inputs: L_mm (mm), a (m), F (kN)
    Returns (N, My, Mz, Vy, Vz) from max |M| and |V|.
    """
    x, V, M, _ = oh_c2_diagram(L_mm, a, F, E=None, I=None, n=1201)
    Vmax = float(np.nanmax(np.abs(V))) if V is not None else 0.0
    Mmax = float(np.nanmax(np.abs(M))) if M is not None else 0.0
    return (0.0, Mmax, 0.0, Vmax, 0.0)


def oh_c2_diagram(L_mm, a, F, E=None, I=None, n=1001):
    """
    OH - C2: Overhanging beam, point load at free end of overhang.
    Supports: x=0 and x=L. Overhang: [L, L+a]. Load F at x=L+a (down).

    Returns: x (m), V (kN), M (kN·m), delta (m or None)
    Deflection computed numerically from curvature with y(0)=0 and y(L)=0.
    """
    L = float(L_mm) / 1000.0  # m
    a = float(a)              # m
    F = float(F)              # kN

    if L <= 0:
        x = np.array([0.0, 0.0])
        return x, np.zeros_like(x), np.zeros_like(x), None

    a = max(0.0, a)
    Lt = L + a

    x = np.linspace(0.0, Lt, n)

    # ---- Reactions by statics ----
    # About x=0: R2*L = F*(L+a)  -> R2 = F*(L+a)/L
    # Vertical: R1 + R2 = F      -> R1 = F - R2 = -F*a/L
    R2 = (F * (L + a)) / L
    R1 = F - R2  # = -F*a/L (uplift possible)

    # ---- Shear V(x) ----
    # 0<=x<L:      V = R1
    # L<=x<L+a:    V = R1 + R2 = F
    # At x=L+a:    drop by F to 0
    H_L  = (x >= L).astype(float)
    H_t  = (x >= (L + a)).astype(float)
    V = R1 + R2 * H_L - F * H_t

    # ---- Moment M(x) ----
    # 0<=x<=L:     M = R1*x
    # L<=x<=L+a:   M = -F*(L+a - x)  (matches your sheet: Mx1 = F(a - x1))
    M = np.zeros_like(x)
    m1 = x <= L
    M[m1] = R1 * x[m1]
    m2 = x >= L
    M[m2] = -F * (L + a - x[m2])

    # ---- Deflection (numerical), enforce y(0)=0 and y(L)=0 ----
    delta = None
    if E and I and I > 0:
        M_Nm = M * 1000.0  # kN·m -> N·m
        kappa = M_Nm / (E * I)

        dx = x[1] - x[0]

        # integrate curvature -> slope (theta), theta(0)=0
        theta = np.zeros_like(x)
        theta[1:] = np.cumsum((kappa[:-1] + kappa[1:]) * 0.5 * dx)

        # integrate slope -> deflection y, y(0)=0
        y = np.zeros_like(x)
        y[1:] = np.cumsum((theta[:-1] + theta[1:]) * 0.5 * dx)

        # enforce y(L)=0 with linear correction y += C1*x
        iL = int(np.argmin(np.abs(x - L)))
        yL = y[iL]
        C1 = -yL / x[iL] if x[iL] != 0 else 0.0
        y = y + C1 * x

        delta = y

    return x, V, M, delta
    
def oh_c3_case(L_mm, a, F):
    """
    OH - C3: Overhanging beam, point load BETWEEN supports.
    Overhang is unloaded and does not affect results.

    Inputs:
      L_mm (mm)  : span between supports
      a (m)      : distance of load from LEFT support
      F (kN)     : point load

    Returns (N, My, Mz, Vy, Vz)
    """
    x, V, M, _ = oh_c3_diagram(L_mm, a, F, E=None, I=None, n=1001)
    Vmax = float(np.nanmax(np.abs(V))) if V is not None else 0.0
    Mmax = float(np.nanmax(np.abs(M))) if M is not None else 0.0
    return (0.0, Mmax, 0.0, Vmax, 0.0)

def oh_c3_diagram(L_mm, a, F, E=None, I=None, n=1001):
    """
    OH - C3: Overhanging beam, point load BETWEEN supports.
    Supports at x=0 and x=L. Overhang unloaded → ignored.

    Returns:
      x (m), V (kN), M (kN·m), delta (m or None)
    """
    L = float(L_mm) / 1000.0
    a = float(a)
    F = float(F)

    if L <= 0:
        x = np.array([0.0, 0.0])
        return x, np.zeros_like(x), np.zeros_like(x), None

    # Clamp load location
    a = max(0.0, min(a, L))
    b = L - a

    x = np.linspace(0.0, L, n)

    # ---- Reactions (from your sheet) ----
    R1 = F * b / L
    R2 = F * a / L

    # ---- Shear ----
    V = np.where(x < a, R1, R1 - F)

    # ---- Moment ----
    M = np.where(
        x < a,
        R1 * x,
        R1 * x - F * (x - a)
    )

    # ---- Deflection (numerical, y(0)=0, y(L)=0) ----
    delta = None
    if E and I and I > 0:
        M_Nm = M * 1000.0
        kappa = M_Nm / (E * I)
        dx = x[1] - x[0]

        theta = np.zeros_like(x)
        theta[1:] = np.cumsum((kappa[:-1] + kappa[1:]) * 0.5 * dx)

        y = np.zeros_like(x)
        y[1:] = np.cumsum((theta[:-1] + theta[1:]) * 0.5 * dx)

        # Enforce y(L)=0
        C1 = -y[-1] / L
        y = y + C1 * x

        delta = y

    return x, V, M, delta

def oh_c4_case(a, b, c, w):
    """
    OH - C4: Overhang both sides, UDL over full length.
    Geometry: left overhang=a, span=b, right overhang=c, total L=a+b+c.
    Supports at x=a and x=a+b.

    Inputs: a (m), b (m), c (m), w (kN/m)
    Returns (N, My, Mz, Vy, Vz) from max |M| and |V|.
    """
    x, V, M, _ = oh_c4_diagram(a, b, c, w, E=None, I=None, n=1401)
    Vmax = float(np.nanmax(np.abs(V))) if V is not None else 0.0
    Mmax = float(np.nanmax(np.abs(M))) if M is not None else 0.0
    return (0.0, Mmax, 0.0, Vmax, 0.0)


def oh_c4_diagram(a, b, c, w, E=None, I=None, n=1201):
    """
    OH - C4: Overhang both supports with UDL over entire length.
    Total length L=a+b+c. Supports at x=a and x=a+b.

    Returns: x (m), V (kN), M (kN·m), delta (m or None)
    Deflection computed numerically from curvature with y(a)=0 and y(a+b)=0.
    """
    a = float(a)
    b = float(b)
    c = float(c)
    w = float(w)  # kN/m

    a = max(0.0, a)
    b = max(1e-9, b)  # avoid zero span
    c = max(0.0, c)

    L = a + b + c
    x = np.linspace(0.0, L, n)

    # supports
    x1 = a
    x2 = a + b

    # ---- Reactions by statics ----
    # total load W = w*L acting at L/2
    # take moments about x1:
    # R2*b = W*(L/2 - x1)  => R2 = w*L*(L/2 - a)/b = w*L*(L-2a)/(2b)
    # then R1 = W - R2 = w*L - R2 = w*L*(L-2c)/(2b)
    R2 = w * L * (L - 2.0 * a) / (2.0 * b)
    R1 = w * L - R2

    # ---- Shear V(x) ----
    # V = -w*x + R1*H(x-x1) + R2*H(x-x2)
    H1 = (x >= x1).astype(float)
    H2 = (x >= x2).astype(float)
    V = -w * x + R1 * H1 + R2 * H2

    # ---- Moment M(x) ----
    # M = -w*x^2/2 + R1*(x-x1)H1 + R2*(x-x2)H2
    M = -w * x**2 / 2.0 + R1 * (x - x1) * H1 + R2 * (x - x2) * H2

    # ---- Deflection (numerical), enforce y(x1)=0 and y(x2)=0 ----
    delta = None
    if E and I and I > 0:
        M_Nm = M * 1000.0  # kN·m -> N·m
        kappa = M_Nm / (E * I)

        dx = x[1] - x[0]

        # integrate curvature -> slope with theta(0)=0
        theta = np.zeros_like(x)
        theta[1:] = np.cumsum((kappa[:-1] + kappa[1:]) * 0.5 * dx)

        # integrate slope -> deflection with y(0)=0
        y = np.zeros_like(x)
        y[1:] = np.cumsum((theta[:-1] + theta[1:]) * 0.5 * dx)

        # Now enforce y(x1)=0 and y(x2)=0 by adding a linear correction y += C0 + C1*x
        i1 = int(np.argmin(np.abs(x - x1)))
        i2 = int(np.argmin(np.abs(x - x2)))

        y1 = y[i1]
        y2 = y[i2]
        X1 = x[i1]
        X2 = x[i2]

        # Solve:
        # y1 + C0 + C1*X1 = 0
        # y2 + C0 + C1*X2 = 0
        denom = (X2 - X1) if (X2 != X1) else 1.0
        C1 = -(y2 - y1) / denom
        C0 = -y1 - C1 * X1

        y = y + C0 + C1 * x
        delta = y

    return x, V, M, delta
def _beam2span_delta_fe(L1, L2, E, I, w1_Nm=0.0, w2_Nm=0.0, point_loads=None, n_per_elem=301):
    """
    2-element Euler-Bernoulli beam FE, nodes at [0, L1, L1+L2].
    Simple supports at ALL 3 nodes -> v=0 at nodes, rotations free.
    Returns x (m) and deflection v(x) (m). (Sign depends on load sign; you use max abs anyway.)
    point_loads: list of tuples (elem_index, x_local, P_N) with elem_index 0 or 1
    """
    import numpy as np

    point_loads = point_loads or []
    Ls = [float(L1), float(L2)]
    n_nodes = 3
    ndof = 2 * n_nodes  # [v0,th0, v1,th1, v2,th2]
    K = np.zeros((ndof, ndof), dtype=float)
    f = np.zeros((ndof,), dtype=float)

    def ke(EI, Le):
        Le = float(Le)
        return (EI / Le**3) * np.array([
            [ 12,   6*Le, -12,   6*Le],
            [  6*Le, 4*Le**2, -6*Le, 2*Le**2],
            [-12,  -6*Le,  12,  -6*Le],
            [  6*Le, 2*Le**2, -6*Le, 4*Le**2]
        ], dtype=float)

    def fe_udl(q, Le):
        # consistent nodal loads for uniform q (N/m) downward positive with v positive
        return q * Le / 2.0 * np.array([1.0, Le/6.0, 1.0, -Le/6.0], dtype=float)

    def fe_point(P, Le, xloc):
        r = float(xloc) / float(Le)
        N1 = 1 - 3*r**2 + 2*r**3
        N2 = Le * (r - 2*r**2 + r**3)
        N3 = 3*r**2 - 2*r**3
        N4 = Le * (-r**2 + r**3)
        return P * np.array([N1, N2, N3, N4], dtype=float)

    EI = float(E) * float(I)

    # assemble 2 elements: e0=[0-1], e1=[1-2]
    x_nodes = [0.0, Ls[0], Ls[0] + Ls[1]]
    for e in [0, 1]:
        Le = Ls[e]
        k_e = ke(EI, Le)

        # dof map
        n1 = e
        n2 = e + 1
        dofs = [2*n1, 2*n1+1, 2*n2, 2*n2+1]

        # assemble K
        for i in range(4):
            for j in range(4):
                K[dofs[i], dofs[j]] += k_e[i, j]

        # UDL on element
        q = w1_Nm if e == 0 else w2_Nm
        if abs(q) > 0:
            f_e = fe_udl(q, Le)
            for i in range(4):
                f[dofs[i]] += f_e[i]

    # point loads
    for (e, xloc, P) in point_loads:
        e = int(e)
        Le = Ls[e]
        xloc = max(0.0, min(float(xloc), Le))
        P = float(P)
        n1 = e
        n2 = e + 1
        dofs = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
        f_e = fe_point(P, Le, xloc)
        for i in range(4):
            f[dofs[i]] += f_e[i]

    # boundary conditions: v=0 at all 3 nodes -> dof 0,2,4 fixed
    fixed = {0, 2, 4}
    free = [i for i in range(ndof) if i not in fixed]

    Kff = K[np.ix_(free, free)]
    ff = f[free]

    # solve
    u = np.zeros((ndof,), dtype=float)
    if len(free) > 0:
        u_free = np.linalg.solve(Kff, ff)
        u[free] = u_free

    # sample deflection along beam
    def shape_v(Le, xloc):
        r = xloc / Le
        N1 = 1 - 3*r**2 + 2*r**3
        N2 = Le * (r - 2*r**2 + r**3)
        N3 = 3*r**2 - 2*r**3
        N4 = Le * (-r**2 + r**3)
        return np.array([N1, N2, N3, N4], dtype=float)

    xs = []
    vs = []

    # element 0
    Le0 = Ls[0]
    dofs0 = [0, 1, 2, 3]
    ue0 = u[dofs0]
    for xi in np.linspace(0.0, Le0, n_per_elem, endpoint=False):
        N = shape_v(Le0, xi)
        vxi = float(N @ ue0)
        xs.append(x_nodes[0] + xi)
        vs.append(vxi)

    # element 1
    Le1 = Ls[1]
    dofs1 = [2, 3, 4, 5]
    ue1 = u[dofs1]
    for xi in np.linspace(0.0, Le1, n_per_elem, endpoint=True):
        N = shape_v(Le1, xi)
        vxi = float(N @ ue1)
        xs.append(x_nodes[1] + xi)
        vs.append(vxi)

    return np.array(xs, dtype=float), np.array(vs, dtype=float)

def cs2_c1_case(a, b, w):
    x, V, M, _ = cs2_c1_diagram(a, b, w, E=None, I=None, n=1201)
    Vmax = float(np.nanmax(np.abs(V))) if V is not None else 0.0
    Mmax = float(np.nanmax(np.abs(M))) if M is not None else 0.0
    return (0.0, Mmax, 0.0, Vmax, 0.0)

def cs2_c1_diagram(a, b, w, E=None, I=None, n=1201):
    """
    Continuous Beam - Two Unequal Spans with UDL on both spans.
    Spans: a (left), b (right). Supports at x=0, x=a, x=a+b.
    """
    a = float(a); b = float(b); w = float(w)
    a = max(1e-9, a); b = max(1e-9, b)
    L = a + b
    x = np.linspace(0.0, L, n)

    # From your sheet:
    # M1 (internal support moment at x=a)
    M1 = -(w*b**3 + w*a**3) / (8.0*(a+b))

    R1 = M1 / a + w*a/2.0
    R3 = M1 / b + w*b/2.0
    R2 = w*a + w*b - R1 - R3

    H_a = (x >= a).astype(float)
    H_L = (x >= L).astype(float)

    # UDL over whole length -> -w*x in shear, -w*x^2/2 in moment
    V = R1 + R2*H_a + R3*H_L - w*x
    M = R1*x + R2*(x-a)*H_a + R3*(x-L)*H_L - (w*x**2)/2.0

    # Deflection by FE (v=0 at 0, a, a+b)
    delta = None
    if E and I and I > 0:
        xs, vs = _beam2span_delta_fe(a, b, E, I, w1_Nm=w*1000.0, w2_Nm=w*1000.0, point_loads=None)
        # interpolate FE deflection to diagram x-grid
        delta = np.interp(x, xs, vs)

    return x, V, M, delta

# ================================
# CS2 - CASE 2 (equal spans, UDL on ONE span)
# ================================
def cs2_c2_case(L, w):
    x, V, M, _ = cs2_c2_diagram(L, w, E=None, I=None, n=1201)
    Vmax = float(np.nanmax(np.abs(V))) if V is not None else 0.0
    Mmax = float(np.nanmax(np.abs(M))) if M is not None else 0.0
    return (0.0, Mmax, 0.0, Vmax, 0.0)

def cs2_c2_diagram(L, w, E=None, I=None, n=1201):
    """
    Continuous Beam - Two equal spans with UDL on ONE span (left span).
    Spans: L (left), L (right). Supports at x=0, x=L, x=2L.
    Load: UDL w [kN/m] on span 1 only (0..L). No load on span 2.

    Uses Clapeyron (three-moment) for internal support moment:
      M_mid = - w*L^2/16

    Returns:
      x (m), V (kN), M (kN·m), delta (m or None)
    """
    L = float(L)
    w = float(w)
    L = max(1e-9, L)

    Ltot = 2.0 * L
    x = np.linspace(0.0, Ltot, n)

    # Internal support moment at x=L
    M_mid = -(w * L**2) / 16.0  # kN·m (hogging negative)

    # Reactions via span-end moment relations
    # Span 1 (0..L), UDL w, end moments M0=0, M1=M_mid:
    # M(L) = M0 + R1*L - w*L^2/2  => R1 = (M1 - M0 + w*L^2/2)/L
    R1 = (M_mid + w * L**2 / 2.0) / L          # kN
    R2_left = w * L - R1                        # kN (middle support contribution from left span)

    # Span 2 (L..2L), no load, end moments M1=M_mid, M2=0:
    # 0 = M_mid + R2_right*L  => R2_right = -M_mid/L
    R2_right = -M_mid / L                       # kN (middle support contribution from right span)
    R3 = -R2_right                              # kN (right support reaction)

    # Total middle support reaction:
    R2 = R2_left + R2_right

    # Build V and M piecewise
    V = np.zeros_like(x, dtype=float)
    M = np.zeros_like(x, dtype=float)

    # Span 1: 0..L
    m1 = (x <= L)
    xx = x[m1]
    V[m1] = R1 - w * xx
    M[m1] = 0.0 + R1 * xx - (w * xx**2) / 2.0

    # Span 2: L..2L
    m2 = ~m1
    xx = x[m2] - L  # local coordinate from middle support
    V[m2] = R2_right  # constant (no distributed load)
    M[m2] = M_mid + R2_right * xx

    # Deflection (use your FE helper, simple supports at 0, L, 2L)
    delta = None
    if E and I and I > 0:
        xs, vs = _beam2span_delta_fe(L, L, E, I, w1_Nm=w*1000.0, w2_Nm=0.0, point_loads=None)
        delta = np.interp(x, xs, vs)

    return x, V, M, delta
    
# ================================
# CS2 - CASE 3 (F1, F2 at midspans)
# ================================
def cs2_c3_case(a, b, F1, F2):
    """
    Continuous beam (2 spans / 3 supports) - Case 3:
    Two unequal spans with point loads at the center of each span.
    Inputs: a (m), b (m), F1 (kN), F2 (kN)
    """
    x, V, M, _ = cs2_c3_diagram(a, b, F1, F2, E=None, I=None, n=1201)
    Vmax = float(np.nanmax(np.abs(V))) if V is not None else 0.0
    Mmax = float(np.nanmax(np.abs(M))) if M is not None else 0.0
    return (0.0, Mmax, 0.0, Vmax, 0.0)


def cs2_c3_diagram(a, b, F1, F2, E=None, I=None, n=1201):
    """
    Continuous Beam - Two Unequal Spans with Point Loads central to each span.
    Spans: a (left), b (right). Supports at x=0, x=a, x=a+b.
    Loads: F1 at x=a/2, F2 at x=a + b/2.

    Returns: x (m), V (kN), M (kN·m), delta (m or None)
    """
    a = float(a); b = float(b); F1 = float(F1); F2 = float(F2)
    a = max(1e-9, a); b = max(1e-9, b)
    L = a + b

    xF1 = a / 2.0
    xF2 = a + b / 2.0

    x = np.linspace(0.0, L, n)

    # From your sheet:
    M2 = -(3.0 / 16.0) * (F1 * a**2 + F2 * b**2) / (a + b)

    R1 = M2 / a + F1 / 2.0
    R3 = M2 / b + F2 / 2.0
    R2 = F1 + F2 - R1 - R3

    H_a  = (x >= a).astype(float)
    H_L  = (x >= L).astype(float)
    H_F1 = (x >= xF1).astype(float)
    H_F2 = (x >= xF2).astype(float)

    V = R1 + R2 * H_a + R3 * H_L - F1 * H_F1 - F2 * H_F2
    M = (R1 * x
         + R2 * (x - a) * H_a
         + R3 * (x - L) * H_L
         - F1 * (x - xF1) * H_F1
         - F2 * (x - xF2) * H_F2)

    delta = None
    if E and I and I > 0:
        pls = [
            (0, a / 2.0, F1 * 1000.0),
            (1, b / 2.0, F2 * 1000.0),
        ]
        xs, vs = _beam2span_delta_fe(a, b, E, I, w1_Nm=0.0, w2_Nm=0.0, point_loads=pls)
        delta = np.interp(x, xs, vs)

    return x, V, M, delta
def cs3_c1_case(L, w1, w2, w3):
    x, V, M, _ = cs3_c1_diagram(L, w1, w2, w3, E=None, I=None, n=1601)
    Vmax = float(np.nanmax(np.abs(V))) if V is not None else 0.0
    Mmax = float(np.nanmax(np.abs(M))) if M is not None else 0.0
    return (0.0, Mmax, 0.0, Vmax, 0.0)


def cs3_c1_diagram(L, w1, w2, w3, E=None, I=None, n=1601):
    """
    Continuous beam: 3 equal spans (L each), 4 simple supports at x=0, L, 2L, 3L
    UDLs: w1 on span 1, w2 on span 2, w3 on span 3  (kN/m)

    Unknown internal support moments: M1 at x=L, M2 at x=2L
    End moments at x=0 and x=3L are zero.

    Returns: x (m), V (kN), M (kN·m), delta (m or None)
    """
    L = float(L)
    L = max(1e-9, L)
    w1 = float(w1); w2 = float(w2); w3 = float(w3)

    # ---- Solve for internal support moments using slope-deflection / three-moment equivalent ----
    # For equal spans and prismatic beam with UDL on a span:
    # Fixed-end moments for UDL on span i: FEM_left = -wL^2/12, FEM_right = +wL^2/12 (sign conv)
    # We use standard continuous-beam equations at internal joints:
    #
    # Joint at x=L:   4*M1 + 1*M2 = - (FEM_right(span1) + FEM_left(span2))*2
    # Joint at x=2L:  1*M1 + 4*M2 = - (FEM_right(span2) + FEM_left(span3))*2
    #
    # This is a compact form of the slope-deflection joint equilibrium for equal spans with far ends pinned.
    #
    # FEM_right(span i) = + w_i*L^2/12
    # FEM_left (span i) = - w_i*L^2/12
    #
    # So:
    # RHS1 = -2*( +w1 L^2/12  + (-w2 L^2/12) ) = - (w1 - w2) * L^2/6
    # RHS2 = -2*( +w2 L^2/12  + (-w3 L^2/12) ) = - (w2 - w3) * L^2/6
    rhs1 = - (w1 - w2) * (L**2) / 6.0
    rhs2 = - (w2 - w3) * (L**2) / 6.0

    A = np.array([[4.0, 1.0],
                  [1.0, 4.0]], dtype=float)
    b = np.array([rhs1, rhs2], dtype=float)

    M1, M2 = np.linalg.solve(A, b)  # moments at x=L and x=2L (kN·m)

    # End moments at outer supports (pinned)
    M0 = 0.0
    M3 = 0.0

    # ---- Reactions per span from end moments ----
    # For span with UDL w and end moments Ma (left), Mb (right):
    # Shear at left end:  Ra = wL/2 + (Mb - Ma)/L
    # Shear at right end: Rb = wL - Ra
    def span_end_reactions(w, Ma, Mb):
        Ra = w * L / 2.0 + (Mb - Ma) / L
        Rb = w * L - Ra
        return Ra, Rb

    # span 1: 0..L
    R0_1, R1_1 = span_end_reactions(w1, M0, M1)
    # span 2: L..2L
    R1_2, R2_2 = span_end_reactions(w2, M1, M2)
    # span 3: 2L..3L
    R2_3, R3_3 = span_end_reactions(w3, M2, M3)

    # total support reactions (sum contributions from adjacent spans)
    R1 = R0_1
    R2 = R1_1 + R1_2
    R3 = R2_2 + R2_3
    R4 = R3_3

    # ---- Build diagrams along x ----
    Ltot = 3.0 * L
    x = np.linspace(0.0, Ltot, n)

    V = np.zeros_like(x, dtype=float)
    M = np.zeros_like(x, dtype=float)

    # Span 1
    m1 = (x <= L)
    xx = x[m1]
    V[m1] = R1 - w1 * xx
    M[m1] = M0 + R1 * xx - (w1 * xx**2) / 2.0

    # Span 2
    m2 = (x > L) & (x <= 2.0 * L)
    xx = x[m2] - L
    V[m2] = (R2) - w2 * xx  # shear just to the right of support 2 includes R2
    M[m2] = M1 + R2 * xx - (w2 * xx**2) / 2.0

    # Span 3
    m3 = (x > 2.0 * L)
    xx = x[m3] - 2.0 * L
    V[m3] = (R3) - w3 * xx
    M[m3] = M2 + R3 * xx - (w3 * xx**2) / 2.0

    # ---- Deflection (optional) ----
    delta = None
    if E and I and I > 0:
        # If you already have a 3-span FE helper, use it here.
        # Otherwise, set delta=None and the app will still show max deflection only where available.
        if "_beam3span_delta_fe" in globals():
            xs, vs = _beam3span_delta_fe(L, L, L, E, I, w1_Nm=w1*1000.0, w2_Nm=w2*1000.0, w3_Nm=w3*1000.0)
            delta = np.interp(x, xs, vs)

    return x, V, M, delta

def cs4_c1_case(L, w1, w2, w3, w4):
    x, V, M, _ = cs4_c1_diagram(L, w1, w2, w3, w4, E=None, I=None, n=2001)
    Vmax = float(np.nanmax(np.abs(V))) if V is not None else 0.0
    Mmax = float(np.nanmax(np.abs(M))) if M is not None else 0.0
    return (0.0, Mmax, 0.0, Vmax, 0.0)


def cs4_c1_diagram(L, w1, w2, w3, w4, E=None, I=None, n=2001):
    """
    Continuous beam: 4 equal spans (L each), 5 supports at x=0, L, 2L, 3L, 4L
    UDLs: w1 on span 1, w2 on span 2, w3 on span 3, w4 on span 4 (kN/m)

    Unknown internal support moments:
      M1 at x=L, M2 at x=2L, M3 at x=3L
    End moments at x=0 and x=4L are zero (pinned ends).

    Returns: x (m), V (kN), M (kN·m), delta (m or None)
    """
    L = float(L)
    L = max(1e-9, L)
    w1 = float(w1); w2 = float(w2); w3 = float(w3); w4 = float(w4)

    # -----------------------------
    # 1) Solve internal moments M1,M2,M3
    # -----------------------------
    # Same compact joint-equilibrium form used in CS3 (equal spans, pinned outer ends):
    #   4*M1 + 1*M2       = -(w1 - w2) * L^2 / 6
    #   1*M1 + 4*M2 + 1*M3 = -(w2 - w3) * L^2 / 6
    #         1*M2 + 4*M3  = -(w3 - w4) * L^2 / 6
    rhs1 = - (w1 - w2) * (L**2) / 6.0
    rhs2 = - (w2 - w3) * (L**2) / 6.0
    rhs3 = - (w3 - w4) * (L**2) / 6.0

    A = np.array([
        [4.0, 1.0, 0.0],
        [1.0, 4.0, 1.0],
        [0.0, 1.0, 4.0],
    ], dtype=float)
    b = np.array([rhs1, rhs2, rhs3], dtype=float)

    M1, M2, M3 = np.linalg.solve(A, b)  # kN·m at x=L,2L,3L

    # end moments at outer supports
    M0 = 0.0
    M4 = 0.0

    # -----------------------------
    # 2) Span end reactions from end moments
    # -----------------------------
    # For span with UDL w and end moments Ma (left), Mb (right):
    #   Ra = wL/2 + (Mb - Ma)/L
    #   Rb = wL - Ra
    def span_end_reactions(w, Ma, Mb):
        Ra = w * L / 2.0 + (Mb - Ma) / L
        Rb = w * L - Ra
        return Ra, Rb

    # span 1 (0..L): moments M0 -> M1
    R0_1, R1_1 = span_end_reactions(w1, M0, M1)
    # span 2 (L..2L): moments M1 -> M2
    R1_2, R2_2 = span_end_reactions(w2, M1, M2)
    # span 3 (2L..3L): moments M2 -> M3
    R2_3, R3_3 = span_end_reactions(w3, M2, M3)
    # span 4 (3L..4L): moments M3 -> M4
    R3_4, R4_4 = span_end_reactions(w4, M3, M4)

    # total support reactions (sum adjacent-span contributions)
    R1 = R0_1
    R2 = R1_1 + R1_2
    R3 = R2_2 + R2_3
    R4 = R3_3 + R3_4
    R5 = R4_4

    # -----------------------------
    # 3) Build V(x), M(x) piecewise
    # -----------------------------
    Ltot = 4.0 * L
    x = np.linspace(0.0, Ltot, n)

    V = np.zeros_like(x, dtype=float)
    M = np.zeros_like(x, dtype=float)

    # Span 1: 0..L
    s1 = (x <= L)
    xx = x[s1]
    V[s1] = R1 - w1 * xx
    M[s1] = M0 + R1 * xx - (w1 * xx**2) / 2.0

    # Span 2: L..2L
    s2 = (x > L) & (x <= 2.0 * L)
    xx = x[s2] - L
    V[s2] = R2 - w2 * xx
    M[s2] = M1 + R2 * xx - (w2 * xx**2) / 2.0

    # Span 3: 2L..3L
    s3 = (x > 2.0 * L) & (x <= 3.0 * L)
    xx = x[s3] - 2.0 * L
    V[s3] = R3 - w3 * xx
    M[s3] = M2 + R3 * xx - (w3 * xx**2) / 2.0

    # Span 4: 3L..4L
    s4 = (x > 3.0 * L)
    xx = x[s4] - 3.0 * L
    V[s4] = R4 - w4 * xx
    M[s4] = M3 + R4 * xx - (w4 * xx**2) / 2.0

    # -----------------------------
    # 4) Deflection (optional)
    # -----------------------------
    delta = None
    if E and I and I > 0:
        # If you have a 4-span FE helper, use it.
        if "_beam4span_delta_fe" in globals():
            xs, vs = _beam4span_delta_fe(L, L, L, L, E, I,
                                         w1_Nm=w1*1000.0, w2_Nm=w2*1000.0,
                                         w3_Nm=w3*1000.0, w4_Nm=w4*1000.0)
            delta = np.interp(x, xs, vs)

    return x, V, M, delta


# ================================
# READY CATALOG (DATA ONLY)
# ================================
READY_CATALOG = {
    "Beam": {
        # Category 1: 5 cases
        "Simply Supported Beams (5 cases)": make_cases(
            "SS", 5, {"L": 6.0, "w": 10.0}
        ),

        # Category 2: 1 case
        "Beams Fixed at one end (1 case)": make_cases(
            "FE", 1, {"L": 6.0, "w": 10.0}
        ),

        # Category 3: 1 case
        "Beams Fixed at both ends (1 case)": make_cases(
            "FB", 1, {"L_mm": 6000.0, "w": 10.0, "a": 2.0, "F": 20.0}
        ),

        # Category 4: 1 case
        "Cantilever Beams (1 case)": make_cases(
            "C", 1, {"L_mm": 3000.0, "w": 10.0, "a": 1.5, "F": 20.0, "M": 0.0}
        ),

        # Category 5: 4 cases
        "Beams with Overhang (4 cases)": make_cases(
            "OH", 4, {"L_mm": 6000.0, "a": 1.5, "w1": 10.0, "w2": 10.0}
        ),

        # Category 6: 3 cases (patched below)
        "Continuous Beams — Two Spans / Three Supports (3 cases)": make_cases(
            "CS2", 3, {}
        ),

        # Category 7: 1 case
        "Continuous Beams — Three Spans / Four Supports (1 case)": make_cases(
            "CS3", 1, {"L1": 4.0, "L2": 4.0, "L3": 4.0, "w": 10.0}
        ),

        # Category 8: 1 case
        "Continuous Beams — Four Spans / Five Supports (1 case)": make_cases(
            "CS4", 1, {"L1": 4.0, "L2": 4.0, "L3": 4.0, "L4": 4.0, "w": 10.0}
        ),
    }
}


# ================================
# PATCH CS2 (AFTER READY_CATALOG EXISTS)
# ================================
_cat = "Continuous Beams — Two Spans / Three Supports (3 cases)"
_cases = READY_CATALOG["Beam"][_cat]

# Case 1: Two Unequal Spans with UDL
_cases[0]["label"] = "CS2 - C1 (Unequal spans + UDL)"
_cases[0]["inputs"] = {"a": 4.0, "b": 6.0, "w": 10.0}
_cases[0]["func"] = cs2_c1_case
_cases[0]["diagram_func"] = cs2_c1_diagram

# Case 2: Two equal spans, UDL on one span
_cases[1]["label"] = "CS2 - C2 (One span UDL)"
_cases[1]["inputs"] = {"L": 5.0, "w": 10.0}
_cases[1]["func"] = cs2_c2_case
_cases[1]["diagram_func"] = cs2_c2_diagram

# Case 3: Two Unequal Spans with central point loads
_cases[2]["label"] = "CS2 - C3 (Central point loads)"
_cases[2]["inputs"] = {"a": 4.0, "b": 6.0, "F1": 20.0, "F2": 20.0}
_cases[2]["func"] = cs2_c3_case
_cases[2]["diagram_func"] = cs2_c3_diagram


# Image mapping for ready beam cases (only cases 1 and 2 for now)
# Image mapping for ready beam cases
CASE_IMAGE_MAP = {
    # --- Simply Supported Beams (KEEP exactly as you already had) ---
    "SS-01": "beam_case_img/SSB-C1.png",
    "SS-02": "beam_case_img/SSB-C2.png",
    "SS-03": "beam_case_img/SSB-C3.png",
    "SS-04": "beam_case_img/SSB-C4.png",
    "SS-05": "beam_case_img/SSB-C5.png",

    # --- Beams Fixed at one end (1 case) ---
    "FE-01": "beam_case_img/FEB-C1.png",

    # --- Beams Fixed at both ends (1 case) ---
    "FB-01": "beam_case_img/FSB-C1.png",

    # --- Cantilever Beams (you told me 1 case = CB-C1.png) ---
    # NOTE: your catalog currently defines Cantilever as (2 cases) with prefix "C"
    # so mapping is for C-01.
    "C-01": "beam_case_img/CB-C1.png",

    # --- Beams with Overhang (4 cases) ---
    # NOTE: your catalog prefix is "OH"
    "OH-01": "beam_case_img/OB-C1.png",
    "OH-02": "beam_case_img/OB-C2.png",
    "OH-03": "beam_case_img/OB-C3.png",
    "OH-04": "beam_case_img/OB-C4.png",

    # --- Continuous Beams — Two Spans / Three Supports ---
    # NOTE: your catalog currently defines 2 cases (CS2-01, CS2-02)
    # You provided 3 images; add 3rd when you increase to 3 cases.
    "CS2-01": "beam_case_img/2S-C1.png",
    "CS2-02": "beam_case_img/2S-C2.png",
    "CS2-03": "beam_case_img/2S-C3.png",

    # --- Continuous Beams — Three Spans / Four Supports (1 case) ---
    "CS3-01": "beam_case_img/3S-C1.png",

    # --- Continuous Beams — Four Spans / Five Supports (1 case) ---
    "CS4-01": "beam_case_img/4S-C1.png",
}

# Apply image paths to cases
if "Beam" in READY_CATALOG:
    for category, cases in READY_CATALOG["Beam"].items():
        for case in cases:
            key = case.get("key")
            if key in CASE_IMAGE_MAP:
                case["img_path"] = CASE_IMAGE_MAP[key]


# ============================================================
# Patch: Simply Supported Beams (5 cases)  -> ALL length inputs in mm
# ============================================================
_cat = "Simply Supported Beams (5 cases)"
_cases = READY_CATALOG["Beam"][_cat]

# Case 1: UDL full span
_cases[0]["label"] = "SSB - C1 (UDL full span)"
_cases[0]["inputs"] = {"L_mm": 6000.0, "w": 10.0}
_cases[0]["func"] = ss_udl_case
_cases[0]["diagram_func"] = ss_udl_diagram
_cases[0]["delta_max_func"] = ss_udl_deflection_max

# Case 2: Central point load
_cases[1]["label"] = "SSB - C2 (Central point load)"
_cases[1]["inputs"] = {"L_mm": 6000.0, "F": 20.0}
_cases[1]["func"] = ss_central_point_case
_cases[1]["diagram_func"] = ss_central_point_diagram
_cases[1]["delta_max_func"] = ss_central_point_deflection_max

# Case 3: Two point loads + partial UDL
_cases[2]["label"] = "SSB - C3 (F1+F2 + partial UDL)"
_cases[2]["inputs"] = {
    "L_mm": 6000.0,
    "F1": 20.0,
    "a1": 1500.0,  # mm from LEFT support
    "F2": 25.0,
    "a2": 1000.0,  # mm from RIGHT support
    "w": 8.0,
    "a": 2000.0,   # mm from LEFT to start of partial UDL
    "b": 1500.0,   # mm length of partial UDL
}
_cases[2]["func"] = ssb_c3_case
_cases[2]["diagram_func"] = ssb_c3_diagram

# Case 4: Two partial UDLs (your existing SSB-C4 but in mm)
_cases[3]["label"] = "SSB - C4 (Two partial UDLs)"
_cases[3]["inputs"] = {"L_mm": 6000.0, "a": 1500.0, "b": 1000.0, "c": 1500.0, "w1": 10.0, "w2": 6.0}
_cases[3]["func"] = ssb_c4_case
_cases[3]["diagram_func"] = ssb_c4_diagram

# Case 5: UDL + mid-point load + end moments (now L_mm)
_cases[4]["label"] = "SSB - C5"
_cases[4]["inputs"] = {"L_mm": 6000.0, "w": 10.0, "P": 20.0, "M1": 50.0, "M2": 20.0}
_cases[4]["func"] = ssb_c5_case
_cases[4]["diagram_func"] = ssb_c5_diagram


# ---- Patch "Beams Fixed at one end (1 case)" to be FEB-C1 (propped cantilever: UDL + point load) ----
READY_CATALOG["Beam"]["Beams Fixed at one end (1 case)"][0]["label"] = "FEB - C1"
READY_CATALOG["Beam"]["Beams Fixed at one end (1 case)"][0]["inputs"] = {
    "L_mm": 6000.0,
    "w": 10.0,
    "a": 2.0,
    "F": 20.0
}
READY_CATALOG["Beam"]["Beams Fixed at one end (1 case)"][0]["func"] = feb_c1_case
READY_CATALOG["Beam"]["Beams Fixed at one end (1 case)"][0]["diagram_func"] = feb_c1_diagram

READY_CATALOG["Beam"]["Beams Fixed at both ends (1 case)"][0]["label"] = "FBB - C1"
READY_CATALOG["Beam"]["Beams Fixed at both ends (1 case)"][0]["func"] = fbb_c1_case
READY_CATALOG["Beam"]["Beams Fixed at both ends (1 case)"][0]["diagram_func"] = fbb_c1_diagram

READY_CATALOG["Beam"]["Cantilever Beams (1 case)"][0]["label"] = "CB - C1"
READY_CATALOG["Beam"]["Cantilever Beams (1 case)"][0]["func"] = cant_c1_case
READY_CATALOG["Beam"]["Cantilever Beams (1 case)"][0]["diagram_func"] = cant_c1_diagram
READY_CATALOG["Beam"]["Cantilever Beams (1 case)"][0]["delta_max_func"] = cant_c1_delta_max

READY_CATALOG["Beam"]["Beams with Overhang (4 cases)"][0]["label"] = "OH - C1"
READY_CATALOG["Beam"]["Beams with Overhang (4 cases)"][0]["func"] = oh_c1_case
READY_CATALOG["Beam"]["Beams with Overhang (4 cases)"][0]["diagram_func"] = oh_c1_diagram
READY_CATALOG["Beam"]["Beams with Overhang (4 cases)"][0]["inputs"] = {"L_mm": 6000.0, "a": 1.5, "w1": 10.0, "w2": 10.0}

READY_CATALOG["Beam"]["Beams with Overhang (4 cases)"][1]["label"] = "OH - C2"
READY_CATALOG["Beam"]["Beams with Overhang (4 cases)"][1]["func"] = oh_c2_case
READY_CATALOG["Beam"]["Beams with Overhang (4 cases)"][1]["diagram_func"] = oh_c2_diagram
READY_CATALOG["Beam"]["Beams with Overhang (4 cases)"][1]["inputs"] = {"L_mm": 6000.0, "a": 1.5, "F": 20.0}

READY_CATALOG["Beam"]["Beams with Overhang (4 cases)"][2]["label"] = "OH - C3"
READY_CATALOG["Beam"]["Beams with Overhang (4 cases)"][2]["func"] = oh_c3_case
READY_CATALOG["Beam"]["Beams with Overhang (4 cases)"][2]["diagram_func"] = oh_c3_diagram
READY_CATALOG["Beam"]["Beams with Overhang (4 cases)"][2]["inputs"] = {
    "L_mm": 6000.0,
    "a": 2.0,
    "F": 20.0
}

READY_CATALOG["Beam"]["Beams with Overhang (4 cases)"][3]["label"] = "OH - C4"
READY_CATALOG["Beam"]["Beams with Overhang (4 cases)"][3]["func"] = oh_c4_case
READY_CATALOG["Beam"]["Beams with Overhang (4 cases)"][3]["diagram_func"] = oh_c4_diagram
READY_CATALOG["Beam"]["Beams with Overhang (4 cases)"][3]["inputs"] = {"a": 1.0, "b": 6.0, "c": 1.0, "w": 10.0}

READY_CATALOG["Beam"]["Continuous Beams — Three Spans / Four Supports (1 case)"][0]["label"] = "CS3 - C1 (3 spans UDL)"
READY_CATALOG["Beam"]["Continuous Beams — Three Spans / Four Supports (1 case)"][0]["inputs"] = {"L": 5.0, "w1": 10.0, "w2": 10.0, "w3": 10.0}
READY_CATALOG["Beam"]["Continuous Beams — Three Spans / Four Supports (1 case)"][0]["func"] = cs3_c1_case
READY_CATALOG["Beam"]["Continuous Beams — Three Spans / Four Supports (1 case)"][0]["diagram_func"] = cs3_c1_diagram

READY_CATALOG["Beam"]["Continuous Beams — Four Spans / Five Supports (1 case)"][0]["label"] = "CS4 - C1 (4 spans UDL)"
READY_CATALOG["Beam"]["Continuous Beams — Four Spans / Five Supports (1 case)"][0]["inputs"] = {
    "L": 5.0,
    "w1": 10.0,
    "w2": 10.0,
    "w3": 10.0,
    "w4": 10.0,
}
READY_CATALOG["Beam"]["Continuous Beams — Four Spans / Five Supports (1 case)"][0]["func"] = cs4_c1_case
READY_CATALOG["Beam"]["Continuous Beams — Four Spans / Five Supports (1 case)"][0]["diagram_func"] = cs4_c1_diagram

def compute_delta_max_from_curve(delta):
    """Return max |delta| in meters from a deflection array (or None)."""
    if delta is None:
        return None
    try:
        val = float(np.nanmax(np.abs(delta)))
        if np.isnan(val):
            return None
        return val
    except Exception:
        return None

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
        
        # Map keys -> nice labels
        label_map = {
            "L_mm": "L (mm)",
            "L": "L (m)",
            "w": "w (kN/m)",
            "a1": "a1 (mm from LEFT)",
            "a2": "a2 (mm from RIGHT)",
            "a": "a (mm)",
            "b": "b (mm)",
            "c": "c (mm)",
            "F": "F (kN)",
            "F1": "F1 (kN)",
            "F2": "F2 (kN)",
            "M": "M (kN·m)",
            "w1": "w1 (kN/m)",
            "w2": "w2 (kN/m)",
            "w3": "w3 (kN/m)",
            "w4": "w4 (kN/m)",
        }
        
        for i in range(0, len(keys), 3):
            cols = st.columns(3)
            for col, k in zip(cols, keys[i : i + 3]):
                with col:
                    label = label_map.get(k, k)
                    input_vals[k] = st.number_input(
                        label,
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
            L_in_m = float(input_vals.get("L", 0.0))
            if (not L_in_m or L_in_m <= 0.0) and ("L_mm" in input_vals):
                L_in_m = float(input_vals["L_mm"]) / 1000.0
            
            st.session_state["L_in"] = L_in_m
            st.session_state["L_mm_in"] = L_in_m * 1000.0

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
        material = st.selectbox("Material", ["S235", "S275", "S355", "S450"], index=2, key="mat_sel")
        st.session_state["material"] = material  # keep consistent with report tab
        # If material changed -> invalidate previous results so Report/Results can't show old run
        prev_mat = st.session_state.get("_prev_material_for_results", None)
        if prev_mat is None:
            st.session_state["_prev_material_for_results"] = material
        elif prev_mat != material:
            st.session_state["_prev_material_for_results"] = material
        
            # Clear computed results (they depend on fy)
            for k in ["run_clicked", "df_rows", "overall_ok", "governing", "extras"]:
                st.session_state.pop(k, None)

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
            
                # Force refresh of stored section properties every rerun
                if selected_row is not None:
                    st.session_state["sr_display"] = selected_row
                    st.session_state["fam_sel_last"] = family
                    st.session_state["size_sel_last"] = selected_name
                    st.session_state["detected_table_last"] = detected_table
                else:
                    st.session_state["sr_display"] = None

                # ---- Force update sr_display if selection changed OR db_rev changed ----
                # Key includes db_rev so the same selection can be reloaded after DB refresh
                sr_key = f"{family}::{selected_name}::{detected_table}::{st.session_state.get('db_rev', 1)}"
                prev_key = st.session_state.get("_sr_key_loaded")

                if (prev_key != sr_key) or (st.session_state.get("sr_display") is None):
                    st.session_state["sr_display"] = selected_row
                    st.session_state["_sr_key_loaded"] = sr_key

                    # Clear computed results so they recompute with the new section properties
                    for k in ["df_rows", "overall_ok", "governing", "extras"]:
                        st.session_state.pop(k, None)

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
def store_design_forces_from_state():
    """Compute design ULS forces from current Loads inputs in session_state
    and store them into st.session_state['inputs']. This replaces the old
    Run button inside the Loads tab; it is now triggered from the Results tab.
    """
    # Raw inputs from Loads form
    L = float(st.session_state.get("L_mm_in", 0.0)) / 1000.0  # mm → m (internal calculations use m)
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
    st.caption("Positive N = tension.")

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
    I_y_m4 = (use_props.get("Iy_cm4", use_props.get("I_y_cm4", 0.0))) * 1e-8
    I_z_m4 = (use_props.get("Iz_cm4", use_props.get("I_z_cm4", 0.0))) * 1e-8
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

    # -------------------------------------------------
    # Basic resistances (cross-section)
    # -------------------------------------------------
    N_Rd_N = A_m2 * fy * 1e6 / gamma_M0
    T_Rd_N = N_Rd_N

    # Plastic (preferred) section modulus in m^3; fallback to 1.1*S (as you had)
    Wpl_y_m3 = (Wpl_y_cm3 * 1e-6) if Wpl_y_cm3 > 0 else 1.1 * S_y_m3
    Wpl_z_m3 = (Wpl_z_cm3 * 1e-6) if Wpl_z_cm3 > 0 else 1.1 * S_z_m3

    M_Rd_y_Nm = Wpl_y_m3 * fy * 1e6 / gamma_M0
    M_Rd_z_Nm = Wpl_z_m3 * fy * 1e6 / gamma_M0  # <-- needed for §6.2.9/§6.2.10

    # For §6.2.9 / §6.2.10 we need plastic bending resistances in kNm
    Mpl_Rd_y_kNm = M_Rd_y_Nm / 1e3
    Mpl_Rd_z_kNm = M_Rd_z_Nm / 1e3

    Av_m2 = 0.6 * A_m2
    V_Rd_N = Av_m2 * fy * 1e6 / (math.sqrt(3) * gamma_M0)

    # Allowable stress for indicative combined check
    sigma_allow_MPa = 0.6 * fy

    # -------------------------------------------------
    # Stresses (indicative)
    # -------------------------------------------------
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
    # Combined effects (checks 7–14) — compute real u(9–14) consistently with Report tab
    # -------------------------------------------------

    # Shear influence ratios (EN 1993-1-1 §6.2.8) used later in §6.2.10
    shear_ratio_y = (Vy_Ed_kN / Vc_y_Rd_kN) if (Vc_y_Rd_kN and Vc_y_Rd_kN > 0) else None
    shear_ratio_z = (Vz_Ed_kN / Vc_z_Rd_kN) if (Vc_z_Rd_kN and Vc_z_Rd_kN > 0) else None

    # Pull basic geometric dims (needed for §6.2.9 / §6.2.10 formulas)
    h_mm = float(use_props.get("h_mm", use_props.get("h", 0.0)) or 0.0)
    b_mm = float(use_props.get("b_mm", use_props.get("b", 0.0)) or 0.0)
    tw_mm = float(use_props.get("tw_mm", use_props.get("tw", 0.0)) or 0.0)
    tf_mm = float(use_props.get("tf_mm", use_props.get("tf", 0.0)) or 0.0)
    r_mm  = float(use_props.get("r_mm",  use_props.get("r", 0.0))  or 0.0)

    A_mm2 = A_m2 * 1e6
    Npl_Rd_kN = (A_mm2 * fy / gamma_M0) / 1e3 if (A_mm2 > 0 and fy > 0) else 0.0
    NEd_kN = abs(N_kN)

    # Section “type” (used to select the right §6.2.9 / §6.2.10 branch like your Report tab)
    sec_label = (
        str(use_props.get("family") or use_props.get("type") or use_props.get("Type") or
            use_props.get("section_family") or use_props.get("Section") or use_props.get("name") or "")
    ).upper()
    is_rhs = any(k in sec_label for k in ["RHS", "SHS", "HSS", "BOX"])
    is_ih  = any(sec_label.startswith(p) for p in ("IPE", "IPN", "HEA", "HEB", "HEM", "HE", "UB", "UC"))

    # Axial ratio
    n = (NEd_kN / Npl_Rd_kN) if (Npl_Rd_kN and Npl_Rd_kN > 0) else None

    # ------------------------------------------------------------
    # (9)–(11) Bending and axial force — EN 1993-1-1 §6.2.9
    # ------------------------------------------------------------
    MN_y_Rd_6209 = None
    MN_z_Rd_6209 = None
    alpha_y_6209 = None
    alpha_z_6209 = None
    u_6209_9 = None
    u_6209_10 = None
    u_6209_11 = None

    if (n is not None) and (A_mm2 > 0) and (Mpl_Rd_y_kNm > 0) and (Mpl_Rd_z_kNm > 0):
        if is_ih:
            a = (A_mm2 - 2.0 * b_mm * tf_mm) / A_mm2
            a = max(0.0, min(0.5, a))

            denom_y = (1.0 - 0.5 * a)
            MN_y_Rd_6209 = Mpl_Rd_y_kNm * (1.0 - n) / denom_y if denom_y > 0 else 0.0
            MN_y_Rd_6209 = min(Mpl_Rd_y_kNm, max(0.0, MN_y_Rd_6209))

            if n <= a:
                MN_z_Rd_6209 = Mpl_Rd_z_kNm
            else:
                ratio = (n - a) / (1.0 - a) if (1.0 - a) > 0 else 1.0
                MN_z_Rd_6209 = Mpl_Rd_z_kNm * (1.0 - ratio**2)
                MN_z_Rd_6209 = max(0.0, MN_z_Rd_6209)

            alpha_y_6209 = 2.0
            alpha_z_6209 = max(1.0, 5.0 * n)

        elif is_rhs:
            aw = (A_mm2 - 2.0 * b_mm * tf_mm) / A_mm2
            af = (A_mm2 - 2.0 * h_mm * tw_mm) / A_mm2
            aw = max(0.0, min(0.5, aw))
            af = max(0.0, min(0.5, af))

            denom_y = (1.0 - 0.5 * aw)
            denom_z = (1.0 - 0.5 * af)

            MN_y_Rd_6209 = Mpl_Rd_y_kNm * (1.0 - n) / denom_y if denom_y > 0 else 0.0
            MN_z_Rd_6209 = Mpl_Rd_z_kNm * (1.0 - n) / denom_z if denom_z > 0 else 0.0
            MN_y_Rd_6209 = min(Mpl_Rd_y_kNm, max(0.0, MN_y_Rd_6209))
            MN_z_Rd_6209 = min(Mpl_Rd_z_kNm, max(0.0, MN_z_Rd_6209))

            if n <= 0.8:
                denom = (1.0 - 1.13 * (n**2))
                alpha = min(6.0, 1.66 / denom) if denom > 0 else 6.0
            else:
                alpha = 6.0
            alpha_y_6209 = alpha
            alpha_z_6209 = alpha

        if (MN_y_Rd_6209 is not None) and (MN_z_Rd_6209 is not None) and (MN_y_Rd_6209 > 0) and (MN_z_Rd_6209 > 0):
            uMy = abs(My_Ed_kNm) / MN_y_Rd_6209   # this is uy in your Report tab
            uMz = abs(Mz_Ed_kNm) / MN_z_Rd_6209   # this is uz in your Report tab
        
            # (9) and (10): show the axis utilizations (NOT raised to alpha)
            u_6209_9  = uMy
            u_6209_10 = uMz
        
            # (11): show the interaction (Eq. 8.56 form)
            u_6209_11 = (uMy ** alpha_y_6209) + (uMz ** alpha_z_6209)


    # ------------------------------------------------------------
    # (12)–(14) Bending, shear and axial force — EN 1993-1-1 §6.2.10
    # ------------------------------------------------------------
    eta_gov = 0.0
    if shear_ratio_y is not None:
        eta_gov = max(eta_gov, float(abs(shear_ratio_y)))
    if shear_ratio_z is not None:
        eta_gov = max(eta_gov, float(abs(shear_ratio_z)))

    shear_small_6210 = True if eta_gov is None else (eta_gov <= 0.50)

    rho_6210 = 0.0
    fy_red_6210 = fy
    if not shear_small_6210:
        rho_6210 = max(0.0, (2.0 * eta_gov - 1.0) ** 2)
        rho_6210 = min(rho_6210, 1.0)
        fy_red_6210 = max(0.0, (1.0 - rho_6210) * fy)

    scale = (fy_red_6210 / fy) if (fy > 0 and not shear_small_6210) else 1.0

    MN_y_Rd_6210 = None
    MN_z_Rd_6210 = None
    alpha_y_6210 = None
    alpha_z_6210 = None
    u_6210_12 = None
    u_6210_13 = None
    u_6210_14 = None

    if (n is not None) and (A_mm2 > 0) and (Mpl_Rd_y_kNm > 0) and (Mpl_Rd_z_kNm > 0):
        Mpl_y_use = Mpl_Rd_y_kNm * scale
        Mpl_z_use = Mpl_Rd_z_kNm * scale

        if is_ih:
            a = (A_mm2 - 2.0 * b_mm * tf_mm) / A_mm2
            a = max(0.0, min(0.5, a))

            denom_y = (1.0 - 0.5 * a)
            MN_y_Rd_6210 = Mpl_y_use * (1.0 - n) / denom_y if denom_y > 0 else 0.0
            MN_y_Rd_6210 = min(Mpl_y_use, max(0.0, MN_y_Rd_6210))

            if n <= a:
                MN_z_Rd_6210 = Mpl_z_use
            else:
                ratio = (n - a) / (1.0 - a) if (1.0 - a) > 0 else 1.0
                MN_z_Rd_6210 = Mpl_z_use * (1.0 - ratio**2)
                MN_z_Rd_6210 = max(0.0, MN_z_Rd_6210)

            alpha_y_6210 = 2.0
            alpha_z_6210 = max(1.0, 5.0 * n)

        elif is_rhs:
            aw = (A_mm2 - 2.0 * b_mm * tf_mm) / A_mm2
            af = (A_mm2 - 2.0 * h_mm * tw_mm) / A_mm2
            aw = max(0.0, min(0.5, aw))
            af = max(0.0, min(0.5, af))

            denom_y = (1.0 - 0.5 * aw)
            denom_z = (1.0 - 0.5 * af)

            MN_y_Rd_6210 = Mpl_y_use * (1.0 - n) / denom_y if denom_y > 0 else 0.0
            MN_z_Rd_6210 = Mpl_z_use * (1.0 - n) / denom_z if denom_z > 0 else 0.0
            MN_y_Rd_6210 = min(Mpl_y_use, max(0.0, MN_y_Rd_6210))
            MN_z_Rd_6210 = min(Mpl_z_use, max(0.0, MN_z_Rd_6210))

            if n <= 0.8:
                denom = (1.0 - 1.13 * (n**2))
                alpha = min(6.0, 1.66 / denom) if denom > 0 else 6.0
            else:
                alpha = 6.0
            alpha_y_6210 = alpha
            alpha_z_6210 = alpha

        if (MN_y_Rd_6210 is not None) and (MN_z_Rd_6210 is not None) and (MN_y_Rd_6210 > 0) and (MN_z_Rd_6210 > 0):
            uMy = abs(My_Ed_kNm) / MN_y_Rd_6210
            uMz = abs(Mz_Ed_kNm) / MN_z_Rd_6210

            if shear_small_6210:
                u_6210_12 = uMy
                u_6210_13 = uMz
                u_6210_14 = (uMy ** alpha_y_6210) + (uMz ** alpha_z_6210)
            else:
                uV_y = (abs(Vy_Ed_kN) / Vc_y_Rd_kN) if (Vc_y_Rd_kN and Vc_y_Rd_kN > 0) else 0.0
                uV_z = (abs(Vz_Ed_kN) / Vc_z_Rd_kN) if (Vc_z_Rd_kN and Vc_z_Rd_kN > 0) else 0.0
                uV = (uV_y ** 2) + (uV_z ** 2)

                u_6210_12 = (uMy ** alpha_y_6210) + uV
                u_6210_13 = (uMz ** alpha_z_6210) + uV
                u_6210_14 = (uMy ** alpha_y_6210) + (uMz ** alpha_z_6210) + uV

    # Prepare detail values for report + Results table (now includes real u(9–14))
    cs_combo = dict(
        shear_ratio_y=shear_ratio_y,
        shear_ratio_z=shear_ratio_z,
        Npl_Rd_kN=Npl_Rd_kN,
        NEd_kN=NEd_kN,
        util_My=util_My,
        util_Mz=util_Mz,

        # §6.2.9
        n_6209=n,
        MN_y_Rd_6209=MN_y_Rd_6209,
        MN_z_Rd_6209=MN_z_Rd_6209,
        alpha_y_6209=alpha_y_6209,
        alpha_z_6209=alpha_z_6209,
        u_6209_9=u_6209_9,
        u_6209_10=u_6209_10,
        u_6209_11=u_6209_11,

        # §6.2.10
        shear_small_6210=shear_small_6210,
        rho_6210=rho_6210,
        fy_red_6210=fy_red_6210,
        MN_y_Rd_6210=MN_y_Rd_6210,
        MN_z_Rd_6210=MN_z_Rd_6210,
        alpha_y_6210=alpha_y_6210,
        alpha_z_6210=alpha_z_6210,
        u_6210_12=u_6210_12,
        u_6210_13=u_6210_13,
        u_6210_14=u_6210_14,
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

    # Determine imperfection factors for FLEXURAL buckling (EN 1993-1-1 §6.3.1)
    alpha_y = float(alpha_curve_db) if alpha_curve_db is not None else 0.49
    alpha_z = float(alpha_curve_db) if alpha_curve_db is not None else 0.49

    # --- LTB imperfection factor alpha_LT (EN 1993-1-1 Table 6.3 + 6.4) ---
    # Rolled I:   h/b <= 2 -> a (0.21),  h/b > 2 -> b (0.34)
    # Welded I:   h/b <= 2 -> c (0.49),  h/b > 2 -> d (0.76)
    # Other:      d (0.76)
    hb = (h_mm / b_mm) if (b_mm and b_mm > 0) else None

    sec_label = (
        str(use_props.get("family") or use_props.get("type") or use_props.get("Type") or
            use_props.get("section_family") or use_props.get("Section") or use_props.get("name") or "")
    ).upper()

    is_welded_i = ("WELD" in sec_label) or ("PLATE" in sec_label) or ("GIRDER" in sec_label)
    is_rolled_i = any(sec_label.startswith(p) for p in ("IPE", "IPN", "HEA", "HEB", "HEM", "HE", "UB", "UC"))

    if hb is None:
        curve_LT = "b"  # safe default
    elif is_welded_i:
        curve_LT = "c" if hb <= 2.0 else "d"
    elif is_rolled_i:
        # NOTE: your earlier correction: rolled I with h/b <= 2.0 -> curve a (0.21), not b
        curve_LT = "a" if hb <= 2.0 else "b"
    else:
        curve_LT = "d"

    alpha_LT = {"a": 0.21, "b": 0.34, "c": 0.49, "d": 0.76}[curve_LT]

    def chi_reduction(lambda_bar: float, alpha: float) -> float:
        # EN 1993-1-1 §6.3.1.2
        phi = 0.5 * (1.0 + alpha * (lambda_bar - 0.20) + lambda_bar**2)
        sqrt_term = max(phi**2 - lambda_bar**2, 0.0)
        denom = phi + math.sqrt(sqrt_term)
        return min(1.0, 1.0 / denom) if denom > 0 else 0.0

    def phi_aux(lambda_bar: float, alpha: float) -> float:
        # EN 1993-1-1 §6.3.1.2 (auxiliary factor Φ)
        return 0.5 * (1.0 + alpha * (lambda_bar - 0.20) + lambda_bar**2)

    # Flexural buckling about y and z
    buck_results = []
    buck_map = {}  # for report/results mapping
    buck_map["alpha_y"] = alpha_y
    buck_map["alpha_z"] = alpha_z

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
        phi = 0.5 * (1.0 + alpha_use * (lambda_bar - 0.20) + lambda_bar**2)
        chi = chi_reduction(lambda_bar, alpha_use)
        Nb_Rd_N = chi * NRk_N / gamma_M1

        util = (abs(N_N) / Nb_Rd_N) if Nb_Rd_N > 0 else float("inf")
        status = "OK" if util <= 1.0 else "EXCEEDS"

        buck_results.append((axis_label, Ncr, lambda_bar, chi, Nb_Rd_N, status))
        buck_map[f"Ncr_{axis_label}"] = Ncr
        buck_map[f"lambda_{axis_label}"] = lambda_bar
        buck_map[f"phi_{axis_label}"] = phi
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
    # EN 1993-1-1 §6.3.1.4(1): relevant for OPEN sections; for closed (RHS/SHS/CHS) -> N/A (OK)
    family = str(use_props.get("family", "") or "").upper()
    is_closed_section = any(tag in family for tag in ("RHS", "SHS", "CHS"))
    
    if is_closed_section:
        rows.append({
            "Check": "Torsional / torsional-flexural buckling",
            "Utilization": f"{0.0:.3f}",
            "Status": "OK",
        })
    
        buck_map["i0_m"] = None
        buck_map["Ncr_T"] = None
        buck_map["lambda_T"] = None
        buck_map["phi_T"] = None
        buck_map["chi_T"] = None
        buck_map["Nb_Rd_T"] = None
        buck_map["util_T"] = 0.0
        buck_map["status_T"] = "OK"
    
    else:
        # ---- KEEP YOUR OPEN-SECTION CALCULATION EXACTLY AS-IS (ONLY INDENTED) ----
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
        alpha_T = alpha_z
        buck_map["alpha_T"] = alpha_T
    
        if i0_m > 0 and J_m4 > 0 and Iw_m6 > 0 and Leff_T > 0:
            Ncr_T = (1.0 / (i0_m**2)) * (G * J_m4 + (math.pi**2) * E * Iw_m6 / (Leff_T**2))
            lambda_T = math.sqrt(NRk_N / Ncr_T) if Ncr_T > 0 else float("inf")
            phi_T = phi_aux(lambda_T, alpha_T)
            chi_T = chi_reduction(lambda_T, alpha_T)
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
        buck_map["lambda_T"] = lambda_T if "lambda_T" in locals() else None
        buck_map["phi_T"] = phi_T if "phi_T" in locals() else None
        buck_map["chi_T"] = chi_T
        buck_map["Nb_Rd_T"] = Nb_Rd_T_N
        buck_map["util_T"] = util_T
        buck_map["status_T"] = status_T

    # (18) Lateral-torsional buckling (uniform moment, zg=0, k=kw=1; NCCI-style)
    # EN 1993-1-1 §6.3.2: LTB relevant for OPEN sections; for closed (RHS/SHS/CHS) -> N/A (OK)
    
    # ---- ALWAYS define locals first (prevents "not associated with a value") ----
    K_LT = float(inputs.get("K_LT", 1.0))
    Leff_LT = K_LT * L
    
    Mcr = None
    lambda_LT = None
    chi_LT = None
    Mb_Rd_Nm = None
    util_LT = None
    status_LT = "n/a"
    
    family = str(use_props.get("family", "") or "").upper()
    is_closed_section = any(tag in family for tag in ("RHS", "SHS", "CHS"))
    
    if is_closed_section:
        util_LT = 0.0
        status_LT = "OK"
    
        rows.append({
            "Check": "Lateral-torsional buckling",
            "Utilization": f"{util_LT:.3f}",
            "Status": status_LT,
        })
    
    else:
        # ---- KEEP YOUR OPEN-SECTION CALCULATION EXACTLY AS-IS (NO CHANGES INSIDE) ----
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
    
    # keep these EXACT keys as your original code expects
    buck_map["Leff_LT"] = Leff_LT
    buck_map["Mcr"] = Mcr
    buck_map["lambda_LT"] = lambda_LT
    buck_map["chi_LT"] = chi_LT
    buck_map["Mb_Rd"] = Mb_Rd_Nm
    buck_map["util_LT"] = util_LT
    buck_map["status_LT"] = status_LT
    buck_map["curve_LT"] = curve_LT
    buck_map["alpha_LT"] = alpha_LT

    # (19),(20) Buckling interaction for bending + axial compression — EN 1993-1-1 Annex B (Method 2)
    psi_y = 1.0
    psi_z = 1.0
    psi_LT = 1.0
    buck_map.update({"psi_y": psi_y, "psi_z": psi_z, "psi_LT": psi_LT})

    chi_y = buck_map.get("chi_y", 1.0) or 1.0
    chi_z = buck_map.get("chi_z", 1.0) or 1.0
    chiLT = chi_LT if chi_LT is not None else 1.0

    lam_y = buck_map.get("lambda_y", 0.0) or 0.0
    lam_z = buck_map.get("lambda_z", 0.0) or 0.0
    buck_map.update({"lam_y": lam_y, "lam_z": lam_z})

    Ny_denom_N = (chi_y * NRk_N / gamma_M1) if (chi_y and NRk_N > 0) else 0.0
    Nz_denom_N = (chi_z * NRk_N / gamma_M1) if (chi_z and NRk_N > 0) else 0.0
    My_denom_Nm = (chiLT * My_Rk_Nm / gamma_M1) if (chiLT and My_Rk_Nm > 0) else 0.0
    Mz_denom_Nm = (Mz_Rk_Nm / gamma_M1) if (Mz_Rk_Nm > 0) else 0.0

    Cmy_B = max(0.4, 0.60 + 0.40 * psi_y)
    Cmz_B = max(0.4, 0.60 + 0.40 * psi_z)
    CmLT_B = max(0.4, 0.60 + 0.40 * psi_LT)
    buck_map.update({"Cmy_B": Cmy_B, "Cmz_B": Cmz_B, "CmLT_B": CmLT_B})

    lam_y_use = min(float(lam_y or 0.0), 1.0)
    lam_z_use = min(float(lam_z or 0.0), 1.0)

    Ny_ratio = (abs(N_N) / Ny_denom_N) if Ny_denom_N > 0 else float("inf")
    Nz_ratio = (abs(N_N) / Nz_denom_N) if Nz_denom_N > 0 else float("inf")

    kyy_B = Cmy_B * (1.0 + (lam_y_use - 0.2) * Ny_ratio)
    kzz_B = Cmz_B * (1.0 + (2.0 * lam_z_use - 0.6) * Nz_ratio)
    kyz_B = 0.6 * kzz_B

    if float(lam_z or 0.0) >= 0.4:
        denom = max((CmLT_B - 0.25), 1e-9)
        kzy_B = 1.0 - 0.1 * lam_z_use / denom * Nz_ratio
    else:
        kzy_B = 1.0

    buck_map.update({
        "lam_y_use_B": lam_y_use,
        "lam_z_use_B": lam_z_use,
        "Ny_ratio_B": Ny_ratio,
        "Nz_ratio_B": Nz_ratio,
        "kyy_B": kyy_B,
        "kzz_B": kzz_B,
        "kyz_B": kyz_B,
        "kzy_B": kzy_B,
    })

    term_Ny = Ny_ratio
    term_My1 = (kyy_B * (abs(My_Ed_kNm) * 1e3) / My_denom_Nm) if My_denom_Nm > 0 else float("inf")
    term_Mz1 = (kyz_B * (abs(Mz_Ed_kNm) * 1e3) / Mz_denom_Nm) if Mz_denom_Nm > 0 else float("inf")
    util_19 = term_Ny + term_My1 + term_Mz1

    term_Nz = Nz_ratio
    term_My2 = (kzy_B * (abs(My_Ed_kNm) * 1e3) / My_denom_Nm) if My_denom_Nm > 0 else float("inf")
    term_Mz2 = (kzz_B * (abs(Mz_Ed_kNm) * 1e3) / Mz_denom_Nm) if Mz_denom_Nm > 0 else float("inf")
    util_20 = term_Nz + term_My2 + term_Mz2

    status_19 = "OK" if util_19 <= 1.0 else "EXCEEDS"
    status_20 = "OK" if util_20 <= 1.0 else "EXCEEDS"

    buck_map.update({
        "term_Ny_B": term_Ny, "term_My1_B": term_My1, "term_Mz1_B": term_Mz1,
        "term_Nz_B": term_Nz, "term_My2_B": term_My2, "term_Mz2_B": term_Mz2,
        "util_19_B": util_19, "util_20_B": util_20,
        "status_19_B": status_19, "status_20_B": status_20,
    })

    rows.append({
        "Check": "Buckling interaction y",
        "Applied": "Interaction",
        "Resistance": "≤ 1.0",
        "Utilization": f"{util_19:.3f}",
        "Status": status_19,
    })
    rows.append({
        "Check": "Buckling interaction z",
        "Applied": "Interaction",
        "Resistance": "≤ 1.0",
        "Utilization": f"{util_20:.3f}",
        "Status": status_20,
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

            val_delta = f"{w_max_mm:.3f}" if w_max_mm is not None else "n/a"
            val_L300  = f"{limit_L300:.3f}" if limit_L300 is not None else "n/a"
            val_L600  = f"{limit_L600:.3f}" if limit_L600 is not None else "n/a"
            val_L900  = f"{limit_L900:.3f}" if limit_L900 is not None else "n/a"

            k_delta = f"{key_prefix}delta_max_mm"
            k_L300  = f"{key_prefix}L300_mm"
            k_L600  = f"{key_prefix}L600_mm"
            k_L900  = f"{key_prefix}L900_mm"

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
        "Buckling interaction y",                                               # 19
        "Buckling interaction z",                                               # 20
    ]

    cs_util = ["" for _ in cs_checks]
    cs_status = ["" for _ in cs_checks]
    buck_util = ["" for _ in buck_checks]
    buck_status = ["" for _ in buck_checks]

    # -----------------------------
    # Helpers
    # -----------------------------
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

        # Robust fallbacks for checks 1–6
        if df_rows is not None:
            def _pick_first(predicate):
                for __i, __r in df_rows.iterrows():
                    __s = str(__r.get("Check","")).lower()
                    if predicate(__s):
                        return __r
                return None

            if idx_out == 0 and cs_util[0] == "" and cs_status[0] == "":
                r = _pick_first(lambda s: "tension" in s)
                if r is not None:
                    cs_util[0] = r.get("Utilization","")
                    cs_status[0] = r.get("Status","")

            if idx_out == 1 and cs_util[1] == "" and cs_status[1] == "":
                r = _pick_first(lambda s: ("compression" in s) and ("buckling" not in s))
                if r is not None:
                    cs_util[1] = r.get("Utilization","")
                    cs_status[1] = r.get("Status","")

            if idx_out == 2 and cs_util[2] == "" and cs_status[2] == "":
                r = _pick_first(lambda s: ("my" in s) and ("+" not in s) and ("vy" not in s) and ("vz" not in s))
                if r is not None:
                    cs_util[2] = r.get("Utilization","")
                    cs_status[2] = r.get("Status","")

            if idx_out == 3 and cs_util[3] == "" and cs_status[3] == "":
                r = _pick_first(lambda s: ("mz" in s) and ("+" not in s) and ("vy" not in s) and ("vz" not in s))
                if r is not None:
                    cs_util[3] = r.get("Utilization","")
                    cs_status[3] = r.get("Status","")

            if idx_out == 4 and cs_util[4] == "" and cs_status[4] == "":
                r = _pick_first(lambda s: ("vy" in s) and ("+" not in s))
                if r is not None:
                    cs_util[4] = r.get("Utilization","")
                    cs_status[4] = r.get("Status","")

            if idx_out == 5 and cs_util[5] == "" and cs_status[5] == "":
                r = _pick_first(lambda s: ("vz" in s) and ("+" not in s))
                if r is not None:
                    cs_util[5] = r.get("Utilization","")
                    cs_status[5] = r.get("Status","")

    # 1) N (tension)
    fill_cs_from_df(idx_out=0, must_contain=["Tension", "N"])

    # 2) N (compression)
    fill_cs_from_df(idx_out=1, must_contain=["Compression", "N"])

    # 3) My
    fill_cs_from_df(idx_out=2, must_contain=["My"], must_not_contain=["+", "Vy", "Vz"])

    # 4) Mz
    fill_cs_from_df(idx_out=3, must_contain=["Mz"], must_not_contain=["+", "Vy", "Vz"])

    # 5) Vy
    fill_cs_from_df(idx_out=4, must_contain=["Vy"], must_not_contain=["+"])

    # 6) Vz
    fill_cs_from_df(idx_out=5, must_contain=["Vz"], must_not_contain=["+"])

    # -------------------------------------------------
    # Fill combined checks (7–14) using cs_combo computed in compute_checks
    # -------------------------------------------------
    extras = st.session_state.get("extras") or {}
    cs_combo = extras.get("cs_combo") or {}

    util_My = cs_combo.get("util_My", None)
    util_Mz = cs_combo.get("util_Mz", None)

    # (7) My + Vy (EN 1993-1-1 §6.2.8) — if shear <= 0.5 Vpl,Rd => ignore shear effect
    cs_util[6] = _fmt_util(util_My)
    cs_status[6] = _ok(util_My)

    # (8) Mz + Vz (EN 1993-1-1 §6.2.8)
    cs_util[7] = _fmt_util(util_Mz)
    cs_status[7] = _ok(util_Mz)

    # (9)–(11) §6.2.9 (use u values computed in compute_checks; same as Report tab)
    cs_util[8] = _fmt_util(cs_combo.get("u_6209_9"))
    cs_status[8] = _ok(cs_combo.get("u_6209_9"))

    cs_util[9] = _fmt_util(cs_combo.get("u_6209_10"))
    cs_status[9] = _ok(cs_combo.get("u_6209_10"))

    cs_util[10] = _fmt_util(cs_combo.get("u_6209_11"))
    cs_status[10] = _ok(cs_combo.get("u_6209_11"))

    # (12)–(14) §6.2.10 (use u values computed in compute_checks; same as Report tab)
    cs_util[11] = _fmt_util(cs_combo.get("u_6210_12"))
    cs_status[11] = _ok(cs_combo.get("u_6210_12"))

    cs_util[12] = _fmt_util(cs_combo.get("u_6210_13"))
    cs_status[12] = _ok(cs_combo.get("u_6210_13"))

    cs_util[13] = _fmt_util(cs_combo.get("u_6210_14"))
    cs_status[13] = _ok(cs_combo.get("u_6210_14"))

    # -------------------------------------------------
    # Fill member stability checks (15–20) from buck_map
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

    # 19–20 (Annex B Method 2)
    buck_util[4] = _fmt_util(buck_map.get("util_19_B", None))
    buck_status[4] = buck_map.get("status_19_B", "") or ""
    buck_util[5] = _fmt_util(buck_map.get("util_20_B", None))
    buck_status[5] = buck_map.get("status_20_B", "") or ""

    # -------------------------------------------------
    # Table builder
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

        for i, name in enumerate(names):
            util = utils[i] if i < len(utils) else ""
            status = statuses[i] if i < len(statuses) else ""
            number = start_no + i

            status_upper = (status or "").strip().upper()
            if status_upper == "OK":
                bg = "#e6f7e6"
            elif status_upper == "EXCEEDS":
                bg = "#fde6e6"
            else:
                bg = "#ffffff" if (i % 2 == 0) else "#f9f9f9"

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

    if not show_footer:
        return

    st.markdown("---")
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
    if (not L_val or L_val <= 0.0) and ("L_mm" in input_vals):
        L_val = float(input_vals["L_mm"]) / 1000.0

    if (not L_val or L_val <= 0.0) and x is not None and len(x) > 1:
        L_val = float(x[-1] - x[0])

    # --------------------------------------
    # 4) Summary for deflection & forces
    # --------------------------------------
    summary = get_beam_summary_for_diagrams(x, V, M, delta, L_val)
    summary["bending_axis"] = bending_axis
    st.session_state["diag_summary"] = summary
    # ---- Fallback: if no deflection curve is returned, use case-specific Δmax (meters) ----
    if (summary.get("w_max_mm") is None) and isinstance(selected_case, dict):
        dfunc = selected_case.get("delta_max_func", None)
        if callable(dfunc):
            try:
                delta_max_m = dfunc(*args, E=E, I=I_m4)
            except TypeError:
                delta_max_m = dfunc(*args, E, I_m4)
    
            if delta_max_m is not None:
                summary["w_max_mm"] = float(delta_max_m) * 1000.0

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

def render_report_tab(key_prefix="rpt"):
    # ----------------------------------------------------
    # Print helpers (screen only)
    # ----------------------------------------------------
    EXPAND_ALL = False  # always keep sections collapsed by default (also for printing)
    st.markdown("<div class='no-print'>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div id="engi_report_root">', unsafe_allow_html=True)
    sr_display = st.session_state.get("sr_display")
    import math
    # Alias: section properties used throughout the report
    use_props = sr_display or {}
    inputs = st.session_state.get("inputs")
    df_rows = st.session_state.get("df_rows")
    overall_ok = st.session_state.get("overall_ok", True)
    governing = st.session_state.get("governing", (None, None))
    extras = st.session_state.get("extras") or {}
    meta = st.session_state.get("meta")
    material = st.session_state.get("material", st.session_state.get("mat_sel", "S355"))
    fy, fu = get_material_props(material)
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
    fy, fu = get_material_props(material)
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
    # 1. Project data (from Project tab)
    # ----------------------------------------------------
    report_h3("1. Project info")
    
    c1, c2, c3 = st.columns([1, 1, 1])
    
    with c1:
        st.text_input("Document title", value=str(doc_name), disabled=True, key=f"{key_prefix}_doc_title")
        st.text_input("Project name", value=str(project_name), disabled=True, key=f"{key_prefix}_project_name")
    
    with c2:
        st.text_input("Position / Location (Beam ID)", value=str(position), disabled=True, key=f"{key_prefix}_position")
        st.text_input("Requested by", value=str(requested_by), disabled=True, key=f"{key_prefix}_requested_by")
    
    with c3:
        st.text_input("Revision", value=str(revision), disabled=True, key=f"{key_prefix}_revision")
        st.text_input("Date", value=str(run_date), disabled=True, key=f"{key_prefix}_date")
    
    # Notes: keep editable in Report tab (optional but usually useful)
    if f"{key_prefix}_rpt_notes" not in st.session_state:
        st.session_state[f"{key_prefix}_rpt_notes"] = str(notes)
    
    st.text_area("Notes / comments", value=str(notes), disabled=True, height=120, key=f"{key_prefix}_notes")
    
    st.markdown("---")

    # ----------------------------------------------------
    # 2. Material design values (EN 1993-1-1)
    # ----------------------------------------------------
    report_h3("2. Material design values (EN 1993-1-1)")
    
    # basic material values
    eps = (235.0 / fy) ** 0.5 if fy and fy > 0 else None
    E = 210000.0  # MPa
    nu = 0.30
    G = E / (2.0 * (1.0 + nu))  # shear modulus [MPa]
    fy, fu = get_material_props(material)
    
    # ---- Line 1 (PRINTS) ----
    c1, c2, c3 = st.columns(3)
    with c1:
        st.text_input(
            "Steel grade",
            value=str(material),
            disabled=True,
            key=f"{key_prefix}_mat_grade",
        )
    with c2:
        st.text_input(
            "γ_M0",
            value=f"{gamma_M0:.2f}",
            disabled=True,
            key=f"{key_prefix}_mat_gM0",
        )
    with c3:
        st.text_input(
            "γ_M1",
            value=f"{gamma_M1:.2f}",
            disabled=True,
            key=f"{key_prefix}_mat_gM1",
        )
    
    # ---- Lines 2–3 (SCREEN ONLY) ----
    st.markdown("<div class='no-print'>", unsafe_allow_html=True)
    
    r1, r2, r3 = st.columns(3)
    with r1:
        st.text_input(
            "Yield strength f_y [MPa]",
            value=f"{fy:.1f}",
            disabled=True,
            key=f"{key_prefix}_mat_fy",
        )
    with r2:
        st.text_input(
            "Ultimate strength f_u [MPa]",
            value=f"{fu:.1f}",
            disabled=True,
            key=f"{key_prefix}_mat_fu",
        )
        
    with r3:
        st.text_input(
            "Elastic modulus E [MPa]",
            value=f"{E:.0f}",
            disabled=True,
            key=f"{key_prefix}_mat_E",
        )
    
    r4, r5, r6 = st.columns(3)
    with r4:
        st.text_input(
            "Shear modulus G [MPa]",
            value=f"{G:.0f}",
            disabled=True,
            key=f"{key_prefix}_mat_G",
        )
    with r5:
        st.text_input(
            "ε = √(235 / f_y)",
            value=f"{eps:.3f}" if eps is not None else "n/a",
            disabled=True,
            key=f"{key_prefix}_mat_eps",
        )
    with r6:
        st.text_input(
            "Poisson’s ratio ν",
            value=f"{nu:.2f}",
            disabled=True,
            key=f"{key_prefix}_mat_nu",
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
        st.text_input("Member type", value=member_type, disabled=True, key=f"{key_prefix}_mem_type")
        st.number_input("Span length L [m]", value=float(L), disabled=True, key=f"{key_prefix}_L")
    with mc2:
        st.text_input("Support conditions", value=support_txt, disabled=True, key=f"{key_prefix}_support")
        st.number_input("Effective length L_y [m]", value=float(Leff_y), disabled=True, key=f"{key_prefix}_Leff_y")
    with mc3:
        st.number_input("Effective length L_z [m]", value=float(Leff_z), disabled=True, key=f"{key_prefix}_Leff_z")
        st.number_input("LT buckling length L_LT [m]", value=float(Leff_LT), disabled=True, key=f"{key_prefix}_Leff_LT")

    # 3.2 Section data
    report_h4("3.2 Section data")

    cs1, cs2 = st.columns(2)
    
    with cs1:
        # --- one-line section identity (same "box row" vibe as the summary below) ---
        r1, r2, r3 = st.columns(3)
    
        with r1:
            st.text_input("Section family", value=fam, disabled=True, key=f"{key_prefix}_cs_family")
        with r2:
            st.text_input("Section size", value=name, disabled=True, key=f"{key_prefix}_cs_name")
        with r3:
            # Total mass = L [m] * unit mass [kg/m]
            try:
                unit_mass = float(sr_display.get("m_kg_per_m", 0.0) or 0.0)
            except Exception:
                unit_mass = 0.0
    
            try:
                L_m = float(L)
            except Exception:
                L_m = 0.0
    
            total_mass = unit_mass * L_m
            st.number_input(
                "Total mass [kg]",
                value=float(total_mass),
                disabled=True,
                key=f"{key_prefix}_total_mass",
            )
    
        # Selected section summary — already nice (6 boxes per 2 rows)
        render_section_summary_like_props(material, sr_display, key_prefix=f"{key_prefix}_sum")

    with cs2:
        img_path = get_section_image(fam)
    
        if img_path and Path(img_path).exists():
            # push it a bit down + center horizontally in the right column
            st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    
            _l, _m, _r = st.columns([0.35, 1.3, 0.35])  # slightly less huge than 0.2/1.6/0.2
            with _m:
                st.image(img_path, use_container_width=True)
        else:
            st.info("No image available for this family.")


    # full DB section properties
    with st.expander("3.3 Section properties from DB", expanded=EXPAND_ALL):
        render_section_properties_readonly(sr_display, key_prefix=f"{key_prefix}_props")

    st.markdown("---")

    # ----------------------------------------------------
    # 4. Section classification (EN 1993-1-1 §5.5)
    # (screen only – hidden in print)
    # ----------------------------------------------------
    st.markdown("<div class='no-print'>", unsafe_allow_html=True)
    
    report_h3("4. Section classification (EN 1993-1-1 §5.5)")
    
    material = st.session_state.get("mat_sel", "S355")
    fy, fu = get_material_props(material)

    cls = calc_section_classes_ec3(
        sr_display, fy,
        NEd_kN=inputs.get("N_kN", 0.0),
        My_Ed_kNm=inputs.get("My_kNm", 0.0),   # you said My controls the web case
    )
    
    flange_class = cls.get("flange_comp", "n/a")     # NEW key
    web_class    = cls.get("web_bend_comp", "n/a")   # NEW key
    gov_class    = cls.get("governing", "n/a")

    # you should already have these 3 from cls:
    # flange_class, web_class, gov_class
    
    st.session_state["rpt_flange_class_comp"] = str(flange_class)
    st.session_state["rpt_web_class_bc"]      = str(web_class)
    st.session_state["rpt_cs_class"]          = str(gov_class)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.text_input("Flange – pure compression", disabled=True, key=f"{key_prefix}_flange_class_comp")
    with c2:
        st.text_input("Web – bending + compression", disabled=True, key=f"{key_prefix}_web_class_bc")
    with c3:
        st.text_input("Governing cross-section class", disabled=True, key=f"{key_prefix}_cs_class")
    
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
        st.text_input("N_Ed [kN]", f"{inputs.get('N_kN',0.0):.3f}", disabled=True, key=f"{key_prefix}_N_Ed")
    with r1c2:
        st.text_input("Vy_Ed [kN]", f"{inputs.get('Vy_kN',0.0):.3f}", disabled=True, key=f"{key_prefix}_Vy_Ed")
    with r1c3:
        st.text_input("Vz_Ed [kN]", f"{inputs.get('Vz_kN',0.0):.3f}", disabled=True, key=f"{key_prefix}_Vz_Ed")

    # --- Row 2: My, Mz ---
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        st.text_input("My_Ed [kNm]", f"{inputs.get('My_kNm',0.0):.3f}", disabled=True, key=f"{key_prefix}_My_Ed")
    with r2c2:
        st.text_input("Mz_Ed [kNm]", f"{inputs.get('Mz_kNm',0.0):.3f}", disabled=True, key=f"{key_prefix}_Mz_Ed")
    with r2c3:
        st.empty()   # clean layout (no empty grey box)
    st.markdown("---")

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
        key_prefix=f"{key_prefix}_rep_",      # different keys than Results tab
    )
    # Status from the checks table (row 'Tension (N≥0)')
    status_ten = "n/a"
    
    # ----------------------------------------------------
    # 6. Detailed calculations
    # ----------------------------------------------------
    with st.expander("Detailed calculations", expanded=EXPAND_ALL):
        report_h3("6. Detailed calculations")
        report_h4("6.1 Verification of cross-section strength (ULS, checks 1–14)")
    
        # 6.1.a Detailed explanation for check (1) Tension
        report_h4("(1) Tension – EN 1993-1-1 §6.2.3")
    
        # Get section area and material
        A_mm2 = float(sr_display.get("A_mm2", 0.0))  # from DB
        fy, fu = get_material_props(material)
    
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
            )
            report_status_badge(u_ten)
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
                )
                report_status_badge(u_comp)
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
            rf" = {util_My:.3f} \le 1.0"
        )
        report_status_badge(util_My)
    
        st.latex(
            rf"u_z = \frac{{M_{{z,Ed}}}}{{M_{{c,z,Rd}}}}"
            rf" = \frac{{{Mz_Ed_kNm:.2f}}}{{{Mc_z_Rd_kNm:.2f}}}"
            rf" = {util_Mz:.3f} \le 1.0"
        )
        report_status_badge(util_Mz)
    
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
            Vc_z_Rd_kN = (Av_z_mm2 * (fy / (3 ** 0.5)) / gamma_M0) / 1000.0
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
            rf" = {util_Vz:.3f} \le 1.0"
        )
    
        st.latex(
            rf"u_y = \frac{{V_{{y,Ed}}}}{{V_{{c,y,Rd}}}}"
            rf" = \frac{{{Vy_Ed_kN:.2f}}}{{{Vc_y_Rd_kN:.2f}}}"
            rf" = {util_Vy:.3f} \le 1.0"
        )
    
        # governing shear utilization
        report_status_badge(max(util_Vz, util_Vy))
    
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
    1. rotation compatibility due to bending of other transversely connected members,  
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
        
        def _eq_line(label_html: str, latex_expr: str):
            cL, cM, cR = st.columns([3, 4, 3])
            with cL:
                st.markdown(label_html, unsafe_allow_html=True)
            with cM:
                st.latex(latex_expr)
        
        st.markdown(
            "This section evaluates the influence of axial force on the bending resistance for Class 1–2 cross-sections "
            "in accordance with **EN 1993-1-1 §6.2.9**."
        )
        
        # --- Inputs from DB / session ---
        family = (sr_display.get("family") or "").upper()
        
        A_mm2  = float(sr_display.get("A_mm2") or 0.0)
        h_mm   = float(sr_display.get("h_mm") or 0.0)
        b_mm   = float(sr_display.get("b_mm") or 0.0)
        tw_mm  = float(sr_display.get("tw_mm") or 0.0)
        tf_mm  = float(sr_display.get("tf_mm") or 0.0)
        
        # Resistances expected from datasheet/DB
        Npl_Rd_kN     = float(sr_display.get("Npl_Rd_kN") or 0.0)
        Mpl_y_Rd_kNm  = float(sr_display.get("Mpl_Rd_y_kNm") or 0.0)
        Mpl_z_Rd_kNm  = float(sr_display.get("Mpl_Rd_z_kNm") or 0.0)
        
        # Design effects
        NEd_kN    = float(inputs.get("N_kN") or 0.0)
        My_Ed_kNm = float(inputs.get("My_kNm") or 0.0)
        Mz_Ed_kNm = float(inputs.get("Mz_kNm") or 0.0)
        
        _eq_line("Design axial force:", rf"N_{{Ed}} = {NEd_kN:.2f}\,\mathrm{{kN}}")
        
        # Normalized axial force
        n = (abs(NEd_kN) / Npl_Rd_kN) if (Npl_Rd_kN > 0.0) else None
        if n is not None:
            _eq_line(
                "Normalized axial force:",
                rf"n=\frac{{|N_{{Ed}}|}}{{N_{{pl,Rd}}}}=\frac{{{abs(NEd_kN):.2f}}}{{{Npl_Rd_kN:.2f}}}={n:.3f}"
            )
        else:
            st.warning("Cannot evaluate interaction because Npl,Rd is not available in the datasheet/DB.")
        
        # Practical family detection
        is_rhs = any(k in family for k in ["RHS", "SHS", "HSS", "BOX"])
        is_ih  = any(k in family for k in ["IPE", "HE", "HEA", "HEB", "HEM", "IPN", "UB", "UC", "W", "I", "H"])
        
        # Outputs
        MN_y_Rd = None
        MN_z_Rd = None
        alpha_y = None
        alpha_z = None
        
        st.markdown("For the examined case:")
        
        # --------------------------
        # A) Doubly-symmetric I / H sections
        # Eq. (8.45–8.47) shown for 'may ignore' check (narrative only),
        # but general reduction Eq. (8.48–8.50) is applied regardless (uniform format).
        # --------------------------
        if is_ih and (n is not None) and (A_mm2 > 0.0) and (Mpl_y_Rd_kNm > 0.0) and (Mpl_z_Rd_kNm > 0.0):
        
            # --- "May ignore effect" criteria (8.45–8.47) ---
            hw_mm = max(h_mm - 2.0 * tf_mm, 0.0)
        
            crit_y_25  = 0.25 * Npl_Rd_kN if Npl_Rd_kN > 0 else None
            crit_y_web = (0.50 * hw_mm * tw_mm * fy / gamma_M0) / 1000.0 if (hw_mm > 0 and tw_mm > 0 and fy > 0) else None
            crit_z_web = (hw_mm * tw_mm * fy / gamma_M0) / 1000.0 if (hw_mm > 0 and tw_mm > 0 and fy > 0) else None
        
            if crit_y_25 is not None:
                _eq_line("Major axis criterion 1:", rf"N_{{Ed}} \le 0.25\,N_{{pl,Rd}} = {crit_y_25:.1f}\,\mathrm{{kN}}")
            if crit_y_web is not None:
                _eq_line("Major axis criterion 2:", rf"N_{{Ed}} \le 0.50\,h_w t_w f_y/\gamma_{{M0}} = {crit_y_web:.1f}\,\mathrm{{kN}}")
            if crit_z_web is not None:
                _eq_line("Minor axis criterion:", rf"N_{{Ed}} \le h_w t_w f_y/\gamma_{{M0}} = {crit_z_web:.1f}\,\mathrm{{kN}}")
        
            axial_ok_y = (
                (crit_y_25 is not None and abs(NEd_kN) <= crit_y_25) and
                (crit_y_web is not None and abs(NEd_kN) <= crit_y_web)
            )
            axial_ok_z = (crit_z_web is not None and abs(NEd_kN) <= crit_z_web)
        
            if axial_ok_y and axial_ok_z:
                st.markdown(
                    "The axial force is sufficiently small to neglect its influence on the plastic bending resistance "
                    "for the current I/H section. Therefore the bending utilizations remain as in **(3)** and **(4)**."
                )
                st.caption("Note: in this report the general reduction method is still applied to keep a uniform calculation format.")
            else:
                st.markdown(
                    "The axial force is not sufficiently small to neglect its influence on the plastic bending resistance. "
                    "The general reduction method is applied below."
                )
        
            # --- General reduction Eq. (8.48–8.50) ---
            a = (A_mm2 - 2.0 * b_mm * tf_mm) / A_mm2
            a = max(0.0, min(0.5, a))
            _eq_line("Section parameter:", rf"a=\min\left(0.5,\frac{{A-2b_f t_f}}{{A}}\right)={a:.3f}")
        
            # Eq. (8.48)
            denom_y = (1.0 - 0.5 * a)
            MN_y_Rd = Mpl_y_Rd_kNm * (1.0 - n) / denom_y if denom_y > 0 else 0.0
            MN_y_Rd = min(Mpl_y_Rd_kNm, max(0.0, MN_y_Rd))
        
            # Eq. (8.49)-(8.50)
            if n <= a:
                MN_z_Rd = Mpl_z_Rd_kNm
            else:
                ratio = (n - a) / (1.0 - a) if (1.0 - a) > 0 else 1.0
                MN_z_Rd = Mpl_z_Rd_kNm * (1.0 - ratio**2)
                MN_z_Rd = max(0.0, MN_z_Rd)
        
            # Exponents for Eq. (8.56)
            alpha_y = 2.0
            alpha_z = max(1.0, 5.0 * n)
        
            st.latex(r"\frac{M_{Ed}}{M_{N,Rd}} \le 1.0")

            # -------------------------------------------------
            # Major axis y–y (EN 1993-1-1 Eq. 8.48)
            # -------------------------------------------------
            _eq_line(
                "Major axis reduction (Eq. 8.48):",
                r"M_{N,y,Rd}=\min\!\left(M_{pl,y,Rd},\;M_{pl,y,Rd}\frac{1-n}{1-0.5a}\right)"
            )
            
            _eq_line(
                "Final reduced resistance:",
                rf"M_{{N,y,Rd}} = {MN_y_Rd:.3f}\,\mathrm{{kNm}}"
            )
            
            # -------------------------------------------------
            # Minor axis z–z (EN 1993-1-1 Eq. 8.49–8.50)
            # -------------------------------------------------
            _eq_line(
                "Minor axis reduction:",
                r"""
                \begin{cases}
                M_{N,z,Rd}=M_{pl,z,Rd} & \text{for } n\le a \\[6pt]
                M_{N,z,Rd}=M_{pl,z,Rd}\!\left[1-\left(\frac{n-a}{1-a}\right)^2\right]
                & \text{for } n>a
                \end{cases}
                """
            )
            
            _eq_line(
                "Final reduced resistance:",
                rf"M_{{N,z,Rd}} = {MN_z_Rd:.3f}\,\mathrm{{kNm}}"
            )
            
            # -------------------------------------------------
            # Biaxial bending interaction (EN 1993-1-1 Eq. 8.56)
            # -------------------------------------------------
            _eq_line(
                "Interaction criterion (Eq. 8.56):",
                r"""
                \left(\frac{M_{y,Ed}}{M_{N,y,Rd}}\right)^{\alpha_y}
                +
                \left(\frac{M_{z,Ed}}{M_{N,z,Rd}}\right)^{\alpha_z}
                \le 1.0
                """
            )
            
            _eq_line(
                "Interaction exponents:",
                rf"\alpha_y = 2.0,\qquad \alpha_z = \max(1,5n) = {alpha_z:.2f}"
            )

        # --------------------------
        # B) Rectangular hollow sections (RHS/SHS/HSS/BOX)
        # Eq. (8.51–8.52), exponents per 8.56 (RHS rules)
        # --------------------------
        elif is_rhs and (n is not None) and (A_mm2 > 0.0) and (Mpl_y_Rd_kNm > 0.0) and (Mpl_z_Rd_kNm > 0.0):
        
            aw = (A_mm2 - 2.0 * b_mm * tf_mm) / A_mm2 if A_mm2 > 0 else 0.0
            af = (A_mm2 - 2.0 * h_mm * tw_mm) / A_mm2 if A_mm2 > 0 else 0.0
            aw = max(0.0, min(0.5, aw))
            af = max(0.0, min(0.5, af))
        
            _eq_line("Section parameter:", rf"a_w=\min\left(0.5,\frac{{A-2bt_f}}{{A}}\right)={aw:.3f}")
            _eq_line("Section parameter:", rf"a_f=\min\left(0.5,\frac{{A-2ht_w}}{{A}}\right)={af:.3f}")
        
            # Eq. (8.51)-(8.52)
            denom_y = (1.0 - 0.5 * aw)
            denom_z = (1.0 - 0.5 * af)
        
            MN_y_Rd = Mpl_y_Rd_kNm * (1.0 - n) / denom_y if denom_y > 0 else 0.0
            MN_z_Rd = Mpl_z_Rd_kNm * (1.0 - n) / denom_z if denom_z > 0 else 0.0
        
            MN_y_Rd = min(Mpl_y_Rd_kNm, max(0.0, MN_y_Rd))
            MN_z_Rd = min(Mpl_z_Rd_kNm, max(0.0, MN_z_Rd))
        
            # Exponents for RHS (8.56 excerpt)
            if n <= 0.8:
                denom = (1.0 - 1.13 * (n**2))
                alpha = min(6.0, 1.66 / denom) if denom > 0 else 6.0
            else:
                alpha = 6.0
            alpha_y = alpha
            alpha_z = alpha
        
            st.latex(r"\frac{M_{Ed}}{M_{N,Rd}} \le 1.0")
            _eq_line("Reduced resistance (y):", rf"M_{{N,y,Rd}}={MN_y_Rd:.3f}\,\mathrm{{kNm}}")
            _eq_line("Reduced resistance (z):", rf"M_{{N,z,Rd}}={MN_z_Rd:.3f}\,\mathrm{{kNm}}")
            _eq_line("Exponents (8.56):", rf"\alpha_y=\alpha_z={alpha:.3f}")
        
        # --------------------------
        # C) Fallback (unknown family)
        # --------------------------
        else:
            st.info(
                "Section family is not recognized as I/H or rectangular hollow. "
                "If you add a 'section_type' field in the datasheet (IH/RHS/CHS/EHS/etc.), "
                "this section can be extended with the corresponding Eurocode formula."
            )
        
        # --- Utilizations (always from reduced resistances) ---
        if (MN_y_Rd is not None) and (MN_y_Rd > 0.0):
            uy = abs(My_Ed_kNm) / MN_y_Rd
            _eq_line("Major-axis utilization:", rf"u_y=\frac{{|M_{{y,Ed}}|}}{{M_{{N,y,Rd}}}}=\frac{{{abs(My_Ed_kNm):.3f}}}{{{MN_y_Rd:.3f}}}={uy:.3f}")
            report_status_badge(uy)
        else:
            uy = None
            st.warning("Cannot compute u_y because M_N,y,Rd is not available.")
        
        if (MN_z_Rd is not None) and (MN_z_Rd > 0.0):
            uz = abs(Mz_Ed_kNm) / MN_z_Rd
            _eq_line("Minor-axis utilization:", rf"u_z=\frac{{|M_{{z,Ed}}|}}{{M_{{N,z,Rd}}}}=\frac{{{abs(Mz_Ed_kNm):.3f}}}{{{MN_z_Rd:.3f}}}={uz:.3f}")
            report_status_badge(uz)
        else:
            uz = None
            st.warning("Cannot compute u_z because M_N,z,Rd is not available.")
        
        # --- Biaxial interaction (Eq. 8.56) ---
        if (uy is not None) and (uz is not None) and (alpha_y is not None) and (alpha_z is not None):
            u_yz = (uy ** alpha_y) + (uz ** alpha_z)
        
            # Main equation (symbolic)
            st.latex(
                rf"u_{{y+z}}="
                rf"\left(\frac{{M_{{y,Ed}}}}{{M_{{N,y,Rd}}}}\right)^{{{alpha_y:.2f}}}"
                rf"+\left(\frac{{M_{{z,Ed}}}}{{M_{{N,z,Rd}}}}\right)^{{{alpha_z:.2f}}}"
            )
        
            # Result line
            st.latex(
                rf"u_{{y+z}}={u_yz:.3f}\le 1.0"
            )
        
            report_status_badge(u_yz)

        else:
            st.info("Biaxial interaction (8.56) not evaluated because required inputs are missing.")
            
        # ----------------------------------------------------
        # (6.2.10) Bending, shear and axial force
        # ----------------------------------------------------
        report_h4("(12), (13), (14) Bending, shear and axial force (EN 1993-1-1 §6.2.10)")
        
        # Helpers: centered equation column (same as your §6.2.9 layout)
        def _eq_line(label_html: str, latex_expr: str):
            cL, cM, cR = st.columns([3, 4, 3])
            with cL:
                st.markdown(label_html, unsafe_allow_html=True)
            with cM:
                st.latex(latex_expr)
        
        # ---- Inputs / section resistances (from DB) ----
        family = (sr_display.get("family") or "").upper()
        
        A_mm2 = float(sr_display.get("A_mm2") or 0.0)
        h_mm  = float(sr_display.get("h_mm") or 0.0)
        b_mm  = float(sr_display.get("b_mm") or 0.0)
        tw_mm = float(sr_display.get("tw_mm") or 0.0)
        tf_mm = float(sr_display.get("tf_mm") or 0.0)
        
        # Shear areas from DB (support both key spellings)
        Av_z_mm2 = float(sr_display.get("Av_z_mm2", sr_display.get("Avz_mm2", 0.0)) or 0.0)
        Av_y_mm2 = float(sr_display.get("Av_y_mm2", sr_display.get("Avy_mm2", 0.0)) or 0.0)
        
        # Plastic axial & plastic bending resistances from DB (already used in §6.2.9 block)
        Npl_Rd_kN    = float(sr_display.get("Npl_Rd_kN") or 0.0)
        Mpl_y_Rd_kNm = float(sr_display.get("Mpl_Rd_y_kNm") or 0.0)
        Mpl_z_Rd_kNm = float(sr_display.get("Mpl_Rd_z_kNm") or 0.0)
        
        # Design actions
        NEd_kN    = float(cs_combo.get("NEd_kN") or abs(inputs.get("N_kN") or 0.0))
        My_Ed_kNm = float(inputs.get("My_kNm") or 0.0)
        Mz_Ed_kNm = float(inputs.get("Mz_kNm") or 0.0)
        Vy_Ed_kN  = float(inputs.get("Vy_kN") or 0.0)
        Vz_Ed_kN  = float(inputs.get("Vz_kN") or 0.0)
        
        st.markdown(
            "This section considers the influence of **shear force and axial force** on the **bending resistance** "
            "in accordance with **EN 1993-1-1 §6.2.10**."
        )
        
        # ---- Shear resistances (Vc,Rd) for each direction ----
        Vc_z_Rd_kN = (Av_z_mm2 * (fy / math.sqrt(3.0)) / gamma_M0) / 1e3 if (Av_z_mm2 > 0 and fy > 0 and gamma_M0 > 0) else 0.0
        Vc_y_Rd_kN = (Av_y_mm2 * (fy / math.sqrt(3.0)) / gamma_M0) / 1e3 if (Av_y_mm2 > 0 and fy > 0 and gamma_M0 > 0) else 0.0
        
        # Ratios
        eta_vz = (abs(Vz_Ed_kN) / Vc_z_Rd_kN) if Vc_z_Rd_kN > 0 else None
        eta_vy = (abs(Vy_Ed_kN) / Vc_y_Rd_kN) if Vc_y_Rd_kN > 0 else None
        
        # Case split per EN 1993-1-1 §6.2.10(2): ignore shear effect if VEd <= 0.5 Vpl,Rd (use Vc,Rd as Vpl,Rd here)
        shear_small = True
        if (eta_vz is not None) and (eta_vz > 0.50):
            shear_small = False
        if (eta_vy is not None) and (eta_vy > 0.50):
            shear_small = False
        
        # ---- Section family detection (same idea as your §6.2.9 block) ----
        is_rhs = any(k in family for k in ["RHS", "SHS", "HSS", "BOX"])
        is_ih  = any(k in family for k in ["IPE", "HE", "HEA", "HEB", "HEM", "IPN", "UB", "UC", "W", "I", "H"])
        
        # ---- Normalized axial force n ----
        n = (NEd_kN / Npl_Rd_kN) if (Npl_Rd_kN > 0) else None
        if n is not None:
            _eq_line("Design axial force:", rf"N_{{Ed}} = {NEd_kN:.2f}\,\mathrm{{kN}}")
            _eq_line("Normalized axial force:", rf"n=\frac{{N_{{Ed}}}}{{N_{{pl,Rd}}}}=\frac{{{NEd_kN:.2f}}}{{{Npl_Rd_kN:.2f}}}={n:.3f}")
        
        # ---- Show shear check for the §6.2.10 split ----
        if eta_vz is not None:
            _eq_line("Shear ratio (z):", rf"\eta_{{v,z}}=\frac{{V_{{z,Ed}}}}{{V_{{c,z,Rd}}}}=\frac{{{abs(Vz_Ed_kN):.2f}}}{{{Vc_z_Rd_kN:.2f}}}={eta_vz:.3f}")
        if eta_vy is not None:
            _eq_line("Shear ratio (y):", rf"\eta_{{v,y}}=\frac{{V_{{y,Ed}}}}{{V_{{c,y,Rd}}}}=\frac{{{abs(Vy_Ed_kN):.2f}}}{{{Vc_y_Rd_kN:.2f}}}={eta_vy:.3f}")
        
        if shear_small:
            st.markdown(
                "Since **all applied shear components** satisfy "
                r"$V_{Ed}\le 0.5\,V_{pl,Rd}$, the shear influence may be **neglected** "
                "per **EN 1993-1-1 §6.2.10(2)**. "
                "Therefore, this verification reduces to the **same bending–axial formulation** used in **§6.2.9**."
            )
        else:
            st.markdown(
                "Since **at least one shear component** exceeds "
                r"$0.5\,V_{pl,Rd}$, a reduction is applied in accordance with **EN 1993-1-1 §6.2.10(3)** "
                "using a reduced yield strength."
            )
        
        # =========================================================
        # Compute reduced yield strength fy,red only if needed
        # =========================================================
        fy_red = fy
        rho = 0.0
        if (not shear_small) and (fy > 0) and (gamma_M0 > 0):
            # Use governing shear ratio for the reduction parameter ρ (conservative, simple)
            eta_gov = 0.0
            if eta_vz is not None:
                eta_gov = max(eta_gov, float(eta_vz))
            if eta_vy is not None:
                eta_gov = max(eta_gov, float(eta_vy))
        
            # EN 1993-1-1:2022 Eq. (8.61):  ρ = (2*VEd/Vc,Rd - 1)^2
            rho = max(0.0, (2.0 * eta_gov - 1.0) ** 2)
            rho = min(rho, 1.0)  # keep physical bounds
            fy_red = max(0.0, (1.0 - rho) * fy)
        
            _eq_line("Reduction parameter:", rf"\rho=\left(2\frac{{V_{{Ed}}}}{{V_{{c,Rd}}}}-1\right)^2={rho:.3f}")
            _eq_line("Reduced yield strength:", rf"f_{{y,red}}=(1-\rho)f_y={(fy_red):.1f}\,\mathrm{{MPa}}")
        
        # =========================================================
        # Compute reduced bending resistances with axial force:
        #   Use same §6.2.9 formulas but with fy or fy_red depending on case
        #   (conservative + consistent with your report style)
        # =========================================================
        MN_y_Rd = None
        MN_z_Rd = None
        alpha_y = None
        alpha_z = None
        
        # If we cannot evaluate, stop gracefully
        if (n is None) or (A_mm2 <= 0) or (gamma_M0 <= 0) or (fy <= 0):
            st.warning("Cannot evaluate §6.2.10 because required section resistances (Npl,Rd / A / fy) are missing.")
        else:
            # Use the yield strength that applies in this §6.2.10 case
            fy_use = fy if shear_small else fy_red
        
            # A) I/H sections — use §6.2.9.1(5) formulas (8.48–8.50)
            if is_ih and (Mpl_y_Rd_kNm > 0 or Mpl_z_Rd_kNm > 0):
                # a = (A - 2 b_f t_f) / A, limited to 0.5  (as in your §6.2.9 block)
                a = min(0.5, (A_mm2 - 2.0 * b_mm * tf_mm) / A_mm2) if A_mm2 > 0 else 0.5
        
                # Scale plastic moments to fy_use (because DB moments were based on fy)
                scale = (fy_use / fy) if fy > 0 else 1.0
                Mpl_y_use = Mpl_y_Rd_kNm * scale
                Mpl_z_use = Mpl_z_Rd_kNm * scale
        
                MN_y_Rd = min(Mpl_y_use, Mpl_y_use * (1.0 - n) / (1.0 - 0.5 * a)) if (Mpl_y_use > 0) else 0.0
                if n <= a:
                    MN_z_Rd = Mpl_z_use
                else:
                    MN_z_Rd = Mpl_z_use * (1.0 - ((n - a) / (1.0 - a)) ** 2) if (Mpl_z_use > 0) else 0.0
        
                # Exponents from EN 1993-1-1 §6.2.9(9) for I/H (as you already used)
                alpha_y = 2.0
                alpha_z = max(1.0, 5.0 * n)
        
            # B) RHS/SHS — use §6.2.9.1(6) formulas (8.51–8.52)
            elif is_rhs and (Mpl_y_Rd_kNm > 0 or Mpl_z_Rd_kNm > 0):
                aw = min(0.5, (A_mm2 - 2.0 * b_mm * tf_mm) / A_mm2) if (A_mm2 > 0) else 0.5
                af = min(0.5, (A_mm2 - 2.0 * h_mm * tw_mm) / A_mm2) if (A_mm2 > 0) else 0.5
        
                scale = (fy_use / fy) if fy > 0 else 1.0
                Mpl_y_use = Mpl_y_Rd_kNm * scale
                Mpl_z_use = Mpl_z_Rd_kNm * scale
        
                MN_y_Rd = min(Mpl_y_use, Mpl_y_use * (1.0 - n) / (1.0 - 0.5 * aw)) if (Mpl_y_use > 0) else 0.0
                MN_z_Rd = min(Mpl_z_use, Mpl_z_use * (1.0 - n) / (1.0 - 0.5 * af)) if (Mpl_z_use > 0) else 0.0
        
                # Exponents per EN 1993-1-1 (RHS guidance you already implemented)
                if n <= 0.8:
                    alpha_y = min(6.0, 1.66 / (1.0 - 1.13 * (n ** 2)))
                else:
                    alpha_y = 6.0
                alpha_z = alpha_y
        
            else:
                st.info("Section family not recognized for §6.2.10 (I/H or RHS). No interaction model applied here.")
                MN_y_Rd, MN_z_Rd, alpha_y, alpha_z = None, None, None, None
        
        # =========================================================
        # Utilization factors (12), (13), (14) — centered + badges
        # Requirement from you:
        #   - If shear_small: case (14) should be EXACTLY the same as case (12)
        # =========================================================
        if (MN_y_Rd is not None) and (MN_z_Rd is not None) and (alpha_y is not None) and (alpha_z is not None) and (MN_y_Rd > 0) and (MN_z_Rd > 0):
            # Moment-only terms
            uMy = (abs(My_Ed_kNm) / MN_y_Rd) if MN_y_Rd > 0 else None
            uMz = (abs(Mz_Ed_kNm) / MN_z_Rd) if MN_z_Rd > 0 else None
        
            # Shear term (only active when reduction case is active)
            uV = 0.0
            if not shear_small:
                uV_y = (abs(Vy_Ed_kN) / Vc_y_Rd_kN) if Vc_y_Rd_kN > 0 else 0.0
                uV_z = (abs(Vz_Ed_kN) / Vc_z_Rd_kN) if Vc_z_Rd_kN > 0 else 0.0
                uV = (uV_y ** 2) + (uV_z ** 2)
        
            # -------------------------------------------------
            # Utilizations (12), (13), (14)
            # IMPORTANT: if shear is neglected => identical to §6.2.9 (9),(10),(11)
            # -------------------------------------------------
        
            # Shear term (only active when reduction case is active)
            uV = 0.0
            if not shear_small:
                uV_y = (abs(Vy_Ed_kN) / Vc_y_Rd_kN) if Vc_y_Rd_kN > 0 else 0.0
                uV_z = (abs(Vz_Ed_kN) / Vc_z_Rd_kN) if Vc_z_Rd_kN > 0 else 0.0
                uV = (uV_y ** 2) + (uV_z ** 2)
        
            if shear_small:
                # EXACT MATCH to §6.2.9:
                # (12) ↔ (9): major only
                # (13) ↔ (10): minor only
                # (14) ↔ (11): biaxial interaction
                u12 = uMy if (uMy is not None) else None
                u13 = uMz if (uMz is not None) else None
                u14 = ((uMy ** alpha_y) + (uMz ** alpha_z)) if (uMy is not None and uMz is not None) else None
            else:
                # Shear present (use same bending-axial interaction form + add shear term)
                u12 = ((uMy ** alpha_y) + uV) if (uMy is not None) else None
                u13 = ((uMz ** alpha_z) + uV) if (uMz is not None) else None
                u14 = ((uMy ** alpha_y) + (uMz ** alpha_z) + uV) if (uMy is not None and uMz is not None) else None

            # ---- Show only main equations + final results (centered) ----
            _eq_line("Main interaction form:", r"u=\left(\frac{M_{y,Ed}}{M_{N,y,Rd}}\right)^{\alpha_y}+\left(\frac{M_{z,Ed}}{M_{N,z,Rd}}\right)^{\alpha_z}+u_V")
            if not shear_small:
                _eq_line("Shear term:", r"u_V=\left(\frac{V_{y,Ed}}{V_{c,y,Rd}}\right)^2+\left(\frac{V_{z,Ed}}{V_{c,z,Rd}}\right)^2")
        
            # Results (center column) + badges
            if u12 is not None:
                _eq_line("Utilization (12):", rf"u_{{y}} = {u12:.3f}\le 1.0")
                report_status_badge(u12)
        
            if u13 is not None:
                _eq_line("Utilization (13):", rf"u_{{z}} = {u13:.3f}\le 1.0")
                report_status_badge(u13)
        
            if u14 is not None:
                _eq_line("Utilization (14):", rf"u_{{y+z}} = {u14:.3f}\le 1.0")
                report_status_badge(u14)
        
            # Store for tables (so compute_checks can use later if you connect it)
            cs_combo.update({
                "u_6210_12": u12,
                "u_6210_13": u13,
                "u_6210_14": u14,
                "shear_small_6210": shear_small,
                "rho_6210": rho,
                "fy_red_6210": fy_red,
            })
        else:
            st.warning("Cannot compute §6.2.10 utilizations because reduced resistances could not be evaluated.")


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
    
        # Section & material (accept both naming styles from DB/session)
        A_mm2 = float(use_props.get("A_mm2", use_props.get("A_mm2", 0.0)) or 0.0)
    
        Iy_mm4 = float(
            use_props.get(
                "Iy_mm4",
                use_props.get("Iy_cm4", use_props.get("I_y_cm4", 0.0)) * 1e4
            ) or 0.0
        )
        Iz_mm4 = float(
            use_props.get(
                "Iz_mm4",
                use_props.get("Iz_cm4", use_props.get("I_z_cm4", 0.0)) * 1e4
            ) or 0.0
        )
    
        iy_mm = float(use_props.get("iy_mm", use_props.get("i_y_mm", 0.0)) or 0.0)
        iz_mm = float(use_props.get("iz_mm", use_props.get("i_z_mm", 0.0)) or 0.0)
    
        It_mm4 = float(
            use_props.get(
                "It_mm4",
                use_props.get("It_cm4", use_props.get("I_t_cm4", 0.0)) * 1e4
            ) or 0.0
        )
        Iw_mm6 = float(
            use_props.get(
                "Iw_mm6",
                use_props.get("Iw_cm6", use_props.get("I_w_cm6", 0.0)) * 1e6
            ) or 0.0
        )
    
        Wel_y_mm3 = float(use_props.get("Wel_y_mm3", use_props.get("Wel_y_cm3", 0.0) * 1e3) or 0.0)
        Wpl_y_mm3 = float(use_props.get("Wpl_y_mm3", use_props.get("Wpl_y_cm3", 0.0) * 1e3) or 0.0)
        Wel_z_mm3 = float(use_props.get("Wel_z_mm3", use_props.get("Wel_z_cm3", 0.0) * 1e3) or 0.0)
        Wpl_z_mm3 = float(use_props.get("Wpl_z_mm3", use_props.get("Wpl_z_cm3", 0.0) * 1e3) or 0.0)
    
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
        report_h4("(15), (16) Flexural buckling (EN 1993-1-1 §6.3.1)")
    
        # centered equation helper (same layout philosophy you used before)
        def _eq_center(latex_expr: str):
            cL, cM, cR = st.columns([3, 4, 3])
            with cM:
                st.latex(latex_expr)
    
        def _eq_line(label_html: str, latex_expr: str):
            cL, cM, cR = st.columns([3, 4, 3])
            with cL:
                st.markdown(label_html, unsafe_allow_html=True)
            with cM:
                st.latex(latex_expr)
    
        st.markdown(
            "Flexural buckling of the compression member is verified in accordance with **EN 1993-1-1 §6.3.1**. "
            "The design condition is:"
        )
        _eq_center(r"\frac{N_{Ed}}{N_{b,Rd}} \le 1.0")
        _eq_center(r"N_{b,Rd} = \chi\,\frac{A f_y}{\gamma_{M1}}")
    
        st.markdown(
            "The reduction factor $\\chi$ is evaluated for buckling about the **major (y–y)** and **minor (z–z)** axes."
        )
    
        E_MPa = 210000.0  # for display
    
        def _axis_report(axis: str,
                         K: float,
                         I_mm4: float,
                         curve_name: str,
                         alpha: float,
                         Ncr_kN: float,
                         lam_bar: float,
                         phi: float,
                         chi: float,
                         Nb_Rd_kN: float,
                         util: float):
    
            st.markdown(f"### Flexural buckling about axis {axis}–{axis}")
    
            # buckling length (all in mm)
            L_mm = L * 1000.0
            Lcr_mm = K * L_mm
    
            _eq_line(
                "Effective buckling length:",
                rf"L_{{cr,{axis}}}=K_{{{axis}}}L={K:.3f}\cdot {L_mm:.0f}"
                rf"={Lcr_mm:.0f}\,\mathrm{{mm}}"
            )
    
            _eq_line(
                "Elastic critical load:",
                rf"N_{{cr,{axis}}}=\frac{{\pi^2 E I_{{{axis}}}}}{{L_{{cr,{axis}}}^2}}"
            )
    
            if Lcr_mm > 0:
                Ncr_disp_kN = (math.pi**2 * E_MPa * I_mm4 / (Lcr_mm**2)) / 1000.0
            else:
                Ncr_disp_kN = 0.0
    
            # _eq_line(
            #     "&nbsp;",
            #     rf"=\frac{{\pi^2\cdot {E_MPa:.0f}\,\mathrm{{MPa}}\cdot {I_mm4:,.0f}\,\mathrm{{mm}}^4}}"
            #     rf"{{({Lcr_mm:.0f}\,\mathrm{{mm}})^2}}"
            #     rf"={Ncr_disp_kN:.1f}\,\mathrm{{kN}}"
            # )

            _eq_line(
                "&nbsp;",
                rf"N_{{cr,{axis}}}={Ncr_kN:.1f}\,\mathrm{{kN}}"
            )
                  
            # quick “ignore buckling” check (EN 1993-1-1 §6.3.1.2)
            ratio = (abs(NEd_kN) / Ncr_kN) if (Ncr_kN and Ncr_kN > 0) else None
            if ratio is not None:
                _eq_line("Check for neglecting buckling:", rf"\frac{{N_{{Ed}}}}{{N_{{cr,{axis}}}}}=\frac{{{abs(NEd_kN):.1f}}}{{{Ncr_kN:.1f}}}={ratio:.3f}")
    
            _eq_line("Non-dimensional slenderness:", rf"\bar{{\lambda}}_{{{axis}}}=\sqrt{{\frac{{A f_y}}{{N_{{cr,{axis}}}}}}}={lam_bar:.3f}")
    
            ignore_ok = ((ratio is not None and ratio <= 0.04) or (lam_bar is not None and lam_bar <= 0.20))
            if ignore_ok:
                st.markdown(
                    "The buckling effects may be **neglected** for this axis because "
                    r"$N_{Ed}/N_{cr}\le 0.04$ or $\bar{\lambda}\le 0.20$. "
                    "For completeness, the reduction factor and utilization are still shown below."
                )
            else:
                st.markdown("Buckling effects **cannot** be neglected for this axis; full reduction is applied.")
    
            # curve + imperfection factor
            st.markdown(f"- Buckling curve group: `{curve_name}`")
            _eq_line("Imperfection factor:", rf"\alpha={alpha:.2f}")
    
            # phi and chi
            _eq_line("Auxiliary factor:", rf"\Phi=\tfrac12\left[1+\alpha(\bar{{\lambda}}-{0.2})+\bar{{\lambda}}^2\right]={phi:.3f}")
            _eq_line("Reduction factor:", rf"\chi=\min\left(1,\frac{{1}}{{\Phi+\sqrt{{\Phi^2-\bar{{\lambda}}^2}}}}\right)={chi:.3f}")
    
            # buckling resistance + utilization
            _eq_line("Buckling resistance:", rf"N_{{b,Rd,{axis}}}=\chi\frac{{A f_y}}{{\gamma_{{M1}}}}={Nb_Rd_kN:.1f}\,\mathrm{{kN}}")
            _eq_line("Utilization:", rf"\frac{{N_{{Ed}}}}{{N_{{b,Rd,{axis}}}}}=\frac{{{abs(NEd_kN):.1f}}}{{{Nb_Rd_kN:.1f}}}={util:.3f}\le 1.0")
    
            report_status_badge(util)
    
        _axis_report(
            axis="y",
            K=float(inputs.get("K_y", 1.0)),
            I_mm4=Iy_mm4,
            curve_name=str(sr_display.get("imperfection_group") or sr_display.get("buckling_curve_y") or "c"),
            alpha=float(extras.get("buck_alpha_y") or 0.49),
            Ncr_kN=float((buck_map.get("Ncr_y") or 0.0) / 1000.0),
            lam_bar=float(buck_map.get("lambda_y") or 0.0),
            phi=float(buck_map.get("phi_y") or 0.0),
            chi=float(buck_map.get("chi_y") or 0.0),
            Nb_Rd_kN=float((buck_map.get("Nb_Rd_y") or 0.0) / 1000.0),
            util=float(buck_map.get("util_buck_y") or 0.0),
        )
    
        _axis_report(
            axis="z",
            K=float(inputs.get("K_z", 1.0)),
            I_mm4=Iz_mm4,
            curve_name=str(sr_display.get("imperfection_group") or sr_display.get("buckling_curve_z") or "c"),
            alpha=float(extras.get("buck_alpha_z") or 0.49),
            Ncr_kN=float((buck_map.get("Ncr_z") or 0.0) / 1000.0),
            lam_bar=float(buck_map.get("lambda_z") or 0.0),
            phi=float(buck_map.get("phi_z") or 0.0),
            chi=float(buck_map.get("chi_z") or 0.0),
            Nb_Rd_kN=float((buck_map.get("Nb_Rd_z") or 0.0) / 1000.0),
            util=float(buck_map.get("util_buck_z") or 0.0),
        )
    
        st.markdown(
            "Note: the buckling resistance obtained is applicable for compression members with end fastener holes neglected, "
            "consistent with the assumptions used for member stability checks."
        )

        # ----------------------------
        # (17) Torsional & torsional-flexural buckling
        # ----------------------------
        report_h4("(17) Torsional and torsional-flexural buckling – EN 1993-1-1 §6.3.1.4")
        
        family = str(sr_display.get("family", "") or "").upper()
        is_closed_section = any(tag in family for tag in ("RHS", "SHS", "CHS"))
        
        if is_closed_section:
            st.markdown(
                "According to **EN 1993-1-1 §6.3.1.4(1)**, torsional and torsional-flexural buckling checks are relevant "
                "for members with **open cross-sections**. "
                f"The selected section is a **closed hollow section ({family})**, therefore this check is **not applicable**. "
                "Member stability is covered by **flexural buckling about y–y and z–z** (EN 1993-1-1 §6.3.1)."
            )
            report_status_badge(0.0)  # green OK tick
        
        else:
            # ---- KEEP EVERYTHING BELOW EXACTLY AS YOU HAVE IT (NO CHANGES) ----
            st.markdown(
                "Torsional and torsional-flexural buckling are treated in **EN 1993-1-1 §6.3.1.4**. "
                "For typical rolled **I/H sections** these checks are often not governing compared to flexural buckling, "
                "but they are reported here for completeness."
            )
        
            # --- Geometry: polar radius of gyration i0 (mm) ---
            i0_mm = (i0_m * 1e3) if (i0_m is not None) else 0.0
            st.latex(
                rf"""
                \begin{{aligned}}
                i_0 &= \sqrt{{i_y^2 + i_z^2 + y_0^2 + z_0^2}} \\[4pt]
                    &= \sqrt{{({iy_mm:.1f})^2 + ({iz_mm:.1f})^2 + 0^2 + 0^2}} \\
                    &= {i0_mm:.1f}\,\text{{mm}}
                \end{{aligned}}
                """
            )
            # --- Effective torsional buckling length ---
            L_mm = float(L) * 1000.0
            LcrT_mm = float(K_T) * L_mm
            _eq_line("Effective torsional buckling length:", r"L_{cr,T}=K_T\,L")
            _eq_line("&nbsp;", rf"={K_T:.3f}\cdot {L_mm:.0f}={LcrT_mm:.0f}\,\mathrm{{mm}}")
        
            # --- Elastic critical force for torsional buckling ---
            E_MPa = 210000.0
            G_MPa = 80769.0
            if i0_mm > 0 and It_mm4 > 0 and Iw_mm6 > 0 and LcrT_mm > 0:
                NcrT_disp_kN = ((1.0 / (i0_mm**2)) * (G_MPa * It_mm4 + (math.pi**2) * E_MPa * Iw_mm6 / (LcrT_mm**2))) / 1000.0
            else:
                NcrT_disp_kN = 0.0
        
            _eq_line(
                "Elastic critical force:",
                r"N_{cr,T}=\frac{1}{i_0^2}\left(G I_T+\frac{\pi^2 E I_w}{L_{cr,T}^2}\right)"
            )
        
            _eq_line(
                "&nbsp;",
                rf"={NcrT_disp_kN:.1f}\,\mathrm{{kN}}"
            )
        
            st.markdown(
                "For **doubly symmetric** sections (shear centre at the centroid: $y_0=z_0=0$), "
                "the torsional-flexural critical load is commonly taken equal to the torsional one: "
                "$N_{cr,TF}=N_{cr,T}$ (see EN 1993-1-1 §6.3.1.4)."
            )
        
            # --- (17) Pull values from compute_checks (same pattern as section 16) ---
            # --- Non-dimensional slenderness (torsional / torsional-flexural) ---
            NcrT_val = float(Ncr_T or 0.0)
        
            # Your report sometimes has Ncr_T in kN, sometimes in N (depends what you pass in).
            # Make it N for lambda calculation:
            NcrT_N = NcrT_val * 1e3 if (0.0 < NcrT_val < 1e5) else NcrT_val
        
            alpha_T = float(buck_map.get("alpha_T") or buck_map.get("alpha_z") or 0.34)  # consistent with checks
            lam_T = math.sqrt((A_mm2 * fy) / NcrT_N) if NcrT_N > 0 else 0.0
        
            phi_T = float(buck_map.get("phi_T") or 0.0)
            chi_T_disp = float(buck_map.get("chi_T") or 0.0)
        
            NbRdT_disp_kN = float((buck_map.get("Nb_Rd_T") or 0.0) / 1000.0)
            utilT_disp = float(buck_map.get("util_T") or 0.0)
        
            _eq_line("Non-dimensional slenderness:", r"\bar{\lambda}_T=\sqrt{\frac{A f_y}{N_{cr,T}}}")
            _eq_line("&nbsp;", rf"=\sqrt{{\frac{{{A_mm2:,.0f}\cdot {fy:.0f}}}{{{NcrT_disp_kN:.1f}\times 10^3}}}}={lam_T:.3f}")
        
            _eq_line("Auxiliary factor:", r"\Phi_T=\frac{1}{2}\left[1+\alpha_T(\bar{\lambda}_T-0.2)+\bar{\lambda}_T^2\right]")
            _eq_line("&nbsp;", rf"=\frac{{1}}{{2}}\left[1+{alpha_T:.2f}({lam_T:.3f}-0.2)+{lam_T:.3f}^2\right]={phi_T:.3f}")
        
            _eq_line("Reduction factor:", r"\chi_T=\min\left(1,\frac{1}{\Phi_T+\sqrt{\Phi_T^2-\bar{\lambda}_T^2}}\right)")
            _eq_line("&nbsp;", rf"={chi_T_disp:.3f}")
        
            _eq_line("Design buckling resistance:", r"N_{b,Rd,T}=\chi_T\,\frac{A f_y}{\gamma_{M1}}")
            _eq_line("&nbsp;", rf"={chi_T_disp:.3f}\cdot\frac{{{A_mm2:,.0f}\cdot {fy:.0f}}}{{{gamma_M1:.2f}}}={NbRdT_disp_kN:.1f}\,\mathrm{{kN}}")
        
            _eq_line("Utilization:", r"u_T=\frac{N_{Ed}}{N_{b,Rd,T}}")
            _eq_line("&nbsp;", rf"=\frac{{{abs(NEd_kN):.1f}}}{{{NbRdT_disp_kN:.1f}}}={utilT_disp:.3f}")
            report_status_badge(utilT_disp)

        # ----------------------------
        # (18) Lateral-torsional buckling
        # ----------------------------
        report_h4("(18) Lateral-torsional buckling – EN 1993-1-1 §6.3.2")
        
        family = str(sr_display.get("family", "") or "").upper()
        is_closed_section = any(tag in family for tag in ("RHS", "SHS", "CHS"))
        
        if is_closed_section:
            st.markdown(
                "Lateral-torsional buckling in major-axis bending is treated in **EN 1993-1-1 §6.3.2** and is relevant "
                "for members with **open cross-sections** that can twist and deflect laterally. "
                f"The selected section is a **closed hollow section ({family})**, which has high torsional rigidity; "
                "therefore the lateral-torsional buckling verification is **not applicable** here. "
                "Member stability is covered by **flexural buckling about y–y and z–z** (EN 1993-1-1 §6.3.1)."
            )
            report_status_badge(0.0)  # green OK tick
        
        else:
            st.markdown(
                "A laterally unrestrained member in **major-axis bending** should be verified against lateral-torsional buckling "
                "in accordance with **EN 1993-1-1 §6.3.2** (see also §8.3.2). The design condition is:"
            )
            _eq_center(r"\frac{M_{Ed}}{M_{b,Rd}}\le 1.0")
            _eq_center(r"M_{b,Rd}=\chi_{LT}\,\frac{M_{Rk}}{\gamma_{M1}}")
        
            # Characteristic bending resistance for display (consistent with compute_checks)
            Wy_mm3 = (Wpl_y_mm3 if (Wpl_y_mm3 > 0) else Wel_y_mm3)
            MRk_kNm = (Wy_mm3 * fy) / 1e6 if Wy_mm3 > 0 else 0.0
            _eq_line("Characteristic resistance:", r"M_{Rk}=W_y f_y")
            _eq_line("&nbsp;", rf"={Wy_mm3:,.0f}\cdot {fy:.0f}={MRk_kNm:.1f}\,\mathrm{{kNm}}")
        
            # Elastic critical moment for LTB
            Mcr_kNm = (Mcr / 1e3) if (Mcr is not None) else 0.0
            lamLT = float(lam_LT or 0.0)
            chiLT_disp = float(chi_LT or 0.0)
            MbRd_kNm = (Mb_Rd / 1e3) if (Mb_Rd is not None) else 0.0
            utilLT_disp = float(util_LT if util_LT is not None else float("inf"))
        
            _eq_line("Elastic critical moment:", r"M_{cr}\;\text{(from gross section properties)}")
            _eq_line("&nbsp;", rf"={Mcr_kNm:.1f}\,\mathrm{{kNm}}")
        
            _eq_line("Relative slenderness:", r"\bar{\lambda}_{LT}=\sqrt{\frac{M_{Rk}}{M_{cr}}}")
            _eq_line("&nbsp;", rf"=\sqrt{{\frac{{{MRk_kNm:.1f}}}{{{Mcr_kNm:.1f}}}}}={lamLT:.3f}")
        
            # Reduction factor (EN 1993-1-1 §6.3.2 / §8.3.2)
            # --- LTB imperfection factor alpha_LT (EN 1993-1-1 Table 6.3 + 6.4) ---
            # Decide curve based on cross-section type + h/b
            # Rolled I:   h/b <= 2 -> a (0.21),  h/b > 2 -> b (0.34)
            # Welded I:   h/b <= 2 -> c (0.49),  h/b > 2 -> d (0.76)
            # Other:      d (0.76)
        
            hb = (h_mm / b_mm) if (b_mm and b_mm > 0) else None
        
            # Try to infer "rolled vs welded vs other" from whatever label exists in use_props
            sec_label = (
                str(use_props.get("family") or use_props.get("type") or use_props.get("Type") or
                    use_props.get("section_family") or use_props.get("Section") or use_props.get("name") or "")
            ).upper()
        
            is_welded_i = ("WELD" in sec_label) or ("PLATE" in sec_label) or ("GIRDER" in sec_label)
            is_rolled_i = any(sec_label.startswith(p) for p in ("IPE", "IPN", "HEA", "HEB", "HEM", "HE", "UB", "UC"))
        
            if hb is None:
                curve_LT = "b"  # safe default if geometry missing
            elif is_welded_i:
                curve_LT = "c" if hb <= 2.0 else "d"
            elif is_rolled_i:
                curve_LT = "a" if hb <= 2.0 else "b"
            else:
                curve_LT = "d"
        
            alpha_LT = {"a": 0.21, "b": 0.34, "c": 0.49, "d": 0.76}[curve_LT]
        
            lamLT0 = 0.40
            beta_LT = 0.75
            phiLT = 0.5 * (1.0 + alpha_LT * (lamLT - lamLT0) + beta_LT * lamLT**2)
        
            _eq_line("Auxiliary factor:", r"\Phi_{LT}=\frac{1}{2}\left[1+\alpha_{LT}(\bar{\lambda}_{LT}-\bar{\lambda}_{LT,0})+\beta\bar{\lambda}_{LT}^2\right]")
            _eq_line("&nbsp;", rf"=\frac{{1}}{{2}}\left[1+{alpha_LT:.2f}({lamLT:.3f}-{lamLT0:.2f})+{beta_LT:.2f}{lamLT:.3f}^2\right]={phiLT:.3f}")
        
            _eq_line("Reduction factor:", r"\chi_{LT}=\min\left(1,\frac{1}{\Phi_{LT}+\sqrt{\Phi_{LT}^2-\bar{\lambda}_{LT}^2}}\right)")
            _eq_line("&nbsp;", rf"={chiLT_disp:.3f}")
        
            _eq_line("Design LTB resistance:", r"M_{b,Rd}=\chi_{LT}\,\frac{M_{Rk}}{\gamma_{M1}}")
            _eq_line("&nbsp;", rf"={chiLT_disp:.3f}\cdot\frac{{{MRk_kNm:.1f}}}{{{gamma_M1:.2f}}}={MbRd_kNm:.1f}\,\mathrm{{kNm}}")
        
            MyEd_kNm = float(My_Ed_kNm)
            _eq_line("Utilization:", r"u_{LT}=\frac{|M_{Ed}|}{M_{b,Rd}}")
            _eq_line("&nbsp;", rf"=\frac{{{abs(MyEd_kNm):.1f}}}{{{MbRd_kNm:.1f}}}={utilLT_disp:.3f}")
            report_status_badge(utilLT_disp)
          
        # ----------------------------
        # (19),(20) Buckling interaction for bending and axial compression — EN 1993-1-1 Annex B (Method 2)
        # ----------------------------
        
        # We only keep Annex B Method 2 (your attached example) and we label the two expressions as (19) and (20).
        
        report_h4("(19),(20) Buckling interaction for bending and axial compression — Method 2 (EN 1993-1-1 Annex B)")
        
        # Pull values from buck_map (computed in compute_checks)
        psi_y  = float(buck_map.get("psi_y", 1.0) or 1.0)
        psi_z  = float(buck_map.get("psi_z", 1.0) or 1.0)
        psi_LT = float(buck_map.get("psi_LT", 1.0) or 1.0)
        
        lam_y = float(buck_map.get("lam_y", 0.0) or 0.0)
        lam_z = float(buck_map.get("lam_z", 0.0) or 0.0)
        
        # Moment factors (Table B.3)
        Cmy_B  = float(buck_map.get("Cmy_B", 0.0) or 0.0)
        Cmz_B  = float(buck_map.get("Cmz_B", 0.0) or 0.0)
        CmLT_B = float(buck_map.get("CmLT_B", 0.0) or 0.0)
        
        # Interaction factors (Table B.2)
        kyy_B = float(buck_map.get("kyy_B", 0.0) or 0.0)
        kzz_B = float(buck_map.get("kzz_B", 0.0) or 0.0)
        kyz_B = float(buck_map.get("kyz_B", 0.0) or 0.0)
        kzy_B = float(buck_map.get("kzy_B", 0.0) or 0.0)
        
        # Utilizations: prefer the NEW keys (util_19_B/util_20_B).
        # If you didn’t update compute_checks yet and still store util_61_B/util_62_B, this fallback keeps it working.
        util_19 = buck_map.get("util_19_B", buck_map.get("util_61_B"))
        util_20 = buck_map.get("util_20_B", buck_map.get("util_62_B"))
        
        # Optional breakdown terms (if you stored them)
        tNy  = buck_map.get("term_Ny_B")
        tMy1 = buck_map.get("term_My1_B")
        tMz1 = buck_map.get("term_Mz1_B")
        
        tNz  = buck_map.get("term_Nz_B")
        tMy2 = buck_map.get("term_My2_B")
        tMz2 = buck_map.get("term_Mz2_B")
        
        st.markdown(
            "This verification follows **EN 1993-1-1 Annex B (Method 2)** as in your attached example. "
            "Moment factors are from **Table B.3** and interaction factors are from **Table B.2** (I-sections susceptible to LTB)."
        )
        
        # --- Equivalent uniform moment factors (Table B.3) ---
        st.markdown("### Equivalent uniform moment factors (Annex B, Table B.3)")
        st.latex(rf"""
        \begin{{aligned}}
        C_{{my}} &= \max(0.4,\;0.60+0.40\,\psi_y) = {Cmy_B:.3f} \\
        C_{{mz}} &= \max(0.4,\;0.60+0.40\,\psi_z) = {Cmz_B:.3f} \\
        C_{{mLT}} &= \max(0.4,\;0.60+0.40\,\psi_{{LT}}) = {CmLT_B:.3f}
        \end{{aligned}}
        """)
        st.latex(rf"""
        \begin{{aligned}}
        \psi_y &= {psi_y:.3f},\qquad
        \psi_z = {psi_z:.3f},\qquad
        \psi_{{LT}} = {psi_LT:.3f}
        \end{{aligned}}
        """)
        
        # --- Slenderness values (reported) ---
        st.markdown("### Slenderness values used (Annex B)")
        st.latex(rf"""
        \begin{{aligned}}
        \bar\lambda_y &= {lam_y:.3f}, \qquad
        \bar\lambda_z = {lam_z:.3f}
        \end{{aligned}}
        """)
        
        # --- Interaction factors (Table B.2) ---
        st.markdown("### Interaction factors (Annex B, Table B.2)")
        st.latex(r"""
        \begin{aligned}
        k_{yy} &= C_{my}\left[1+\left(\min(\bar\lambda_y,1.0)-0.2\right)\frac{N_{Ed}}{\chi_y N_{Rk}/\gamma_{M1}}\right] \\
        k_{zz} &= C_{mz}\left[1+\left(2\min(\bar\lambda_z,1.0)-0.6\right)\frac{N_{Ed}}{\chi_z N_{Rk}/\gamma_{M1}}\right] \\
        k_{yz} &= 0.6\,k_{zz} \\
        k_{zy} &= 1-\frac{0.1\,\min(\bar\lambda_z,1.0)}{(C_{mLT}-0.25)}\frac{N_{Ed}}{\chi_z N_{Rk}/\gamma_{M1}}
        \end{aligned}
        """)
        st.latex(rf"""
        \begin{{aligned}}
        k_{{yy}} &= {kyy_B:.3f}, \qquad
        k_{{zz}} = {kzz_B:.3f}, \qquad
        k_{{yz}} = {kyz_B:.3f}, \qquad
        k_{{zy}} = {kzy_B:.3f}
        \end{{aligned}}
        """)
        
        # --- Verification expressions (now labeled as 19 and 20 only) ---
        st.markdown("### Verification of member resistance (Annex B)")
        
        u19 = float(util_19) if util_19 is not None else float("nan")
        u20 = float(util_20) if util_20 is not None else float("nan")
        
        st.markdown("**(19) Expression 1 (about y):**")
        st.latex(r"""
        \frac{N_{Ed}}{\chi_y N_{Rk}/\gamma_{M1}}
        +k_{yy}\frac{M_{y,Ed}}{\chi_{LT} M_{y,Rk}/\gamma_{M1}}
        +k_{yz}\frac{M_{z,Ed}}{M_{z,Rk}/\gamma_{M1}}
        \le 1.0
        """)
        if (tNy is not None) and (tMy1 is not None) and (tMz1 is not None):
            st.latex(rf"u_{{y}} = {float(tNy):.3f} + {float(tMy1):.3f} + {float(tMz1):.3f} = {u19:.3f}")
        else:
            st.latex(rf"u_{{y}} = {u19:.3f}")
        report_status_badge(util_19)
        
        st.markdown("**(20) Expression 2 (about z):**")
        st.latex(r"""
        \frac{N_{Ed}}{\chi_z N_{Rk}/\gamma_{M1}}
        +k_{zy}\frac{M_{y,Ed}}{\chi_{LT} M_{y,Rk}/\gamma_{M1}}
        +k_{zz}\frac{M_{z,Ed}}{M_{z,Rk}/\gamma_{M1}}
        \le 1.0
        """)
        if (tNz is not None) and (tMy2 is not None) and (tMz2 is not None):
            st.latex(rf"u_{{z}} = {float(tNz):.3f} + {float(tMy2):.3f} + {float(tMz2):.3f} = {u20:.3f}")
        else:
            st.latex(rf"u_{{z}} = {u20:.3f}")
        report_status_badge(util_20)
        
        # Optional governing (still useful)
        u_g = max(u19, u20)
        st.markdown("Governing utilization (Annex B):")
        st.latex(rf"u_g = \max(u_{{y}},u_{{z}}) = {u_g:.3f}")
        report_status_badge(u_g)

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
    
    # Close report root wrapper (important for proper printing)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# APP ENTRY
# =========================================================
# --- PAGE CONFIG ---
st.set_page_config(
    page_title="EngiSnap Beam Design Eurocode Checker",
    page_icon=str(asset_path("EngiSnap-Logo.png")) if asset_path("EngiSnap-Logo.png").exists() else "🧰",
    layout="wide"
)
# --- GLOBAL PRINT FIX (Edge / Streamlit) ---
# --- GLOBAL PRINT FIX (Streamlit tabs + full-page printing) ---
st.markdown(
    """
    <style>
    @media print {

      /* Hide Streamlit chrome */
      header, footer, #MainMenu { display: none !important; }
      section[data-testid="stSidebar"] { display: none !important; }

      /* Page setup */
      @page { size: A4; margin: 12mm; }

      /* Keep colors */
      * {
        -webkit-print-color-adjust: exact !important;
        print-color-adjust: exact !important;
      }

      /* Hide anything you wrapped as no-print */
      /* Hide anything you wrapped as no-print */
      .no-print, .no-print * { display: none !important; }
    
      /* also hide Streamlit's debug toolbar if it appears */
      div[data-testid="stToolbar"] { display: none !important; }


      /* Hide ONLY the tab headers (NOT the tab content) */
      div[data-testid="stTabs"] [role="tablist"] { display: none !important; }

      /* THE IMPORTANT PART:
         Tabs/containers often have fixed height + overflow that clips printing.
         Force everything to expand fully for print. */
      html, body { height: auto !important; overflow: visible !important; }

     div[data-testid="stAppViewContainer"],
     div[data-testid="stAppViewContainer"] > .main,
     section.main,
     div.block-container,
     div[data-testid="stTabs"],
     div[data-testid="stHorizontalBlock"],
     div[data-testid="stVerticalBlock"],
     div.element-container,
     div.stMarkdown,
     div[data-testid="stToolbar"] {
     height: auto !important;
     max-height: none !important;
     overflow: visible !important;
     }
      /* Streamlit tab panel containers (BaseWeb) */
      div[data-baseweb="tab-panel"],
      div[data-baseweb="tab-panel"] > div {
        height: auto !important;
        max-height: none !important;
        overflow: visible !important;
      }

      /* Some Streamlit wrappers still clip */
      div[data-testid="stVerticalBlock"],
      div[data-testid="stVerticalBlock"] > div,
      div[data-testid="stMarkdownContainer"] {
        overflow: visible !important;
        height: auto !important;
        max-height: none !important;
      }

      /* Make sure your report wrapper itself never clips */
      #engi_report_root {
        height: auto !important;
        overflow: visible !important;
        max-height: none !important;
      }
    }
    </style>
    """,
    unsafe_allow_html=True
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
    padding-top: 0.8rem;
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
st.markdown("</div>", unsafe_allow_html=True)

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


# =========================================================
# FRAME DESIGN APP (Beam + Column) — based on the beam app
# =========================================================

def _render_instability_ratios_member(member_prefix: str, member_title: str):
    """
    UI only: show instability length ratios like the beam app.
    Internally we still store them in the existing keys:
      K_y_in, K_z_in, K_LT_in, K_T_in
    """
    st.markdown(f"**{member_title}**")

    help_y = (
        "Flexural buckling about y–y.\n"
        "Lcr,y / L = 1.0 → pinned–pinned typical.\n"
        "Smaller = more restraint; larger = less restraint."
    )
    help_z = (
        "Flexural buckling about z–z.\n"
        "Lcr,z / L = 1.0 → pinned–pinned typical.\n"
        "Smaller = more restraint; larger = less restraint."
    )
    help_lt = (
        "Lateral–torsional buckling length ratio.\n"
        "For braced beams this can be < 1.\n"
        "For unbraced length equal to span → 1.0."
    )
    help_tf = (
        "Torsional / flexural–torsional buckling length ratio.\n"
        "Often relevant for open sections.\n"
        "Use 1.0 if uncertain."
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.number_input(
            "Lcr,y / L (flexural y–y)",
            min_value=0.1,
            value=float(st.session_state.get(f"{member_prefix}K_y_in", 1.0)),
            step=0.05,
            key=f"{member_prefix}K_y_in",
            help=help_y,
        )
    with c2:
        st.number_input(
            "Lcr,z / L (flexural z–z)",
            min_value=0.1,
            value=float(st.session_state.get(f"{member_prefix}K_z_in", 1.0)),
            step=0.05,
            key=f"{member_prefix}K_z_in",
            help=help_z,
        )
    with c3:
        st.number_input(
            "L_LT / L (lateral–torsional)",
            min_value=0.1,
            value=float(st.session_state.get(f"{member_prefix}K_LT_in", 1.0)),
            step=0.05,
            key=f"{member_prefix}K_LT_in",
            help=help_lt,
        )
    with c4:
        st.number_input(
            "L_TF / L (torsional / flexural–torsional)",
            min_value=0.1,
            value=float(st.session_state.get(f"{member_prefix}K_T_in", 1.0)),
            step=0.05,
            key=f"{member_prefix}K_T_in",
            help=help_tf,
        )


def _render_section_selection_member(member_prefix: str, title: str):
    """Same as render_section_selection(), but with isolated Streamlit keys per member."""
    st.subheader(title)
    types, sizes_map, detected_table = fetch_types_and_sizes()

    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        material = st.selectbox(
            "Material",
            ["S235", "S275", "S355", "S450"],
            index=2,
            key=f"{member_prefix}mat_sel",
        )

    with c2:
        family = st.selectbox(
            "Section family / Type (DB)",
            ["-- choose --"] + types if types else ["-- choose --"],
            key=f"{member_prefix}fam_sel",
        )

    with c3:
        selected_name = None
        selected_row = None
        if family and family != "-- choose --":
            names = sizes_map.get(family, [])
            selected_name = st.selectbox(
                "Size / Name",
                ["-- choose --"] + names if names else ["-- choose --"],
                key=f"{member_prefix}size_sel",
            )
            if selected_name and selected_name != "-- choose --":
                selected_row = get_section_row_db(
                    family, selected_name,
                    None if (detected_table in (None, "sample")) else detected_table
                )


    return material, family, selected_name, selected_row, detected_table

def _render_member_section_panel(member_prefix: str, title: str):
    material, family, selected_name, selected_row, detected_table = _render_section_selection_member(member_prefix, title)

    # Store to session (member-specific)
    st.session_state[f"{member_prefix}material"] = material

    if selected_row is None:
        st.info("Select a section to continue.")
        return

    # Build display struct
    sr_display, bad_fields = build_section_display(selected_row)
    st.session_state[f"{member_prefix}sr_display"] = sr_display

    prefix_id = f"{member_prefix}{family}_{selected_name}".replace(" ", "_")

    # Summary (same as beam app)
    render_section_summary_like_props(
        material,
        sr_display,
        key_prefix=f"sum_{prefix_id}"
    )

    # -----------------------------
    # Axis selection (Option A)
    # -----------------------------
    st.markdown("### Bending axis in frame plane")
    default_val = st.session_state.get(f"{member_prefix}bend_axis_sel", "Strong axis (yy)")
    idx = 0 if str(default_val).lower().startswith("strong") else 1
    st.selectbox(
        "Use bending about",
        ["Strong axis (yy)", "Weak axis (zz)"],
        index=idx,
        key=f"{member_prefix}bend_axis_sel",
        help="Strong axis = yy (default, higher stiffness). Weak axis = zz."
    )

    # -----------------------------
    # Cross-section preview
    # -----------------------------
    st.markdown("### Cross-section preview")
    img_path = get_section_image(sr_display.get("family", ""))
    _l, _m, _r = st.columns([0.45, 1.1, 0.45])
    with _m:
        if img_path and Path(img_path).exists():
            st.image(img_path, use_container_width=True)
        else:
            render_section_preview_placeholder(
                title=f"{sr_display.get('family','')} {sr_display.get('name','')}",
                key_prefix=f"prev_{prefix_id}"
            )

    # -----------------------------
    # Properties table (readonly)
    # -----------------------------
    with st.expander("Section properties", expanded=False):
        render_section_properties_readonly(
            sr_display,
            key_prefix=f"db_{prefix_id}"
        )

    # -----------------------------
    # Frame sketch (FIXED: no L_m/h_m/iax NameErrors)
    # -----------------------------
    # Read lengths from your widget keys
    L_mm_key = f"{member_prefix}L_mm_in"
    L_mm = float(st.session_state.get(L_mm_key, 6000.0 if member_prefix == "beam_" else 3000.0))
    h_mm = float(st.session_state.get("col_L_mm_in", 3000.0))

    L = L_mm / 1000.0
    h = h_mm / 1000.0
    if L <= 0 or h <= 0:
        return

def _render_design_settings():
    """
    Beam-app style Design settings (ULS) box.
    - γF radio: 1.35 / 1.50 / Custom
    - Manual internal forces: Characteristic vs Design values
    Stores:
      st.session_state["gamma_F"]
      st.session_state["manual_forces_type"]
    """
    # Defaults
    if "gamma_F" not in st.session_state:
        st.session_state["gamma_F"] = 1.35
    if "manual_forces_type" not in st.session_state:
        st.session_state["manual_forces_type"] = "Characteristic"

    with st.expander("Design settings (ULS)", expanded=True):
        st.markdown("##### Load factor γ_F")

        # Decide which option is currently active
        g = float(st.session_state.get("gamma_F", 1.35))
        if abs(g - 1.35) < 1e-6:
            gamma_choice = "1.35 (static)"
        elif abs(g - 1.50) < 1e-6:
            gamma_choice = "1.50 (dynamic)"
        else:
            gamma_choice = "Custom"

        gamma_choice = st.radio(
            "Load factor γ_F",
            ["1.35 (static)", "1.50 (dynamic)", "Custom"],
            index=["1.35 (static)", "1.50 (dynamic)", "Custom"].index(gamma_choice),
            key="gammaF_choice_ui",
            label_visibility="collapsed",
        )

        if gamma_choice.startswith("1.35"):
            st.session_state["gamma_F"] = 1.35
        elif gamma_choice.startswith("1.50"):
            st.session_state["gamma_F"] = 1.50
        else:
            st.session_state["gamma_F"] = st.number_input(
                "Custom γ_F",
                min_value=1.0,
                value=float(st.session_state.get("gamma_F", 1.35)),
                step=0.05,
                key="gammaF_custom_ui",
            )

        st.markdown("##### Manual internal forces are")

        mf = st.session_state.get("manual_forces_type", "Characteristic")
        # Normalize (just in case old string exists)
        if str(mf).lower().startswith("design"):
            mf_idx = 1
        else:
            mf_idx = 0

        manual_choice = st.radio(
            "Manual internal forces are",
            ["Characteristic", "Design values (N_Ed, M_Ed, ...)"],
            index=mf_idx,
            key="manual_forces_ui",
            label_visibility="collapsed",
        )

        if manual_choice.startswith("Characteristic"):
            st.session_state["manual_forces_type"] = "Characteristic"
        else:
            st.session_state["manual_forces_type"] = "Design"
            

def _store_design_forces_from_state_member(member_prefix: str, inputs_key: str):
    """Compute design ULS forces from member-prefixed Loads inputs and store into st.session_state[inputs_key]."""

    # Raw loads (UI uses mm, kN, kN·m)
    L = float(st.session_state.get(f"{member_prefix}L_mm_in", 0.0)) / 1000.0  # mm → m
    N_kN  = float(st.session_state.get(f"{member_prefix}N_in", 0.0))
    Vy_kN = float(st.session_state.get(f"{member_prefix}Vy_in", 0.0))
    Vz_kN = float(st.session_state.get(f"{member_prefix}Vz_in", 0.0))
    My_kNm = float(st.session_state.get(f"{member_prefix}My_in", 0.0))
    Mz_kNm = float(st.session_state.get(f"{member_prefix}Mz_in", 0.0))

    # Beam torsion only (keep key stable even if hidden elsewhere)
    Tx_kNm = float(st.session_state.get(f"{member_prefix}Tx_in", 0.0)) if member_prefix == "beam_" else 0.0

    # NEW UI (ratios)
    r_y  = float(st.session_state.get(f"{member_prefix}Lcr_y_ratio", 1.0))
    r_z  = float(st.session_state.get(f"{member_prefix}Lcr_z_ratio", 1.0))
    r_LT = float(st.session_state.get(f"{member_prefix}L_LT_ratio",  1.0))
    r_TF = float(st.session_state.get(f"{member_prefix}L_TF_ratio",  1.0))

    # Map ratios → the old “K” names your check engine already expects
    K_y  = r_y
    K_z  = r_z
    K_LT = r_LT
    K_T  = r_TF

    # Design settings
    gamma_F = float(st.session_state.get("gamma_F", 1.35))
    manual_forces_type = str(st.session_state.get("manual_forces_type", "Characteristic"))

    factor = gamma_F if manual_forces_type.lower().startswith("character") else 1.0

    # Design (Ed)
    st.session_state[inputs_key] = dict(
        L=L,
        N_kN=N_kN * factor,
        Vy_kN=Vy_kN * factor,
        Vz_kN=Vz_kN * factor,
        My_kNm=My_kNm * factor,
        Mz_kNm=Mz_kNm * factor,
        Tx_kNm=Tx_kNm * factor,
        K_y=K_y,
        K_z=K_z,
        K_LT=K_LT,
        K_T=K_T,
    )

# ============================================================
# FRAME HELPERS (TM-PR) — keep this block only ONCE
# - E getter (future user input ready)
# - strong/weak axis mapping
# - write V/M to correct member components (Vz/My or Vy/Mz)
# - small diagram generators
# - plotting helper (V/M)
# - apply helper (fills session_state for ready cases)
# ============================================================

from typing import Optional
import io
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# -----------------------------
# Global elastic modulus (Pa)
# -----------------------------
def get_E_Pa() -> float:
    """
    Global elastic modulus getter (Pa).
    Default: 210 GPa (steel).

    Later: add a UI number_input that writes st.session_state["E_GPa"].
    """
    E_GPa = float(st.session_state.get("E_GPa", 210.0))
    if E_GPa <= 0:
        E_GPa = 210.0
    return E_GPa * 1e9


# -----------------------------
# Axis selection (strong / weak)
# -----------------------------
def _axis_is_weak(member_prefix: str) -> bool:
    """
    True if member uses weak-axis bending (zz).
    Reads selector: st.session_state[f"{member_prefix}bend_axis_sel"]
    """
    sel = st.session_state.get(f"{member_prefix}bend_axis_sel", "Strong axis (yy)")
    return isinstance(sel, str) and sel.lower().startswith("weak")


# -----------------------------
# Write envelope V/M into member inputs
# -----------------------------
def _fill_VM_into_member_inputs(member_prefix: str, Vmax_kN: float, Mmax_kNm: float) -> None:
    """
    Writes V and M into the correct session_state inputs depending on axis:

    Strong axis -> Vz_in, My_in
    Weak axis   -> Vy_in, Mz_in
    """
    weak = _axis_is_weak(member_prefix)

    if weak:
        st.session_state[f"{member_prefix}Vy_in"] = float(Vmax_kN)
        st.session_state[f"{member_prefix}Vz_in"] = 0.0
        st.session_state[f"{member_prefix}My_in"] = 0.0
        st.session_state[f"{member_prefix}Mz_in"] = float(Mmax_kNm)
    else:
        st.session_state[f"{member_prefix}Vy_in"] = 0.0
        st.session_state[f"{member_prefix}Vz_in"] = float(Vmax_kN)
        st.session_state[f"{member_prefix}My_in"] = float(Mmax_kNm)
        st.session_state[f"{member_prefix}Mz_in"] = 0.0


# -----------------------------
# Optional: frame inset (if you later want it)
# -----------------------------
def _add_frame_inset(ax, L_m: float, h_m: float) -> None:
    """Small frame sketch inset so main plot y-axis can stay V/M units."""
    iax = ax.inset_axes([0.30, 0.62, 0.40, 0.33])  # [x0,y0,w,h] in axes fraction
    iax.set_axis_off()

    if L_m <= 0 or h_m <= 0:
        return

    FRAME_COLOR = "#1f77b4"
    LW = 3.0

    iax.plot([0, 0], [0, 1], color=FRAME_COLOR, linewidth=LW)
    iax.plot([1, 1], [0, 1], color=FRAME_COLOR, linewidth=LW)
    iax.plot([0, 1], [1, 1], color=FRAME_COLOR, linewidth=LW)


# -----------------------------
# Diagram generator(s) — keep ONE copy only
# -----------------------------
def _tm_pr_01_diagrams(L_mm: float, P_kN: float, n: int = 401):
    """
    Simply supported beam with central point load P (kN).
    Returns:
      x (m), V (kN), M (kN·m)
    """
    L_mm = float(L_mm)
    P_kN = float(P_kN)

    x_m = np.linspace(0.0, L_mm / 1000.0, int(n))
    L_m = float(x_m[-1])
    mid = 0.5 * L_m

    R = 0.5 * P_kN
    V = np.where(x_m <= mid, R, -R)
    M = np.where(x_m <= mid, R * x_m, R * (L_m - x_m))
    return x_m, V, M


# -----------------------------
# Plotting helper (V/M like beam tool)
# -----------------------------
def _render_member_vm(
    x_m,
    V_kN,
    M_kNm,
    member_prefix: str = "beam_",
    key_prefix: str = "frame_vm_",
    x_label: str = "x (m)",
) -> None:
    """
    Plot V and M like beam tool. Labels follow strong/weak selection per member.
    Also stores PNG bytes in session_state for later PDF/report use.
    """
    weak = _axis_is_weak(member_prefix)
    V_lbl = "Vy (kN)" if weak else "Vz (kN)"
    M_lbl = "Mz (kN·m)" if weak else "My (kN·m)"

    colV, colM = st.columns(2)

    with colV:
        small_title(f"Shear force diagram {V_lbl}")
        fig1, ax1 = plt.subplots()
        ax1.plot(x_m, V_kN, linewidth=2)
        ax1.axhline(0, linewidth=1)
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(V_lbl)
        ax1.grid(True)

        buf_v = io.BytesIO()
        fig1.savefig(buf_v, format="png", dpi=200, bbox_inches="tight")
        buf_v.seek(0)
        st.session_state[f"{key_prefix}V_png"] = buf_v.getvalue()
        st.pyplot(fig1)

    with colM:
        small_title(f"Bending moment diagram {M_lbl}")
        fig2, ax2 = plt.subplots()
        ax2.plot(x_m, M_kNm, linewidth=2)
        ax2.axhline(0, linewidth=1)
        ax2.set_xlabel(x_label)
        ax2.set_ylabel(M_lbl)
        ax2.grid(True)

        buf_m = io.BytesIO()
        fig2.savefig(buf_m, format="png", dpi=200, bbox_inches="tight")
        buf_m.seek(0)
        st.session_state[f"{key_prefix}M_png"] = buf_m.getvalue()
        st.pyplot(fig2)

# -----------------------------
# Apply helper for "ready case" dicts  (APPLY-ONLY)
# -----------------------------
from typing import Optional

def make_frame_cases(prefix: str, n: int, beam_defaults: dict, col_defaults: dict):
    """Build placeholder case cards for the gallery."""
    out = []
    for i in range(1, int(n) + 1):
        key = f"{prefix}-{i:02d}"
        out.append({
            "key": key,
            "label": f"Case {i}",
            "img_path": f"assets/frame_cases/{key}.png",
            "beam": beam_defaults.copy(),
            "col":  col_defaults.copy(),
        })
    return out


def _apply_ready_frame_case(case: dict) -> None:
    """
    Fill BOTH beam_ and col_ inputs in session_state from a ready frame case dict:
      case["beam"] -> st.session_state["beam_<k>"] = v
      case["col"]  -> st.session_state["col_<k>"]  = v
    """
    if not isinstance(case, dict):
        return

    for pref in ("beam_", "col_"):
        data = case.get("beam" if pref == "beam_" else "col", {}) or {}
        for k, v in data.items():
            st.session_state[f"{pref}{k}"] = v


# -----------------------------
# Small wrappers / helpers used by frame cases
# -----------------------------
def _fill_beam_vm_to_components(member_prefix: str, V_kN: float, M_kNm: float) -> None:
    """
    Backward compatible helper:
    many of your apply blocks call this name.
    It does the same as _fill_VM_into_member_inputs().
    """
    _fill_VM_into_member_inputs(member_prefix, V_kN, M_kNm)

def _render_support_forces(
    case_key: str,
    RA_kN: float,
    RE_kN: float,
    HA_kN: Optional[float] = None,
    HE_kN: Optional[float] = None,
    HD_kN: Optional[float] = None,
) -> None:
    """
    Support forces box for frame cases.

    Conventions:
      - RA, RE: vertical reactions (kN), upward positive
      - H*: horizontal reactions (kN), positive to the RIGHT
        (use HA at A, HE at E, HD if your diagram uses D)
    """
    st.markdown("### Support forces")
    st.caption("Upward positive for R. Positive to the right for H.")

    st.session_state[f"{case_key}_RA_out"] = float(RA_kN)
    st.session_state[f"{case_key}_RE_out"] = float(RE_kN)

    # optional horizontal reactions in display order
    H_items = []
    # ALWAYS show HA (default 0.0 if not provided)
    HA_val = 0.0 if HA_kN is None else float(HA_kN)
    st.session_state[f"{case_key}_HA_out"] = HA_val
    H_items.append(("HA (kN)", f"{case_key}_HA_out"))

    if HE_kN is not None:
        st.session_state[f"{case_key}_HE_out"] = float(HE_kN)
        H_items.append(("HE (kN)", f"{case_key}_HE_out"))
    if HD_kN is not None:
        st.session_state[f"{case_key}_HD_out"] = float(HD_kN)
        H_items.append(("HD (kN)", f"{case_key}_HD_out"))

    base_items = [("RA (kN)", f"{case_key}_RA_out"), ("RE (kN)", f"{case_key}_RE_out")]
    all_items = base_items + H_items

    cols = st.columns(len(all_items))
    for col, (label, key) in zip(cols, all_items):
        with col:
            st.number_input(label, disabled=True, step=0.1, key=key)

def _member_I_m4(member_prefix: str) -> float:
    """
    Returns I (m^4) for the member based on its strong/weak axis selection.
    Expects section props: I_y_cm4, I_z_cm4.
    """
    sr = st.session_state.get(f"{member_prefix}sr_display", {}) or {}
    Iy = float(sr.get("I_y_cm4", 0.0)) * 1e-8
    Iz = float(sr.get("I_z_cm4", 0.0)) * 1e-8
    I = Iz if _axis_is_weak(member_prefix) else Iy
    return float(I) if I > 0 else 0.0


def _set_deflection_summary(delta_max_m: float, L_ref_m: float) -> None:
    """Push a minimal diag_summary so render_results() shows deflection."""
    if L_ref_m <= 0:
        return

    w_max_mm = abs(float(delta_max_m)) * 1000.0
    L_mm = float(L_ref_m) * 1000.0

    st.session_state["diag_summary"] = dict(
        defl_available=True,
        w_max_mm=w_max_mm,
        limit_L300=L_mm / 300.0,
        limit_L600=L_mm / 600.0,
        limit_L900=L_mm / 900.0,
    )
    st.session_state["diag_delta_max_mm"] = w_max_mm
# ============================================================
# Three-member frame helpers + TM-PP / TM-FF (equation-based)
# ============================================================

def _frame_beta_e(beam_prefix="beam_", col_prefix="col_") -> tuple[float, float]:
    """
    e = h/L , beta = I_beam / I_col (about the chosen bending axis for each member)
    Uses the app's strong/weak axis selection through _member_I_m4(prefix).
    """
    L = max(float(st.session_state.get(f"{beam_prefix}L_mm_in", 0.0)) / 1000.0, 1e-9)
    h = max(float(st.session_state.get(f"{col_prefix}L_mm_in", 0.0)) / 1000.0, 1e-9)
    I_b = max(_member_I_m4(beam_prefix), 0.0)
    I_c = max(_member_I_m4(col_prefix), 0.0)
    e = h / L
    beta = (I_b / I_c) if I_c > 0 else 0.0
    return beta, e


def _deflection_from_M_numeric(x_m: np.ndarray, M_kNm: np.ndarray, bc: str, member_prefix="beam_") -> float:
    """
    Compute deflection from M(x) by double integration of curvature.
    - x_m in meters
    - M_kNm in kN*m
    - bc: "ss" (v(0)=v(L)=0) or "ff" (v(0)=v(L)=0 and theta(0)=theta(L)=0)
    Returns max |v| in meters.
    """
    if x_m is None or M_kNm is None or len(x_m) < 5:
        return 0.0
    L = float(x_m[-1])
    if L <= 0:
        return 0.0

    E = get_E_Pa()
    I = _member_I_m4(member_prefix)
    if E <= 0 or I <= 0:
        return 0.0

    # curvature kappa = M / (E I). Convert kN*m -> N*m: *1000
    kappa = (M_kNm * 1000.0) / (E * I)

    # integrate twice (trapezoidal)
    theta = np.zeros_like(x_m)
    v = np.zeros_like(x_m)
    for i in range(1, len(x_m)):
        dx = x_m[i] - x_m[i - 1]
        theta[i] = theta[i - 1] + 0.5 * (kappa[i] + kappa[i - 1]) * dx
        v[i] = v[i - 1] + 0.5 * (theta[i] + theta[i - 1]) * dx

    # apply boundary conditions by adding a linear (and optionally cubic) correction
    # v_corr = v + a*x + b
    # for fixed-fixed we also correct slope with c*x*(L-x)
    x = x_m
    if bc == "ss":
        # enforce v(0)=0 already; enforce v(L)=0 by subtracting (v(L)/L)*x
        v = v - (v[-1] / L) * x
        return float(np.max(np.abs(v)))

    # fixed-fixed: enforce v(L)=0 and theta(0)=theta(L)=0 approximately
    # We use a quadratic correction term c*x*(L-x) to adjust slope.
    # theta_corr = theta + a + c*(L-2x)
    # v_corr = v + a*x + c*x*(L-x)  (since integral of (L-2x) is x(L-x))
    th0 = theta[0]
    thL = theta[-1]
    vL = v[-1]

    # solve for a and c from:
    # theta_corr(0)=0 => th0 + a + c*L = 0
    # theta_corr(L)=0 => thL + a - c*L = 0
    # => subtract: (th0 - thL) + 2cL = 0 => c = (thL - th0)/(2L)
    c = (thL - th0) / (2.0 * L)
    a = -th0 - c * L

    v = v + a * x + c * x * (L - x)

    # now enforce v(L)=0 with an additional linear tweak
    v = v - (v[-1] / L) * x
    return float(np.max(np.abs(v)))
# -----------------------------
# TM-FR (Fixed / Free) — equation-based preview + deflection
# Conventions used here:
#   L_mm, h_mm are geometry
#   F_kN is point load (kN) or moment (kN*m) depending on the case label
#   w_kNm is UDL (kN/m)
# Deflection: use closed-form from your images; EI uses COLUMN I (col_) by default.
# -----------------------------

def _EI_col() -> float:
    E = get_E_Pa()
    I = _member_I_m4("col_")
    return float(E * I) if E > 0 and I > 0 else 0.0


def _render_tm_fr_01_whole_frame_diagrams(L_mm: float, h_mm: float, F_kN: float):
    """TM-FR-01: Free End Horizontal Point Load (P at D to +x)."""
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    P = float(F_kN)
    if L <= 0 or h <= 0:
        return

    # Support reactions from your image
    RA = 0.0
    HA = -P  # sign per your convention (+ right)

    # --- Diagrams first (KEEP LAYOUT CONSISTENT) ---
    # Beam: constant moment ~ P*h (envelope)
    x = np.linspace(0.0, L, 401)
    Vb = np.zeros_like(x)
    Mb = np.full_like(x, P * h)
    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(
            x_m=x, V_kN=Vb, M_kNm=Mb,
            member_prefix="beam_", key_prefix="tmfr01_beam_", x_label="x (m)"
        )

    # Column: triangular moment, constant shear
    y = np.linspace(0.0, h, 401)
    Vc = np.full_like(y, P)
    Mc = P * (h - y)
    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(
            x_m=y, V_kN=Vc, M_kNm=Mc,
            member_prefix="col_", key_prefix="tmfr01_col_", x_label="y (m)"
        )

    # --- Support forces last ---
    _render_support_forces("tmfr01", RA_kN=RA, RE_kN=0.0, HA_kN=HA)

    # Deflection (your existing approach)
    EI = _EI_col()
    if EI > 0:
        Dx = (P * h**2 / (3.0 * EI)) * (3.0 * L + 4.0 * h)
        Dy = -(P * h * L / (2.0 * EI)) * (L + h)
        _set_deflection_summary(abs(Dy), L_ref_m=L)
def _render_tm_fr_02_whole_frame_diagrams(L_mm: float, h_mm: float, F_kN: float):
    """TM-FR-02: Free End Vertical Point Load (P at D downward)."""
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    P = float(F_kN)
    if L <= 0 or h <= 0:
        return

    # Support reactions from your image
    RA = P
    HA = 0.0

    # --- Diagrams first ---
    # Beam: linear moment from PL at A/B down to 0 (simple envelope)
    x = np.linspace(0.0, L, 401)
    Vb = np.full_like(x, P)
    Mb = P * (L - x)
    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(
            x_m=x, V_kN=Vb, M_kNm=Mb,
            member_prefix="beam_", key_prefix="tmfr02_beam_", x_label="x (m)"
        )

    # Column: approx constant moment ~ P*L (envelope)
    y = np.linspace(0.0, h, 251)
    Vc = np.zeros_like(y)
    Mc = np.full_like(y, P * L)
    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(
            x_m=y, V_kN=Vc, M_kNm=Mc,
            member_prefix="col_", key_prefix="tmfr02_col_", x_label="y (m)"
        )

    # --- Support forces last ---
    _render_support_forces("tmfr02", RA_kN=RA, RE_kN=0.0, HA_kN=HA)

    # Deflection (your existing block)
    EI = _EI_col()
    if EI > 0:
        Dx = -(P * h * L / (2.0 * EI)) * (L + 2.0 * h)
        Dy = (P * L**2 / (3.0 * EI)) * (L + 3.0 * h)
        _set_deflection_summary(abs(Dy), L_ref_m=L)
def _render_tm_fr_03_whole_frame_diagrams(L_mm: float, h_mm: float, F_kN: float):
    """TM-FR-03: Side Top Point Load (P to +x at top)."""
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    P = float(F_kN)
    if L <= 0 or h <= 0:
        return

    # Support reactions from image
    RA = 0.0
    HA = -P

    # --- Diagrams first ---
    x = np.linspace(0.0, L, 401)
    Vb = np.zeros_like(x)
    Mb = np.full_like(x, P * h)
    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(
            x_m=x, V_kN=Vb, M_kNm=Mb,
            member_prefix="beam_", key_prefix="tmfr03_beam_", x_label="x (m)"
        )

    y = np.linspace(0.0, h, 401)
    Vc = np.full_like(y, P)
    Mc = P * (h - y)
    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(
            x_m=y, V_kN=Vc, M_kNm=Mc,
            member_prefix="col_", key_prefix="tmfr03_col_", x_label="y (m)"
        )

    # --- Support forces last ---
    _render_support_forces("tmfr03", RA_kN=RA, RE_kN=0.0, HA_kN=HA)

    EI = _EI_col()
    if EI > 0:
        Dy_D = (P * L * h**2 / (2.0 * EI))
        _set_deflection_summary(abs(Dy_D), L_ref_m=L)
def _render_tm_fr_04_whole_frame_diagrams(L_mm: float, h_mm: float, M_kNm: float):
    """TM-FR-04: Free End Bending Moment at D (MD)."""
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    MD = float(M_kNm)
    if L <= 0 or h <= 0:
        return

    # --- Diagrams first ---
    x = np.linspace(0.0, L, 401)
    Vb = np.zeros_like(x)
    Mb = np.full_like(x, MD)
    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(
            x_m=x, V_kN=Vb, M_kNm=Mb,
            member_prefix="beam_", key_prefix="tmfr04_beam_", x_label="x (m)"
        )

    y = np.linspace(0.0, h, 251)
    Vc = np.zeros_like(y)
    Mc = np.full_like(y, MD)
    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(
            x_m=y, V_kN=Vc, M_kNm=Mc,
            member_prefix="col_", key_prefix="tmfr04_col_", x_label="y (m)"
        )

    # --- Support forces last ---
    _render_support_forces("tmfr04", RA_kN=0.0, RE_kN=0.0, HA_kN=0.0)

    EI = _EI_col()
    if EI > 0:
        Dy = -(MD * L / (2.0 * EI)) * (L + 2.0 * h)
        _set_deflection_summary(abs(Dy), L_ref_m=L)
def _render_tm_fr_05_whole_frame_diagrams(L_mm: float, h_mm: float, w_kNm: float):
    """TM_attach: TM-FR-05: Top UDL on beam (w)."""
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    w = float(w_kNm)
    if L <= 0 or h <= 0:
        return

    # Support reactions from image
    RA = w * L
    HA = 0.0

    # --- Diagrams first ---
    x = np.linspace(0.0, L, 401)
    Vb = RA - w * x
    Mb = 0.5 * w * (L - x) ** 2  # envelope-style
    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(
            x_m=x, V_kN=Vb, M_kNm=Mb,
            member_prefix="beam_", key_prefix="tmfr05_beam_", x_label="x (m)"
        )

    y = np.linspace(0.0, h, 251)
    Vc = np.zeros_like(y)
    Mc = np.full_like(y, 0.5 * w * L**2)
    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(
            x_m=y, V_kN=Vc, M_kNm=Mc,
            member_prefix="col_", key_prefix="tmfr05_col_", x_label="y (m)"
        )

    # --- Support forces last ---
    _render_support_forces("tmfr05", RA_kN=RA, RE_kN=0.0, HA_kN=HA)

    EI = _EI_col()
    if EI > 0:
        Dy = (w * L**3 / (8.0 * EI)) * (L + 4.0 * h)
        _set_deflection_summary(abs(Dy), L_ref_m=L)

def _render_dm_pp_01_whole_frame_diagrams(L_mm: float, h_mm: float, F_kN: float):
    """DM-PP-01: Two Member Frame (Pin/Pin) — Top Point Load at C (assumed midspan)."""
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    P = float(F_kN)
    if L <= 0 or h <= 0:
        return

    beta, e = _frame_beta_e("beam_", "col_")
    x0 = 0.5 * L  # C at midspan

    # reactions from your STRUCT image (Pin/Pin)
    RA = (P * x0 * (L**2 * (2.0 * beta * e + 3.0) - x0**2)) / (2.0 * L**3 * (beta * e + 1.0))
    RD = P - RA
    HA = (P * x0 * (L**2 - x0**2)) / (2.0 * h * L**2 * (beta * e + 1.0))  # HA = HD

    # -------------------------
    # Beam diagrams (first)
    # -------------------------
    x = np.linspace(0.0, L, 801)
    step = (x >= x0).astype(float)
    Vb = RA - P * step
    Mb = RA * x - P * (x - x0) * step
    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(
            x_m=x, V_kN=Vb, M_kNm=Mb,
            member_prefix="beam_", key_prefix="dmpp01_beam_", x_label="x (m)"
        )

    # -------------------------
    # Column diagrams (second)
    # Use the horizontal reaction to generate a consistent column envelope
    # -------------------------
    y = np.linspace(0.0, h, 401)
    Vc = np.full_like(y, HA)
    Mc = HA * (h - y)
    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(
            x_m=y, V_kN=Vc, M_kNm=Mc,
            member_prefix="col_", key_prefix="dmpp01_col_", x_label="y (m)"
        )

    # -------------------------
    # Support forces (last)
    # -------------------------
    _render_support_forces("dmpp01", RA_kN=RA, RE_kN=RD, HA_kN=HA, HD_kN=HA)

    # Deflection summary (simple beam ref, optional)
    E = get_E_Pa()
    I = _member_I_m4("beam_")
    if E > 0 and I > 0:
        delta = abs(P) * L**3 / (48.0 * E * I)
        _set_deflection_summary(delta, L_ref_m=L)
        
def _render_dm_pp_02_whole_frame_diagrams(L_mm: float, h_mm: float, w_kNm: float):
    """DM-PP-02: Two Member Frame (Pin/Pin) — Top UDL."""
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    w = float(w_kNm)
    if L <= 0 or h <= 0:
        return

    beta, e = _frame_beta_e("beam_", "col_")

    # reactions from your STRUCT image (Pin/Pin)
    RA = (w * L / 8.0) * ((4.0 * beta * e + 5.0) / (beta * e + 1.0))
    RC = (w * L / 8.0) * ((4.0 * beta * e + 3.0) / (beta * e + 1.0))
    HA = (w * L**2) / (8.0 * h * (beta * e + 1.0))  # HA = HC

    # -------------------------
    # Beam diagrams (first)
    # -------------------------
    x = np.linspace(0.0, L, 801)
    Vb = RA - w * x
    Mb = RA * x - 0.5 * w * x**2
    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(
            x_m=x, V_kN=Vb, M_kNm=Mb,
            member_prefix="beam_", key_prefix="dmpp02_beam_", x_label="x (m)"
        )

    # -------------------------
    # Column diagrams (second)
    # -------------------------
    y = np.linspace(0.0, h, 401)
    Vc = np.full_like(y, HA)
    Mc = HA * (h - y)
    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(
            x_m=y, V_kN=Vc, M_kNm=Mc,
            member_prefix="col_", key_prefix="dmpp02_col_", x_label="y (m)"
        )

    # -------------------------
    # Support forces (last)
    # -------------------------
    _render_support_forces("dmpp02", RA_kN=RA, RE_kN=RC, HA_kN=HA, HE_kN=HA)

    # Deflection (simple SS beam ref)
    E = get_E_Pa()
    I = _member_I_m4("beam_")
    if E > 0 and I > 0:
        delta = 5.0 * abs(w) * L**4 / (384.0 * E * I)
        _set_deflection_summary(delta, L_ref_m=L)
        
def _render_dm_ff_01_whole_frame_diagrams(L_mm: float, h_mm: float, F_kN: float):
    """DM-FF-01: Two Member Frame (Fixed/Fixed) — Top Point Load at C (assumed midspan)."""
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    P = float(F_kN)
    if L <= 0 or h <= 0:
        return

    beta, e = _frame_beta_e("beam_", "col_")
    x0 = 0.5 * L

    # reactions from your STRUCT image (Fixed/Fixed)
    RA = (P * x0**2 / (2.0 * L**3 * (beta * e + 1.0))) * (
        beta * e * (3.0 * L - x0) + 2.0 * (3.0 * L - 2.0 * x0)
    )
    RD = P - RA
    HA = (3.0 * P * x0**2 * (L - x0)) / (2.0 * h * L**2 * (beta * e + 1.0))  # HA = HD

    # -------------------------
    # Beam diagrams (first)
    # -------------------------
    x = np.linspace(0.0, L, 801)
    step = (x >= x0).astype(float)
    Vb = RA - P * step
    Mb = RA * x - P * (x - x0) * step
    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(
            x_m=x, V_kN=Vb, M_kNm=Mb,
            member_prefix="beam_", key_prefix="dmff01_beam_", x_label="x (m)"
        )

    # -------------------------
    # Column diagrams (second)
    # -------------------------
    y = np.linspace(0.0, h, 401)
    Vc = np.full_like(y, HA)
    Mc = HA * (h - y)
    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(
            x_m=y, V_kN=Vc, M_kNm=Mc,
            member_prefix="col_", key_prefix="dmff01_col_", x_label="y (m)"
        )

    # -------------------------
    # Support forces (last)
    # -------------------------
    _render_support_forces("dmff01", RA_kN=RA, RE_kN=RD, HA_kN=HA, HD_kN=HA)

    # Deflection (simple fixed-fixed beam ref)
    E = get_E_Pa()
    I = _member_I_m4("beam_")
    if E > 0 and I > 0:
        delta = abs(P) * L**3 / (192.0 * E * I)
        _set_deflection_summary(delta, L_ref_m=L)
        
def _render_dm_ff_02_whole_frame_diagrams(L_mm: float, h_mm: float, w_kNm: float):
    """DM-FF-02: Two Member Frame (Fixed/Fixed) — Top UDL."""
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    w = float(w_kNm)
    if L <= 0 or h <= 0:
        return

    beta, e = _frame_beta_e("beam_", "col_")

    # If you already have exact STRUCT formulas in your notes, put them here.
    # For now: use consistent reactions to build diagrams (no layout differences).
    RA = w * L / 2.0
    RD = w * L / 2.0
    HA = (w * L**2) / (8.0 * h * (beta * e + 1.0))

    # -------------------------
    # Beam diagrams (first)
    # -------------------------
    x = np.linspace(0.0, L, 801)
    Vb = RA - w * x
    Mb = RA * x - 0.5 * w * x**2
    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(
            x_m=x, V_kN=Vb, M_kNm=Mb,
            member_prefix="beam_", key_prefix="dmff02_beam_", x_label="x (m)"
        )

    # -------------------------
    # Column diagrams (second)
    # -------------------------
    y = np.linspace(0.0, h, 401)
    Vc = np.full_like(y, HA)
    Mc = HA * (h - y)
    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(
            x_m=y, V_kN=Vc, M_kNm=Mc,
            member_prefix="col_", key_prefix="dmff02_col_", x_label="y (m)"
        )

    # -------------------------
    # Support forces (last)
    # -------------------------
    _render_support_forces("dmff02", RA_kN=RA, RE_kN=RD, HA_kN=HA, HD_kN=HA)

    # Deflection (simple fixed-fixed UDL beam ref)
    E = get_E_Pa()
    I = _member_I_m4("beam_")
    if E > 0 and I > 0:
        delta = abs(w) * L**4 / (384.0 * E * I)
        _set_deflection_summary(delta, L_ref_m=L)


# -----------------------------
# TM-PP-01: Top point load at C (midspan)
# -----------------------------
def _render_tm_pp_01_whole_frame_diagrams(L_mm: float, h_mm: float, F_kN: float):
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0

    # IMPORTANT: user may input negative for downward (default -50)
    # STRUCT formulas assume P is a positive downward magnitude
    P = abs(float(F_kN))

    if L <= 0 or h <= 0:
        return

    beta, e = _frame_beta_e("beam_", "col_")

    x_load = 0.5 * L  # C at midspan
    RA = P * (L - x_load) / L
    RE = P * x_load / L

    denom = (2.0 * beta * e + 3.0)
    HA = (3.0 * P * x_load * (L - x_load)) / (2.0 * h * L * denom) if denom != 0 else 0.0
    HE = HA

    MB = (3.0 * P * x_load / (2.0 * L)) * ((L - x_load) / denom) if denom != 0 else 0.0
    MD = MB
    MC = (P * x_load * (L - x_load) / (2.0 * L)) * ((4.0 * beta * e + 3.0) / denom) if denom != 0 else 0.0

    # -------------------
    # Beam V/M:
    # - M_end: linear between end moments (hogging, shown positive in STRUCT sketch)
    # - M_simple: sagging simply-supported moment from the point load (positive sagging)
    # - Total plotted moment (STRUCT style): M = M_end - M_simple
    #   => starts positive at B, dips in middle, returns positive at D
    # -------------------
    x = np.linspace(0.0, L, 401)

    # shear (using downward load magnitude)
    V = np.where(x <= x_load, RA, RA - P)

    # simply-supported sagging moment (positive sagging)
    M_simple = np.where(
        x <= x_load,
        RA * x,
        RA * x - P * (x - x_load),
    )

    # end-moment line (positive at ends like reference)
    M_end = MB + (MD - MB) * (x / L)

    # STRUCT-style beam moment shape
    M_plot = M_end - M_simple

    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(
            x_m=x, V_kN=V, M_kNm=M_plot,
            member_prefix="beam_", key_prefix="tmpp01_beam_", x_label="x (m)"
        )

    # -------------------
    # Column diagrams (AB):
    # show linear moment from horizontal reaction (same sign convention as beam ends)
    # -------------------
    y = np.linspace(0.0, h, 251)
    Mcol_plot = HA * y
    Vcol = np.full_like(y, HA)

    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(
            x_m=y, V_kN=Vcol, M_kNm=Mcol_plot,
            member_prefix="col_", key_prefix="tmpp01_col_", x_label="y (m)"
        )

    _render_support_forces("tmpp01", RA_kN=RA, RE_kN=RE, HA_kN=HA, HE_kN=HE)

    # Deflection: use physical beam moment magnitude (sagging) combined with end-moment line
    # same expression as plotted (this is the internal moment diagram we intend)
    delta = _deflection_from_M_numeric(x, M_plot, bc="ss", member_prefix="beam_")
    _set_deflection_summary(delta, L_ref_m=L)

# -----------------------------
# TM-PP-02: Top horizontal point load at B (left top corner)
# -----------------------------
def _render_tm_pp_02_whole_frame_diagrams(L_mm: float, h_mm: float, F_kN: float):
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    P = float(F_kN)
    if L <= 0 or h <= 0:
        return

    # From reference sheet (magnitudes)
    RA_mag = (P * h) / L
    RE_mag = RA_mag
    HA_mag = 0.5 * P
    HE_mag = 0.5 * P
    MB_mag = 0.5 * P * h  # magnitude of end moment

    # ---- Apply your desired sign convention ----
    # You want: moment at B (beam start + column top) NEGATIVE and then increases along beam.
    RA = RA_mag
    RE = RE_mag
    HA = HA_mag
    HE = HE_mag
    MB = -MB_mag  # NEGATIVE at B

    # -------------------
    # Beam diagrams:
    # V constant (non-zero) and M increases from negative end moment.
    # M(x) = MB + RA*x
    # -------------------
    x = np.linspace(0.0, L, 401)
    V = np.full_like(x, RA)
    M = MB + RA * x

    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(
            x_m=x, V_kN=V, M_kNm=M,
            member_prefix="beam_", key_prefix="tmpp02_beam_", x_label="x (m)"
        )

    # -------------------
    # Column diagrams (left column AB):
    # Moment at top (y=h) should be same as beam at B: negative.
    # Linear from 0 at base to MB at top.
    # -------------------
    y = np.linspace(0.0, h, 251)
    Mcol = MB * (y / h)     # 0 at y=0, MB(negative) at y=h
    Vcol = np.full_like(y, HA)

    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(
            x_m=y, V_kN=Vcol, M_kNm=Mcol,
            member_prefix="col_", key_prefix="tmpp02_col_", x_label="y (m)"
        )

    _render_support_forces("tmpp02", RA_kN=RA, RE_kN=RE, HA_kN=HA, HE_kN=HE)

    delta = _deflection_from_M_numeric(x, M, bc="ss", member_prefix="beam_")
    _set_deflection_summary(delta, L_ref_m=L)

# -----------------------------
# TM-PP-03: Side point load on left column at distance y from top
# Hogging at top joints is NEGATIVE (as you requested)
# -----------------------------
def _render_tm_pp_03_whole_frame_diagrams(L_mm: float, h_mm: float, y_m: float, F_kN: float):
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    P = float(F_kN)
    y = float(y_m)

    if L <= 0 or h <= 0:
        return

    # clamp y to [0, h]
    if y < 0.0:
        y = 0.0
    elif y > h:
        y = h

    beta, e = _frame_beta_e("beam_", "col_")

    # k = β y (2h - y) / [ h (2βh + 3L) ]
    denom_h = h * (2.0 * beta * h + 3.0 * L)
    k = (beta * y * (2.0 * h - y) / denom_h) if denom_h != 0 else 0.0

    # Support reactions (STRUCT reference)
    RA = P * (h - y) / L
    RE = RA

    HA = (P / (2.0 * h)) * (h + y - (h - y) * k) if h != 0 else 0.0
    HE = (P * (h - y) / (2.0 * h)) * (1.0 + k) if h != 0 else 0.0

    # End moments (reference magnitudes)
    MB_mag = (P * (h - y) / (2.0 * h)) * (h + y - (h - y) * k) if h != 0 else 0.0
    MC_mag = (P * (h - y) / 2.0) * (1.0 - k)
    MD_mag = (P * (h - y) / 2.0) * (1.0 + k)

    # --- SIGN per STRUCT reference (TM-PP-03):
    # MC is NEGATIVE, MD is POSITIVE
    MB = MB_mag
    MC = -abs(MC_mag)
    MD = +abs(MD_mag)

    # -------------------
    # Beam diagrams: no vertical load on beam -> V constant, M linear
    # M(0)=MC (at left top joint C), M(L)=MD (at right top joint D)
    # -------------------
    x = np.linspace(0.0, L, 401)
    V_beam = np.full_like(x, (MD - MC) / L)
    M_beam = MC + ((MD - MC) / L) * x

    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(
            x_m=x, V_kN=V_beam, M_kNm=M_beam,
            member_prefix="beam_", key_prefix="tmpp03_beam_", x_label="x (m)"
        )

    # -------------------
    # Left column AB moment: piecewise linear with kink at load point
    # Use:
    #   A: M=0
    #   C(top): M=MC (negative)
    # At load point B (from base): y_load = h - y
    # We keep kink magnitude MB but enforce continuity up to MC.
    # -------------------
    y_base = np.linspace(0.0, h, 251)
    y_load_from_base = h - y

    # Choose sign for MB to follow the same "hogging negative" family near the top:
    # Put MB negative as well (this matches your request: top region negative).
    MB = -abs(MB_mag)

    M_left = np.zeros_like(y_base)
    if y_load_from_base <= 0.0 or y_load_from_base >= h:
        # load at top or base -> straight A->C
        M_left = (MC / h) * y_base if h != 0 else np.zeros_like(y_base)
    else:
        m1 = MB / y_load_from_base
        m2 = (MC - MB) / (h - y_load_from_base)
        M_left = np.where(
            y_base <= y_load_from_base,
            m1 * y_base,
            MB + m2 * (y_base - y_load_from_base),
        )

    V_left = np.full_like(y_base, HA)

    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(
            x_m=y_base, V_kN=V_left, M_kNm=M_left,
            member_prefix="col_", key_prefix="tmpp03_col_", x_label="y (m)"
        )

    _render_support_forces("tmpp03", RA_kN=RA, RE_kN=RE, HA_kN=HA, HE_kN=HE)

    delta = _deflection_from_M_numeric(x, M_beam, bc="ss", member_prefix="beam_")
    _set_deflection_summary(delta, L_ref_m=L)

# -----------------------------
# TM-PP-04: Top UDL on beam
# -----------------------------
def _render_tm_pp_04_whole_frame_diagrams(L_mm: float, h_mm: float, w_kNm: float):
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    w = float(w_kNm)  # kN/m (downward negative is allowed)
    if L <= 0 or h <= 0:
        return

    beta, e = _frame_beta_e("beam_", "col_")
    denom = (2.0 * beta * e + 3.0)

    RA = 0.5 * w * L
    RE = RA
    HA = (w * L) / (4.0 * e * denom) if (e != 0 and denom != 0) else 0.0
    HE = HA

    MB = (w * L**2) / (4.0 * denom) if denom != 0 else 0.0
    MD = MB

    x = np.linspace(0.0, L, 401)
    V = RA - w * x
    M = MB + RA * x - 0.5 * w * x**2

    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(x_m=x, V_kN=V, M_kNm=M, member_prefix="beam_", key_prefix="tmpp04_beam_", x_label="x (m)")

    y = np.linspace(0.0, h, 251)
    Vcol = np.full_like(y, HA)
    Mcol = HA * y
    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(x_m=y, V_kN=Vcol, M_kNm=Mcol, member_prefix="col_", key_prefix="tmpp04_col_", x_label="y (m)")

    _render_support_forces("tmpp04", RA_kN=RA, RE_kN=RE, HA_kN=HA, HE_kN=HE)

    delta = _deflection_from_M_numeric(x, M, bc="ss", member_prefix="beam_")
    _set_deflection_summary(delta, L_ref_m=L)


# -----------------------------
# TM-PP-05: Side UDL on left column (use sheet moments/reactions)
# -----------------------------
def _render_tm_pp_05_whole_frame_diagrams(L_mm: float, h_mm: float, w_kNm: float):
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    w = float(w_kNm)  # kN/m (outward +)
    if L <= 0 or h <= 0:
        return

    beta, e = _frame_beta_e("beam_", "col_")
    denom = (2.0 * beta * e + 3.0)

    RA = (w * h**2) / (2.0 * L)
    RE = RA

    HA = (w * h / 8.0) * ((11.0 * beta * e + 18.0) / denom) if denom != 0 else 0.0
    HE = (w * h / 8.0) * ((5.0 * beta * e + 6.0) / denom) if denom != 0 else 0.0

    MB = (3.0 * w * h**2 / 8.0) * ((beta * e + 2.0) / denom) if denom != 0 else 0.0
    MC = (w * h**2 / 8.0) * ((5.0 * beta * e + 6.0) / denom) if denom != 0 else 0.0

    x = np.linspace(0.0, L, 401)
    V = np.full_like(x, RA)
    M = MB + RA * x  # simple linear visualization

    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(x_m=x, V_kN=V, M_kNm=M, member_prefix="beam_", key_prefix="tmpp05_beam_", x_label="x (m)")

    y = np.linspace(0.0, h, 251)
    Vcol = np.full_like(y, HA)
    Mcol = HA * y
    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(x_m=y, V_kN=Vcol, M_kNm=Mcol, member_prefix="col_", key_prefix="tmpp05_col_", x_label="y (m)")

    _render_support_forces("tmpp05", RA_kN=RA, RE_kN=RE, HA_kN=HA, HE_kN=HE)

    delta = _deflection_from_M_numeric(x, M, bc="ss", member_prefix="beam_")
    _set_deflection_summary(delta, L_ref_m=L)


# -----------------------------
# TM-FF-01: Central point load
# -----------------------------
def _render_tm_ff_01_whole_frame_diagrams(L_mm: float, h_mm: float, F_kN: float):
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    P = float(F_kN)
    if L <= 0 or h <= 0:
        return

    beta, e = _frame_beta_e("beam_", "col_")
    denom = (beta * e + 2.0)

    RA = 0.5 * P
    RE = RA
    HA = (3.0 * P * L) / (8.0 * h * denom) if (h != 0 and denom != 0) else 0.0
    HE = HA

    MB = (P * L) / (4.0 * denom) if denom != 0 else 0.0
    MD = MB
    x_load = 0.5 * L

    x = np.linspace(0.0, L, 401)
    V = np.where(x <= x_load, RA, RA - P)
    M = np.where(x <= x_load, MB + RA * x, MB + RA * x - P * (x - x_load))

    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(x_m=x, V_kN=V, M_kNm=M, member_prefix="beam_", key_prefix="tmff01_beam_", x_label="x (m)")

    y = np.linspace(0.0, h, 251)
    # fixed base: include base moment MA (from sheet)
    MA = (P * L) / (8.0 * denom) if denom != 0 else 0.0
    Mcol = MA + (MB - MA) * (y / h)
    Vcol = np.full_like(y, (MB - MA) / h if h != 0 else 0.0)

    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(x_m=y, V_kN=Vcol, M_kNm=Mcol, member_prefix="col_", key_prefix="tmff01_col_", x_label="y (m)")

    _render_support_forces("tmff01", RA_kN=RA, RE_kN=RE, HA_kN=HA, HE_kN=HE)

    delta = _deflection_from_M_numeric(x, M, bc="ff", member_prefix="beam_")
    _set_deflection_summary(delta, L_ref_m=L)


# -----------------------------
# TM-FF-02: Top UDL
# -----------------------------
def _render_tm_ff_02_whole_frame_diagrams(L_mm: float, h_mm: float, w_kNm: float):
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    w = float(w_kNm)
    if L <= 0 or h <= 0:
        return

    beta, e = _frame_beta_e("beam_", "col_")
    denom = (beta * e + 2.0)

    RA = 0.5 * w * L
    RE = RA
    HA = (w * L**2) / (4.0 * h * denom) if (h != 0 and denom != 0) else 0.0
    HE = HA

    MB = (w * L**2) / (6.0 * denom) if denom != 0 else 0.0
    MD = MB

    x = np.linspace(0.0, L, 401)
    V = RA - w * x
    M = MB + RA * x - 0.5 * w * x**2

    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(x_m=x, V_kN=V, M_kNm=M, member_prefix="beam_", key_prefix="tmff02_beam_", x_label="x (m)")

    y = np.linspace(0.0, h, 251)
    MA = (w * L**2) / (12.0 * denom) if denom != 0 else 0.0
    Mcol = MA + (MB - MA) * (y / h)
    Vcol = np.full_like(y, (MB - MA) / h if h != 0 else 0.0)

    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(x_m=y, V_kN=Vcol, M_kNm=Mcol, member_prefix="col_", key_prefix="tmff02_col_", x_label="y (m)")

    _render_support_forces("tmff02", RA_kN=RA, RE_kN=RE, HA_kN=HA, HE_kN=HE)

    delta = _deflection_from_M_numeric(x, M, bc="ff", member_prefix="beam_")
    _set_deflection_summary(delta, L_ref_m=L)


# -----------------------------
# TM-FF-03: Side UDL (sheet is messy; keep a stable implementation)
# -----------------------------
def _render_tm_ff_03_whole_frame_diagrams(L_mm: float, h_mm: float, w_kNm: float):
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    w = float(w_kNm)
    if L <= 0 or h <= 0:
        return

    beta, e = _frame_beta_e("beam_", "col_")

    # Keep a conservative portal approximation:
    # base fixed -> bigger column moments; beam moments taken from joint shear * h.
    H = w * h  # resultant horizontal
    HA = 0.5 * H
    HE = 0.5 * H

    RA = 0.0
    RE = 0.0

    MB = HA * h
    MD = HE * h

    x = np.linspace(0.0, L, 401)
    V = np.zeros_like(x)
    M = MB + (MD - MB) * (x / L)

    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(x_m=x, V_kN=V, M_kNm=M, member_prefix="beam_", key_prefix="tmff03_beam_", x_label="x (m)")

    y = np.linspace(0.0, h, 251)
    # fixed base: moment at base not zero (approx MB/2)
    MA = 0.5 * MB
    Mcol = MA + (MB - MA) * (y / h)
    Vcol = np.full_like(y, (MB - MA) / h if h != 0 else 0.0)

    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(x_m=y, V_kN=Vcol, M_kNm=Mcol, member_prefix="col_", key_prefix="tmff03_col_", x_label="y (m)")

    _render_support_forces("tmff03", RA_kN=RA, RE_kN=RE, HA_kN=HA, HE_kN=HE)

    delta = _deflection_from_M_numeric(x, M, bc="ff", member_prefix="beam_")
    _set_deflection_summary(delta, L_ref_m=L)


# ============================================================
# TM-PR-01..08 — diagram preview functions (MUST exist globally)
# ============================================================
def _render_tm_pr_01_whole_frame_diagrams(L_mm: float, h_mm: float, P_kN: float):
    """TM-PR-01: central point load on beam."""
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    P = float(P_kN)
    if L <= 0:
        return

    x = np.linspace(0.0, L, 401)
    mid = 0.5 * L
    R = 0.5 * P
    V = np.where(x <= mid, R, -R)
    M = np.where(x <= mid, R * x, R * (L - x))

    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(x_m=x, V_kN=V, M_kNm=M, member_prefix="beam_", key_prefix="tmpr01_beam_", x_label="x (m)")

    if h > 0:
        y = np.linspace(0.0, h, 200)
        with st.expander("Column diagrams", expanded=False):
            small_title("Column diagrams")
            _render_member_vm(x_m=y, V_kN=np.zeros_like(y), M_kNm=np.zeros_like(y),
                             member_prefix="col_", key_prefix="tmpr01_col_", x_label="y (m)")

    _render_support_forces("tmpr01", RA_kN=0.5 * P, RE_kN=0.5 * P)

    # Beam deflection (mid-point load, SS beam): δmax = P L^3 /(48 E I)
    E = get_E_Pa()
    I = _member_I_m4("beam_")
    if I > 0:
        delta = (abs(P) * 1000.0 * L**3) / (48.0 * E * I)
        _set_deflection_summary(delta, L_ref_m=L)


def _render_tm_pr_02_whole_frame_diagrams(L_mm: float, h_mm: float, F_kN: float):
    """TM-PR-02: horizontal point load at top-right."""
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    F = float(F_kN)
    if L <= 0 or h <= 0:
        return

    RA = (F * h) / L
    RE = (F * h) / L

    x_b = np.linspace(0.0, L, 401)
    V_b = np.full_like(x_b, RA)
    M_b = (F * h) * (1.0 - x_b / L)

    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(x_m=x_b, V_kN=V_b, M_kNm=M_b, member_prefix="beam_", key_prefix="tmpr02_beam_", x_label="x (m)")

    y = np.linspace(0.0, h, 251)
    V_c = np.full_like(y, F)
    M_c = F * y

    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(x_m=y, V_kN=V_c, M_kNm=M_c, member_prefix="col_", key_prefix="tmpr02_col_", x_label="y (m)")

    _render_support_forces("tmpr02", RA_kN=RA, RE_kN=RE, HA_kN=F)


def _render_tm_pr_03_whole_frame_diagrams(L_mm: float, h_mm: float, F_kN: float):
    """TM-PR-03: side bottom point load (horizontal)."""
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    F = float(F_kN)
    if L <= 0 or h <= 0:
        return

    x = np.linspace(0.0, L, 401)
    V_b = np.zeros_like(x)
    M_b = np.full_like(x, F * h)  # envelope per your reference

    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(x_m=x, V_kN=V_b, M_kNm=M_b, member_prefix="beam_", key_prefix="tmpr03_beam_", x_label="x (m)")

    y = np.linspace(0.0, h, 251)
    V_c = np.full_like(y, F)
    M_c = F * y

    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(x_m=y, V_kN=V_c, M_kNm=M_c, member_prefix="col_", key_prefix="tmpr03_col_", x_label="y (m)")

    _render_support_forces("tmpr03", RA_kN=0.0, RE_kN=0.0, HA_kN=F)


def _render_tm_pr_04_whole_frame_diagrams(L_mm: float, h_mm: float, Mc_kNm: float):
    """TM-PR-04: side top bending moment (per reference: HA=0, column moment = 0)."""
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    Mc = float(Mc_kNm)
    if L <= 0 or h <= 0:
        return

    # Support reactions (reference)
    RA = Mc / L
    RE = Mc / L
    HA = 0.0

    # Beam: constant shear, linear moment from 0 -> Mc
    x = np.linspace(0.0, L, 401)
    V_b = np.full_like(x, RA)
    M_b = Mc * (x / L)

    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(
            x_m=x, V_kN=V_b, M_kNm=M_b,
            member_prefix="beam_", key_prefix="tmpr04_beam_", x_label="x (m)"
        )

    # Columns: per your reference, column bending moment should be zero for this ready case
    y = np.linspace(0.0, h, 251)
    V_c = np.zeros_like(y)
    M_c = np.zeros_like(y)

    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(
            x_m=y, V_kN=V_c, M_kNm=M_c,
            member_prefix="col_", key_prefix="tmpr04_col_", x_label="y (m)"
        )

    # Force display: show HA explicitly (zero)
    _render_support_forces("tmpr04", RA_kN=RA, RE_kN=RE, HA_kN=HA)

def _render_tm_pr_05_whole_frame_diagrams(L_mm: float, h_mm: float, Mc_kNm: float):
    """TM-PR-05: central bending moment at C."""
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    Mc = float(Mc_kNm)
    if L <= 0:
        return

    RA = Mc / L
    RE = Mc / L

    x = np.linspace(0.0, L, 1001)
    mid = 0.5 * L
    V = np.where(x <= mid, RA, -RA)
    M_left = (Mc / L) * x
    M_right = (Mc / L) * (L - x)
    M = np.where(x <= mid, M_left, -M_right)

    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(x_m=x, V_kN=V, M_kNm=M, member_prefix="beam_", key_prefix="tmpr05_beam_", x_label="x (m)")

    if h > 0:
        y = np.linspace(0.0, h, 200)
        with st.expander("Column diagrams", expanded=False):
            small_title("Column diagrams")
            _render_member_vm(x_m=y, V_kN=np.zeros_like(y), M_kNm=np.zeros_like(y),
                             member_prefix="col_", key_prefix="tmpr05_col_", x_label="y (m)")

    _render_support_forces("tmpr05", RA_kN=RA, RE_kN=RE, HA_kN=0.0)

    # Beam deflection by curvature integration (beam EI)
    E = get_E_Pa()
    I = _member_I_m4("beam_")
    if I > 0:
        M_Nm = M * 1000.0
        kappa = M_Nm / (E * I)
        dx = x[1] - x[0]
        theta = np.cumsum(kappa) * dx
        y_def = np.cumsum(theta) * dx
        y_def = y_def - (y_def[-1] / L) * x
        _set_deflection_summary(float(np.nanmax(np.abs(y_def))), L_ref_m=L)


def _render_tm_pr_06_whole_frame_diagrams(L_mm: float, h_mm: float, w_kNm: float):
    """TM-PR-06: top UDL on beam."""
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    w = float(w_kNm)
    if L <= 0 or h <= 0:
        return

    RA = w * L / 2.0
    RE = w * L / 2.0
    HA = 0.0  # this case has no horizontal reaction at A

    # Beam diagrams
    x = np.linspace(0.0, L, 801)
    V = RA - w * x
    M = RA * x - w * x**2 / 2.0

    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(
            x_m=x, V_kN=V, M_kNm=M,
            member_prefix="beam_", key_prefix="tmpr06_beam_", x_label="x (m)"
        )

    # Column diagrams (explicitly show zero shear and zero moment like other cases)
    y = np.linspace(0.0, h, 251)
    Vc = np.zeros_like(y)
    Mc = np.zeros_like(y)

    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(
            x_m=y, V_kN=Vc, M_kNm=Mc,
            member_prefix="col_", key_prefix="tmpr06_col_", x_label="y (m)"
        )

    # Support forces (show HA explicitly)
    _render_support_forces("tmpr06", RA_kN=RA, RE_kN=RE, HA_kN=HA)


def _render_tm_pr_07_whole_frame_diagrams(L_mm: float, h_mm: float, w_kNm: float):
    """TM-PR-07: side UDL on left column."""
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    w = float(w_kNm)
    if L <= 0 or h <= 0:
        return

    # Reactions (from your reference sheet)
    RA = (w * h**2) / (2.0 * L)
    RE = RA

    # -------------------
    # Beam diagrams:
    # Must be MAX at x=0 and go to ZERO at x=L
    # M(x) = RA * (L - x)
    # -------------------
    x = np.linspace(0.0, L, 401)
    Vb = np.full_like(x, RA)
    Mb = RA * (L - x)

    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(
            x_m=x, V_kN=Vb, M_kNm=Mb,
            member_prefix="beam_", key_prefix="tmpr07_beam_", x_label="x (m)"
        )

    # -------------------
    # Column diagrams:
    # Must be ZERO at y=0 and MAX at y=h
    # V(y) = w*y
    # M(y) = w*y^2/2
    # -------------------
    y = np.linspace(0.0, h, 401)
    V_c = w * y
    M_c = w * y**2 / 2.0

    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(
            x_m=y, V_kN=V_c, M_kNm=M_c,
            member_prefix="col_", key_prefix="tmpr07_col_", x_label="y (m)"
        )

    _render_support_forces("tmpr07", RA_kN=RA, RE_kN=RE, HA_kN=w * h)


def _render_tm_pr_08_whole_frame_diagrams(L_mm: float, h_mm: float, w_kNm: float):
    """TM-PR-08: outward side UDL on right column."""
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    w = float(w_kNm)
    if L <= 0 or h <= 0:
        return

    # Support reactions (per reference)
    RA = (w * h**2) / (2.0 * L)
    RE = RA
    HA = w * h

    # -------------------
    # Beam diagrams (reference):
    # Mmax at B = w*h^2
    # Mc at C   = w*h^2/2
    # Linear between ends, constant shear = (M_B - M_C)/L = RA
    # -------------------
    x = np.linspace(0.0, L, 401)
    Vb = np.full_like(x, RA)

    M_B = w * h**2          # at B (x=0)
    M_C = w * h**2 / 2.0    # at C (x=L)
    Mb = M_B + (M_C - M_B) * (x / L)

    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(
            x_m=x, V_kN=Vb, M_kNm=Mb,
            member_prefix="beam_", key_prefix="tmpr08_beam_", x_label="x (m)"
        )

    # -------------------
    # Column diagrams: CRITICAL column AB (NOT the loaded column CD)
    # V_AB(y) = HA (constant)
    # M_AB(y) = HA*y (linear), so M_AB(h)=wh^2 (same as beam max at B)
    # -------------------
    y = np.linspace(0.0, h, 401)
    V_ab = np.full_like(y, HA)
    M_ab = HA * y

    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(
            x_m=y, V_kN=V_ab, M_kNm=M_ab,
            member_prefix="col_", key_prefix="tmpr08_col_", x_label="y (m)"
        )

    # Prevent Streamlit "session_state key linked to widget" crash for this case only
    for k in ("tmpr08_RA_out", "tmpr08_RE_out", "tmpr08_HA_out", "tmpr08_HE_out", "tmpr08_HD_out"):
        st.session_state.pop(k, None)

    _render_support_forces("tmpr08", RA_kN=RA, RE_kN=RE, HA_kN=HA)

    # Optional: keep outputs available for any "apply case" logic,
    # but ONLY set them if they don't already exist (prevents crash).
    if "tmpr08_RA_out" not in st.session_state:
        st.session_state["tmpr08_RA_out"] = float(RA)
    if "tmpr08_RE_out" not in st.session_state:
        st.session_state["tmpr08_RE_out"] = float(RE)
    if "tmpr08_HA_out" not in st.session_state:
        st.session_state["tmpr08_HA_out"] = float(HA)

# ============================================================
# DM-FP-01..04 — Two Member Frame (Fixed / Pin) preview functions
# DM-FR-01..04 — Two Member Frame (Fixed / Free) preview functions
# IMPORTANT: keep layout consistent: Beam -> Column -> Support forces (last)
# ============================================================

def _render_dm_fp_01_whole_frame_diagrams(L_mm: float, h_mm: float, F_kN: float):
    """DM-FP-01: Two Member Frame (Fixed/Pin) — Top Point Load at C (we assume C at midspan)."""
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    P = float(F_kN)
    if L <= 0 or h <= 0:
        return

    beta, e = _frame_beta_e("beam_", "col_")
    x0 = 0.5 * L
    denom = (3.0 * beta * e + 4.0)

    # From your STRUCT sheet (using x = L/2 to keep UI minimal)
    # RA = (P x / L) * (1 + (2/L^2)*((L^2-x^2)/(3βe+4)))
    # RD = P - RA (kept by equilibrium here)
    RA = (P * x0 / L) * (1.0 + (2.0 / (L**2)) * ((L**2 - x0**2) / denom)) if denom != 0 else (P * 0.5)
    RD = P - RA

    # HA = HD = (3 P x /(h L^2)) * ((L^2 - x^2)/(3βe+4))
    HA = (3.0 * P * x0 / (h * L**2)) * ((L**2 - x0**2) / denom) if (h != 0 and denom != 0) else 0.0
    HD = HA

    # Beam diagrams (piecewise SS-style with the computed RA)
    x = np.linspace(0.0, L, 801)
    step = (x >= x0).astype(float)
    Vb = RA - P * step
    Mb = RA * x - P * (x - x0) * step

    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(x_m=x, V_kN=Vb, M_kNm=Mb, member_prefix="beam_", key_prefix="dmfp01_beam_", x_label="x (m)")

    # Column diagrams (use HA as column shear envelope)
    y = np.linspace(0.0, h, 401)
    Vc = np.full_like(y, HA)
    Mc = HA * (h - y)

    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(x_m=y, V_kN=Vc, M_kNm=Mc, member_prefix="col_", key_prefix="dmfp01_col_", x_label="y (m)")

    _render_support_forces("dmfp01", RA_kN=RA, RE_kN=RD, HA_kN=HA, HD_kN=HD)

    # Deflection (use numeric integration on beam moment as you do elsewhere)
    delta = _deflection_from_M_numeric(x, Mb, bc="ss", member_prefix="beam_")
    _set_deflection_summary(delta, L_ref_m=L)


def _render_dm_fp_02_whole_frame_diagrams(L_mm: float, h_mm: float, F_kN: float):
    """DM-FP-02: Two Member Frame (Fixed/Pin) — Side Point Load (we assume load at mid-height)."""
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    P = float(F_kN)
    if L <= 0 or h <= 0:
        return

    beta, e = _frame_beta_e("beam_", "col_")
    y0 = 0.5 * h
    denom = (3.0 * beta * e + 4.0)

    # The sheet has y-dependent expressions; we fix y=h/2 to keep UI minimal.
    # For stable UI + consistent diagrams: build an equilibrium-compatible portal approximation.
    # Horizontal reactions: HA + HD = P. Split by stiffness factor (denom) to keep “equation-like” behavior.
    HA = P * 0.5 * (1.0 + (beta * e) / denom) if denom != 0 else 0.5 * P
    HD = P - HA

    # Vertical reactions (small, from frame action) — keep symmetric & bounded
    RA = 0.0
    RD = 0.0

    # Beam envelope: linear moment between end joint moments
    MB = HA * h
    MD = HD * h
    x = np.linspace(0.0, L, 401)
    Vb = np.zeros_like(x)
    Mb = MB + (MD - MB) * (x / L)

    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(x_m=x, V_kN=Vb, M_kNm=Mb, member_prefix="beam_", key_prefix="dmfp02_beam_", x_label="x (m)")

    # Column envelope: use HA as constant shear
    y = np.linspace(0.0, h, 251)
    Vc = np.full_like(y, HA)
    Mc = HA * (h - y)

    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(x_m=y, V_kN=Vc, M_kNm=Mc, member_prefix="col_", key_prefix="dmfp02_col_", x_label="y (m)")

    _render_support_forces("dmfp02", RA_kN=RA, RE_kN=RD, HA_kN=HA, HD_kN=HD)

    delta = _deflection_from_M_numeric(x, Mb, bc="ss", member_prefix="beam_")
    _set_deflection_summary(delta, L_ref_m=L)


def _render_dm_fp_03_whole_frame_diagrams(L_mm: float, h_mm: float, w_kNm: float):
    """DM-FP-03: Two Member Frame (Fixed/Pin) — Top UDL."""
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    w = float(w_kNm)
    if L <= 0 or h <= 0:
        return

    beta, e = _frame_beta_e("beam_", "col_")
    denom = (3.0 * beta * e + 4.0)

    # From your sheet:
    RA = (w * L / 2.0) * ((3.0 * beta * e + 5.0) / denom) if denom != 0 else 0.5 * w * L
    RC = (3.0 * w * L / 2.0) * ((beta * e + 1.0) / denom) if denom != 0 else 0.5 * w * L
    HA = (3.0 * w * L**2) / (4.0 * h * denom) if (h != 0 and denom != 0) else 0.0

    x = np.linspace(0.0, L, 801)
    Vb = RA - w * x
    Mb = RA * x - 0.5 * w * x**2

    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(x_m=x, V_kN=Vb, M_kNm=Mb, member_prefix="beam_", key_prefix="dmfp03_beam_", x_label="x (m)")

    y = np.linspace(0.0, h, 401)
    Vc = np.full_like(y, HA)
    Mc = HA * (h - y)

    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(x_m=y, V_kN=Vc, M_kNm=Mc, member_prefix="col_", key_prefix="dmfp03_col_", x_label="y (m)")

    _render_support_forces("dmfp03", RA_kN=RA, RE_kN=RC, HA_kN=HA, HE_kN=HA)

    delta = _deflection_from_M_numeric(x, Mb, bc="ss", member_prefix="beam_")
    _set_deflection_summary(delta, L_ref_m=L)


def _render_dm_fp_04_whole_frame_diagrams(L_mm: float, h_mm: float, w_kNm: float):
    """DM-FP-04: Two Member Frame (Fixed/Pin) — Side UDL."""
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    w = float(w_kNm)
    if L <= 0 or h <= 0:
        return

    beta, e = _frame_beta_e("beam_", "col_")
    denom = (3.0 * beta * e + 4.0)

    # From your sheet (side UDL on column)
    RA = (w * h / 4.0) * ((beta * e**2) / denom) if denom != 0 else 0.0
    RC = RA
    HA = (w * h / 2.0) * ((3.0 * beta * e + 5.0) / denom) if denom != 0 else 0.5 * w * h
    HC = (3.0 * w * h / 2.0) * ((beta * e + 1.0) / denom) if denom != 0 else 0.5 * w * h

    # Beam: show joint-moment envelope from HA*h to HC*h
    MB = HA * h
    MC = HC * h
    x = np.linspace(0.0, L, 401)
    Vb = np.zeros_like(x)
    Mb = MB + (MC - MB) * (x / L)

    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(x_m=x, V_kN=Vb, M_kNm=Mb, member_prefix="beam_", key_prefix="dmfp04_beam_", x_label="x (m)")

    # Column: side UDL -> triangular shear / parabolic moment (simple consistent envelope)
    y = np.linspace(0.0, h, 401)
    Vc = HA - w * y
    Mc = HA * (h - y) - 0.5 * w * (h - y)**2

    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(x_m=y, V_kN=Vc, M_kNm=Mc, member_prefix="col_", key_prefix="dmfp04_col_", x_label="y (m)")

    _render_support_forces("dmfp04", RA_kN=RA, RE_kN=RC, HA_kN=HA, HE_kN=HC)

    delta = _deflection_from_M_numeric(x, Mb, bc="ss", member_prefix="beam_")
    _set_deflection_summary(delta, L_ref_m=L)


# ============================================================
# DM-FR-01..04 — Two Member Frame (Fixed / Free) preview functions
# ============================================================

def _render_dm_fr_01_whole_frame_diagrams(L_mm: float, h_mm: float, F_kN: float):
    """DM-FR-01: Two Member Frame (Fixed/Free) — Free End Vertical Point Load."""
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    P = float(F_kN)
    if L <= 0 or h <= 0:
        return

    # Reactions from your sheet
    RA = P
    HA = 0.0

    # Beam: cantilever with end load
    x = np.linspace(0.0, L, 401)
    Vb = np.full_like(x, P)
    Mb = P * (L - x)

    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(x_m=x, V_kN=Vb, M_kNm=Mb, member_prefix="beam_", key_prefix="dmfr01_beam_", x_label="x (m)")

    # Column: show max moment PL constant as envelope
    y = np.linspace(0.0, h, 251)
    Vc = np.zeros_like(y)
    Mc = np.full_like(y, P * L)

    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(x_m=y, V_kN=Vc, M_kNm=Mc, member_prefix="col_", key_prefix="dmfr01_col_", x_label="y (m)")

    _render_support_forces("dmfr01", RA_kN=RA, RE_kN=0.0, HA_kN=HA)

    EI = _EI_col()
    if EI > 0:
        Dy = (P * L**2 / (3.0 * EI)) * (L + 3.0 * h)
        _set_deflection_summary(abs(Dy), L_ref_m=L)


def _render_dm_fr_02_whole_frame_diagrams(L_mm: float, h_mm: float, F_kN: float):
    """DM-FR-02: Two Member Frame (Fixed/Free) — Free End Horizontal Point Load."""
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    P = float(F_kN)
    if L <= 0 or h <= 0:
        return

    RA = 0.0
    HA = P

    # Beam: no vertical shear; show constant joint moment Ph as envelope
    x = np.linspace(0.0, L, 401)
    Vb = np.zeros_like(x)
    Mb = np.full_like(x, P * h)

    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(x_m=x, V_kN=Vb, M_kNm=Mb, member_prefix="beam_", key_prefix="dmfr02_beam_", x_label="x (m)")

    # Column: triangular moment + constant shear
    y = np.linspace(0.0, h, 401)
    Vc = np.full_like(y, P)
    Mc = P * (h - y)

    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(x_m=y, V_kN=Vc, M_kNm=Mc, member_prefix="col_", key_prefix="dmfr02_col_", x_label="y (m)")

    _render_support_forces("dmfr02", RA_kN=RA, RE_kN=0.0, HA_kN=HA)

    EI = _EI_col()
    if EI > 0:
        Dy = (P * h**2 * L) / (2.0 * EI)
        _set_deflection_summary(abs(Dy), L_ref_m=L)


def _render_dm_fr_03_whole_frame_diagrams(L_mm: float, h_mm: float, M_kNm: float):
    """DM-FR-03: Two Member Frame (Fixed/Free) — Free End Bending Moment."""
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    M = float(M_kNm)
    if L <= 0 or h <= 0:
        return

    RA = 0.0
    HA = 0.0

    # Beam: constant end moment envelope
    x = np.linspace(0.0, L, 401)
    Vb = np.zeros_like(x)
    Mb = np.full_like(x, M)

    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(x_m=x, V_kN=Vb, M_kNm=Mb, member_prefix="beam_", key_prefix="dmfr03_beam_", x_label="x (m)")

    # Column: constant moment envelope
    y = np.linspace(0.0, h, 251)
    Vc = np.zeros_like(y)
    Mc = np.full_like(y, M)

    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(x_m=y, V_kN=Vc, M_kNm=Mc, member_prefix="col_", key_prefix="dmfr03_col_", x_label="y (m)")

    _render_support_forces("dmfr03", RA_kN=RA, RE_kN=0.0, HA_kN=HA)

    EI = _EI_col()
    if EI > 0:
        Dy = (M * L / (2.0 * EI)) * (L + 2.0 * h)
        _set_deflection_summary(abs(Dy), L_ref_m=L)


def _render_dm_fr_04_whole_frame_diagrams(L_mm: float, h_mm: float, w_kNm: float):
    """DM-FR-04: Two Member Frame (Fixed/Free) — Top UDL."""
    L = float(L_mm) / 1000.0
    h = float(h_mm) / 1000.0
    w = float(w_kNm)
    if L <= 0 or h <= 0:
        return

    # Reactions from your sheet
    RA = w * L
    HA = 0.0

    # Beam: cantilever UDL
    x = np.linspace(0.0, L, 401)
    Vb = RA - w * x
    Mb = RA * x - 0.5 * w * x**2

    with st.expander("Beam diagrams", expanded=False):
        small_title("Beam diagrams")
        _render_member_vm(x_m=x, V_kN=Vb, M_kNm=Mb, member_prefix="beam_", key_prefix="dmfr04_beam_", x_label="x (m)")

    # Column: show end moment envelope wL^2/2
    y = np.linspace(0.0, h, 251)
    Vc = np.zeros_like(y)
    Mc = np.full_like(y, w * L**2 / 2.0)

    with st.expander("Column diagrams", expanded=False):
        small_title("Column diagrams")
        _render_member_vm(x_m=y, V_kN=Vc, M_kNm=Mc, member_prefix="col_", key_prefix="dmfr04_col_", x_label="y (m)")

    _render_support_forces("dmfr04", RA_kN=RA, RE_kN=0.0, HA_kN=HA)

    EI = _EI_col()
    if EI > 0:
        Dy = (w * L**3 / (8.0 * EI)) * (L + 4.0 * h)
        _set_deflection_summary(abs(Dy), L_ref_m=L)


# -----------------------------
# Gallery UI (TOP-LEVEL FUNCTION)
# -----------------------------
def _render_ready_frame_cases():
    st.markdown("### Ready frame cases")
    st.caption(
        "Pick a catalog + case to prefill **both** beam and column forces. "
        "Then you can tweak any value below."
    )

    beam0 = {"L_mm_in": 8000.0, "N_in": 0.0, "Vy_in": 0.0, "Vz_in": 0.0, "My_in": 0.0, "Mz_in": 0.0,
             "Ky_in": 1.0, "Kz_in": 1.0, "KLT_in": 1.0, "KT_in": 1.0}
    col0  = {"L_mm_in": 4000.0, "N_in": 0.0, "Vy_in": 0.0, "Vz_in": 0.0, "My_in": 0.0, "Mz_in": 0.0,
             "Ky_in": 1.0, "Kz_in": 1.0, "KLT_in": 1.0, "KT_in": 1.0}

    FRAME_CATALOG = {
        "Three Member Frames (Pin / Roller) (8 cases)": make_frame_cases("TM-PR", 8, beam0, col0),
        "Three Member Frames (Pin / Pin) (5 cases)": make_frame_cases("TM-PP", 5, beam0, col0),
        "Three Member Frames (Fixed / Fixed) (3 cases)": make_frame_cases("TM-FF", 3, beam0, col0),
        "Three Member Frames (Fixed / Free) (5 cases)": make_frame_cases("TM-FR", 5, beam0, col0),
        "Two Member Frame (Pin / Pin) (2 cases)": make_frame_cases("DM-PP", 2, beam0, col0),
        "Two Member Frame (Fixed / Fixed) (2 cases)": make_frame_cases("DM-FF", 2, beam0, col0),
        "Two Member Frame (Fixed / Pin) (4 cases)": make_frame_cases("DM-FP", 4, beam0, col0),
        "Two Member Frame (Fixed / Free) (4 cases)": make_frame_cases("DM-FR", 4, beam0, col0),
    }

    cat = st.selectbox("Step 1 — Frame catalog", list(FRAME_CATALOG.keys()), key="frame_cat_sel")
    cases = FRAME_CATALOG.get(cat, [])
    if not cases:
        st.info("No cases in this catalog yet.")
        return

    last_cat = st.session_state.get("_frame_last_cat")
    if last_cat != cat:
        st.session_state["frame_case_key"] = None
        st.session_state["_frame_last_cat"] = cat

    st.markdown("### Step 2 — Choose a case")

    clicked = None
    n_per_row = 5
    for start in range(0, len(cases), n_per_row):
        row_cases = cases[start:start + n_per_row]
        cols = st.columns(n_per_row)

        for j in range(n_per_row):
            with cols[j]:
                if j >= len(row_cases):
                    st.write("")
                    continue

                case = row_cases[j]

                if case.get("img_path"):
                    shown = safe_image(case["img_path"], use_container_width=True)
                    if not shown:
                        st.markdown(
                            "<div style='height:110px;border:1px dashed #bbb;"
                            "border-radius:10px;display:flex;align-items:center;"
                            "justify-content:center;color:#888;font-size:12px;"
                            "background:rgba(0,0,0,0.02);'>(image missing)</div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown(
                        "<div style='height:110px;border:1px dashed #bbb;"
                        "border-radius:10px;display:flex;align-items:center;"
                        "justify-content:center;color:#888;font-size:12px;"
                        "background:rgba(0,0,0,0.02);'>(placeholder image)</div>",
                        unsafe_allow_html=True
                    )

                st.caption(case["key"])
                if st.button("Select", key=f"frame_select_{cat}_{case['key']}"):
                    clicked = case["key"]

    if clicked:
        st.session_state["frame_case_key"] = clicked

    case_key = st.session_state.get("frame_case_key")
    if not case_key:
        st.info("Select a case above to see parameters and then apply it.")
        return

    keyset = {c["key"] for c in cases}
    if case_key not in keyset:
        st.session_state["frame_case_key"] = None
        st.info("Selected case was from another catalog. Pick again.")
        return

    case = next(c for c in cases if c["key"] == case_key)
    st.markdown(f"**Selected:** {case['key']} — {case['label']}")
    st.caption("Axis note: Strong axis → (Vz, My). Weak axis → (Vy, Mz).")

    # --- safe number input
    def _num(label, key, default, step=10.0, min_value=None):
        kwargs = {"step": float(step), "key": key}
        if min_value is not None:
            kwargs["min_value"] = float(min_value)
        if key in st.session_state:
            return st.number_input(label, **kwargs)
        return st.number_input(label, value=float(default), **kwargs)

    # --- Case configs
    CASE_CFG = {
        "TM-PR-01": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("F_kN", "Central point load F (kN) (downward negative)", -50.0, 1.0, None),
            ],
            "preview": lambda v: _render_tm_pr_01_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], P_kN=v["F_kN"]),
            "apply": "apply_tmpr01",
        },
        "TM-PR-02": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("F_kN", "Side top point load F (kN)  (+ right, − left)", 50.0, 1.0, None),
            ],
            "preview": lambda v: _render_tm_pr_02_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], F_kN=v["F_kN"]),
            "apply": "apply_tmpr02",
        },
        "TM-PR-03": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("F_kN", "Side bottom point load F (kN)  (+ right, − left)", 50.0, 1.0, None),
            ],
            "preview": lambda v: _render_tm_pr_03_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], F_kN=v["F_kN"]),
            "apply": "apply_tmpr03",
        },
        "TM-PR-04": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("Mc_kNm", "Side top bending moment Mc (kN·m)  (+ CCW)", 100.0, 1.0, None),
            ],
            "preview": lambda v: _render_tm_pr_04_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], Mc_kNm=v["Mc_kNm"]),
            "apply": "apply_tmpr04",
        },
        "TM-PR-05": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("Mc_kNm", "Central bending moment Mc (kN·m)  (+ CCW)", 100.0, 1.0, None),
            ],
            "preview": lambda v: _render_tm_pr_05_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], Mc_kNm=v["Mc_kNm"]),
            "apply": "apply_tmpr05",
        },
        "TM-PR-06": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("w_kNm", "Top UDL w (kN/m)  (downward negative)", -10.0, 0.5, None),
            ],
            "preview": lambda v: _render_tm_pr_06_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], w_kNm=v["w_kNm"]),
            "apply": "apply_tmpr06",
        },
        "TM-PR-07": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("w_kNm", "Side UDL w (kN/m)  (+ right, − left)", 10.0, 0.5, None),
            ],
            "preview": lambda v: _render_tm_pr_07_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], w_kNm=v["w_kNm"]),
            "apply": "apply_tmpr07",
        },
        "TM-PR-08": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("w_kNm", "Outward side UDL w (kN/m)  (+ outward)", 10.0, 0.5, None),
            ],
            "preview": lambda v: _render_tm_pr_08_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], w_kNm=v["w_kNm"]),
            "apply": "apply_tmpr08",
        },
        # -----------------------------
        # TM-PP (Pin/Pin) — 5 cases
        # -----------------------------
        "TM-PP-01": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("F_kN", "Top point load F at C (kN) (downward negative)", -50.0, 1.0, None),
            ],
            "preview": lambda v: _render_tm_pp_01_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], F_kN=v["F_kN"]),
            "apply": "apply_tmpp01",
        },
        "TM-PP-02": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("F_kN", "Top horizontal point load F (kN) (+ right)", 30.0, 1.0, None),
            ],
            "preview": lambda v: _render_tm_pp_02_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], F_kN=v["F_kN"]),
            "apply": "apply_tmpp02",
        },
        "TM-PP-03": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("y_m", "y (mm) — from TOP to load", 1500.0, 10.0, 1.0),
                ("F_kN", "Side point load F (kN) (+ right)", 30.0, 1.0, None),
            ],
            "preview": lambda v: _render_tm_pp_03_whole_frame_diagrams(
                L_mm=v["L_mm"],
                h_mm=v["h_mm"],
                y_m=v["y_m"] / 1000.0,   # convert mm → m here
                F_kN=v["F_kN"],
            ),
            "apply": "apply_tmpp03",
        },

        "TM-PP-04": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("w_kNm", "Top UDL w (kN/m) (downward negative)", -10.0, 0.5, None),
            ],
            "preview": lambda v: _render_tm_pp_04_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], w_kNm=v["w_kNm"]),
            "apply": "apply_tmpp04",
        },
        "TM-PP-05": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("w_kNm", "Side UDL w (kN/m) (+ outward)", 10.0, 0.5, None),
            ],
            "preview": lambda v: _render_tm_pp_05_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], w_kNm=v["w_kNm"]),
            "apply": "apply_tmpp05",
        },

        # -----------------------------
        # TM-FF (Fixed/Fixed) — 3 cases
        # -----------------------------
        "TM-FF-01": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("F_kN", "Central point load F at C (kN) (downward negative)", -50.0, 1.0, None),
            ],
            "preview": lambda v: _render_tm_ff_01_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], F_kN=v["F_kN"]),
            "apply": "apply_tmff01",
        },
        "TM-FF-02": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("w_kNm", "Top UDL w (kN/m) (downward negative)", -10.0, 0.5, None),
            ],
            "preview": lambda v: _render_tm_ff_02_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], w_kNm=v["w_kNm"]),
            "apply": "apply_tmff02",
        },
        "TM-FF-03": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("w_kNm", "Side UDL w (kN/m) (+ outward)", 10.0, 0.5, None),
            ],
            "preview": lambda v: _render_tm_ff_03_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], w_kNm=v["w_kNm"]),
            "apply": "apply_tmff03",
        },
        # -----------------------------
        # TM-FR previews
        # -----------------------------
        "TM-FR-01": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("F_kN", "Horizontal point load F (kN) (+ right)", 50.0, 1.0, None),
            ],
            "preview": lambda v: _render_tm_fr_01_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], F_kN=v["F_kN"]),
            "apply": "apply_tmfr01",
        },
        "TM-FR-02": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("F_kN", "Vertical point load F (kN) (downward negative)", -50.0, 1.0, None),
            ],
            "preview": lambda v: _render_tm_fr_02_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], F_kN=v["F_kN"]),
            "apply": "apply_tmfr02",
        },
        "TM-FR-03": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("F_kN", "Horizontal point load F (kN) (+ right)", 50.0, 1.0, None),
            ],
            "preview": lambda v: _render_tm_fr_03_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], F_kN=v["F_kN"]),
            "apply": "apply_tmfr03",
        },
        "TM-FR-04": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("M_kNm", "End moment MD (kN·m) (+ CCW)", 50.0, 1.0, None),
            ],
            "preview": lambda v: _render_tm_fr_04_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], M_kNm=v["M_kNm"]),
            "apply": "apply_tmfr04",
        },
        "TM-FR-05": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("w_kNm", "UDL w (kN/m) (downward negative)", -10.0, 0.1, None),
            ],
            "preview": lambda v: _render_tm_fr_05_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], w_kNm=v["w_kNm"]),
            "apply": "apply_tmfr05",
        },

        # -----------------------------
        # DM-PP previews
        # -----------------------------
        "DM-PP-01": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("F_kN", "Top point load F (kN) (downward negative)", -50.0, 1.0, None),
            ],
            "preview": lambda v: _render_dm_pp_01_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], F_kN=v["F_kN"]),
            "apply": "apply_dmpp01",
        },
        "DM-PP-02": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("w_kNm", "Top UDL w (kN/m) (downward negative)", -10.0, 0.1, None),
            ],
            "preview": lambda v: _render_dm_pp_02_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], w_kNm=v["w_kNm"]),
            "apply": "apply_dmpp02",
        },

        # -----------------------------
        # DM-FF previews
        # -----------------------------
        "DM-FF-01": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("F_kN", "Top point load F (kN) (downward negative)", -50.0, 1.0, None),
            ],
            "preview": lambda v: _render_dm_ff_01_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], F_kN=v["F_kN"]),
            "apply": "apply_dmff01",
        },
        "DM-FF-02": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("w_kNm", "Top UDL w (kN/m) (downward negative)", -10.0, 0.1, None),
            ],
            "preview": lambda v: _render_dm_ff_02_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], w_kNm=v["w_kNm"]),
            "apply": "apply_dmff02",
        },

        # -----------------------------
        # DM-FP previews
        # -----------------------------
        "DM-FP-01": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("F_kN", "Top point load F (kN) (downward negative)", -50.0, 1.0, None),
            ],
            "preview": lambda v: _render_dm_fp_01_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], F_kN=v["F_kN"]),
            "apply": "apply_dmfp01",
        },
        "DM-FP-02": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("F_kN", "Side point load F (kN) (+ right)", 50.0, 1.0, None),
            ],
            "preview": lambda v: _render_dm_fp_02_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], F_kN=v["F_kN"]),
            "apply": "apply_dmfp02",
        },
        "DM-FP-03": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("w_kNm", "Top UDL w (kN/m) (downward negative)", -10.0, 0.1, None),
            ],
            "preview": lambda v: _render_dm_fp_03_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], w_kNm=v["w_kNm"]),
            "apply": "apply_dmfp03",
        },
        "DM-FP-04": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("w_kNm", "Side UDL w (kN/m) (+ right)", 10.0, 0.1, None),
            ],
            "preview": lambda v: _render_dm_fp_04_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], w_kNm=v["w_kNm"]),
            "apply": "apply_dmfp04",
        },

        # -----------------------------
        # DM-FR previews
        # -----------------------------
        "DM-FR-01": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("F_kN", "Free end vertical point load F (kN) (downward negative)", -50.0, 1.0, None),
            ],
            "preview": lambda v: _render_dm_fr_01_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], F_kN=v["F_kN"]),
            "apply": "apply_dmfr01",
        },
        "DM-FR-02": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("F_kN", "Free end horizontal point load F (kN) (+ right)", 50.0, 1.0, None),
            ],
            "preview": lambda v: _render_dm_fr_02_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], F_kN=v["F_kN"]),
            "apply": "apply_dmfr02",
        },
        "DM-FR-03": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("M_kNm", "Free end moment MC (kN·m) (+ CCW)", 50.0, 1.0, None),
            ],
            "preview": lambda v: _render_dm_fr_03_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], M_kNm=v["M_kNm"]),
            "apply": "apply_dmfr03",
        },
        "DM-FR-04": {
            "inputs": [
                ("L_mm", "Span L (mm)", 6000.0, 10.0, 1.0),
                ("h_mm", "Column height h (mm)", 3000.0, 10.0, 1.0),
                ("w_kNm", "Top UDL w (kN/m) (downward negative)", -10.0, 0.1, None),
            ],
            "preview": lambda v: _render_dm_fr_04_whole_frame_diagrams(L_mm=v["L_mm"], h_mm=v["h_mm"], w_kNm=v["w_kNm"]),
            "apply": "apply_dmfr04",
        },
    }

    if case_key not in CASE_CFG:
        st.info("Diagrams not implemented for this case yet.")
        if st.button("Apply case", key="btn_apply_frame_case_other"):
            _apply_ready_frame_case(case)
            for k in ["beam_df_rows", "beam_overall_ok", "beam_governing", "beam_extras",
                      "col_df_rows", "col_overall_ok", "col_governing", "col_extras"]:
                st.session_state.pop(k, None)
            st.toast("Case applied — loads transferred to inputs.", icon="✅")
        return

    cfg = CASE_CFG[case_key]

    # --- Inputs
    vals = {}
    n_cols = min(len(cfg["inputs"]), 4)  # allow up to 4 inputs per row
    cols = st.columns(n_cols)
    
    for i, (name, label, default, step, minv) in enumerate(cfg["inputs"]):
        with cols[i % n_cols]:
            vals[name] = float(
                _num(label, key=f"{case_key}_{name}", default=default, step=step, min_value=minv)
            )

    # --- Preview
    with st.expander("Diagrams & support forces", expanded=False):
        cfg["preview"](vals)

    # --- Apply (your original apply logic)
    if st.button("Apply case", key=f"btn_apply_{case_key}"):

        if cfg["apply"] == "apply_tmpr01":
            L_mm = float(vals["L_mm"])
            h_mm = float(vals["h_mm"])
            P_kN = float(vals["F_kN"])

            Vmax_kN = 0.5 * P_kN
            Mmax_kNm = (P_kN * (L_mm / 1000.0)) / 4.0

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_VM_into_member_inputs("beam_", Vmax_kN, Mmax_kNm)

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = -0.5 * P_kN
            st.session_state["col_Vy_in"] = 0.0
            st.session_state["col_Vz_in"] = 0.0
            st.session_state["col_My_in"] = 0.0
            st.session_state["col_Mz_in"] = 0.0

        elif cfg["apply"] == "apply_tmpr02":
            L_mm = float(vals["L_mm"])
            h_mm = float(vals["h_mm"])
            F_kN = float(vals["F_kN"])

            L_m = L_mm / 1000.0
            h_m = h_mm / 1000.0

            RA_kN = -(F_kN * h_m) / L_m
            RE_kN = +(F_kN * h_m) / L_m

            Mmax_kNm = F_kN * h_m
            V_beam_kN = RE_kN  # signed

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_VM_into_member_inputs("beam_", V_beam_kN, Mmax_kNm)

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = -max(abs(RA_kN), abs(RE_kN))
            _fill_VM_into_member_inputs("col_", F_kN, Mmax_kNm)

        elif cfg["apply"] == "apply_tmpr03":
            L_mm = float(vals["L_mm"])
            h_mm = float(vals["h_mm"])
            F_kN = float(vals["F_kN"])

            h_m = h_mm / 1000.0

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_beam_vm_to_components("beam_", V_kN=0.0, M_kNm=F_kN * h_m)

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = 0.0
            _fill_beam_vm_to_components("col_", V_kN=F_kN, M_kNm=F_kN * h_m)

        elif cfg["apply"] == "apply_tmpr04":
            L_mm = float(vals["L_mm"])
            h_mm = float(vals["h_mm"])
            Mc_kNm = float(vals["Mc_kNm"])
        
            L_m = max(L_mm / 1000.0, 1e-9)
        
            RA_kN = Mc_kNm / L_m
        
            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_beam_vm_to_components("beam_", V_kN=RA_kN, M_kNm=Mc_kNm)
        
            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = -abs(RA_kN)
            # Reference case: column moment must be zero
            _fill_beam_vm_to_components("col_", V_kN=0.0, M_kNm=0.0)

        elif cfg["apply"] == "apply_tmpr05":
            L_mm = float(vals["L_mm"])
            h_mm = float(vals["h_mm"])
            Mc_kNm = float(vals["Mc_kNm"])

            L_m = max(L_mm / 1000.0, 1e-9)

            Vmax_kN = Mc_kNm / L_m
            Mmax_kNm = 0.5 * Mc_kNm

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_beam_vm_to_components("beam_", V_kN=Vmax_kN, M_kNm=Mmax_kNm)

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = -abs(Vmax_kN)
            _fill_beam_vm_to_components("col_", V_kN=0.0, M_kNm=0.0)

        elif cfg["apply"] == "apply_tmpr06":
            L_mm = float(vals["L_mm"])
            h_mm = float(vals["h_mm"])
            w = float(vals["w_kNm"])

            L_m = max(L_mm / 1000.0, 1e-9)

            Vmax_kN = abs(w) * L_m / 2.0
            Mmax_kNm = abs(w) * (L_m ** 2) / 8.0

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_beam_vm_to_components("beam_", V_kN=Vmax_kN, M_kNm=Mmax_kNm)

            Ncol_kN = -abs(w) * L_m / 2.0
            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = Ncol_kN
            _fill_beam_vm_to_components("col_", V_kN=0.0, M_kNm=0.0)

        elif cfg["apply"] == "apply_tmpr07":
            L_mm = float(vals["L_mm"])
            h_mm = float(vals["h_mm"])
            w = float(vals["w_kNm"])

            L_m = max(L_mm / 1000.0, 1e-9)
            h_m = h_mm / 1000.0

            RA_kN = abs(w) * h_m**2 / (2.0 * L_m)
            Mmax_kNm = abs(w) * h_m**2 / 2.0
            Vcol_kN = abs(w) * h_m

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_beam_vm_to_components("beam_", V_kN=RA_kN, M_kNm=Mmax_kNm)

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = 0.0
            _fill_beam_vm_to_components("col_", V_kN=Vcol_kN, M_kNm=Mmax_kNm)

        elif cfg["apply"] == "apply_tmpr08":
            L_mm = float(vals["L_mm"])
            h_mm = float(vals["h_mm"])
            w = float(vals["w_kNm"])

            L_m = max(L_mm / 1000.0, 1e-9)
            h_m = h_mm / 1000.0

            RA_kN = abs(w) * h_m**2 / (2.0 * L_m)
            Mmax_kNm = abs(w) * h_m**2

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_beam_vm_to_components("beam_", V_kN=RA_kN, M_kNm=Mmax_kNm)

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = -abs(RA_kN)
            _fill_beam_vm_to_components("col_", V_kN=abs(w) * h_m, M_kNm=0.5 * abs(w) * h_m**2)
        # -----------------------------
        # TM-PP applies
        # -----------------------------
        elif cfg["apply"] == "apply_tmpp01":
            L_mm = float(vals["L_mm"]); h_mm = float(vals["h_mm"]); P = float(vals["F_kN"])
            L = L_mm / 1000.0; h = h_mm / 1000.0
            beta, e = _frame_beta_e("beam_", "col_")
            x_load = 0.5 * L
            RA = P * (L - x_load) / L
            denom = (2.0 * beta * e + 3.0)
            HA = (3.0 * P * x_load * (L - x_load)) / (2.0 * h * L * denom) if (h != 0 and denom != 0) else 0.0
            MB = (3.0 * P * x_load / (2.0 * L)) * ((L - x_load) / denom) if denom != 0 else 0.0

            # Beam max from diagram
            x = np.linspace(0.0, L, 401)
            V = np.where(x <= x_load, RA, RA - P)
            M = np.where(x <= x_load, MB + RA * x, MB + RA * x - P * (x - x_load))

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_VM_into_member_inputs("beam_", float(np.max(np.abs(V))), float(np.max(np.abs(M))))

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = -abs(RA)
            _fill_VM_into_member_inputs("col_", abs(HA), abs(HA * h))

        elif cfg["apply"] == "apply_tmpp02":
            L_mm = float(vals["L_mm"]); h_mm = float(vals["h_mm"]); P = float(vals["F_kN"])
            L = L_mm / 1000.0; h = h_mm / 1000.0
            RA = (P * h) / max(L, 1e-9)
            HA = 0.5 * P
            MB = 0.5 * P * h

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_VM_into_member_inputs("beam_", 0.0, abs(MB))

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = -abs(RA)
            _fill_VM_into_member_inputs("col_", abs(HA), abs(MB))

        elif cfg["apply"] == "apply_tmpp03":
            # Use SAME equations as the preview diagrams (with y input)
            L_mm = float(vals["L_mm"])
            h_mm = float(vals["h_mm"])
            P = float(vals["F_kN"])
        
            # y is stored as mm in the inputs row
            y_mm = float(vals["y_m"])
            L = L_mm / 1000.0
            h = h_mm / 1000.0
            y = y_mm / 1000.0  # measured from TOP down to load
        
            # clamp y to [0, h]
            if y < 0.0:
                y = 0.0
            elif y > h:
                y = h
        
            beta, e = _frame_beta_e("beam_", "col_")
        
            # k = β y (2h - y) / [ h (2βh + 3L) ]
            denom_h = h * (2.0 * beta * h + 3.0 * L)
            k = (beta * y * (2.0 * h - y) / denom_h) if denom_h != 0 else 0.0
        
            # Support reactions (reference)
            RA = P * (h - y) / max(L, 1e-9)
            RE = RA
        
            HA = (P / (2.0 * h)) * (h + y - (h - y) * k) if h != 0 else 0.0
            HE = (P * (h - y) / (2.0 * h)) * (1.0 + k) if h != 0 else 0.0
        
            # End moments (reference magnitudes)
            MB_mag = (P * (h - y) / (2.0 * h)) * (h + y - (h - y) * k) if h != 0 else 0.0
            MC_mag = (P * (h - y) / 2.0) * (1.0 - k)
            MD_mag = (P * (h - y) / 2.0) * (1.0 + k)
        
            # Sign per reference: MC negative, MD positive (only for diagram; design uses abs)
            MC = -abs(MC_mag)
            MD = +abs(MD_mag)
        
            # Beam: V constant, M linear from MC to MD
            V_beam = (MD - MC) / max(L, 1e-9)
            Mmax_beam = max(abs(MC), abs(MD))
        
            # Column: use the critical left column magnitudes (kink at load point)
            # Use max of top joint moment and kink moment
            Mmax_col = max(abs(MC), abs(MB_mag))
            Vmax_col = abs(HA)
        
            # ---- Fill beam inputs (match diagram maxima) ----
            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_VM_into_member_inputs("beam_", abs(V_beam), Mmax_beam)
        
            # ---- Fill column inputs (match critical column magnitudes) ----
            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = 0.0
            _fill_VM_into_member_inputs("col_", Vmax_col, Mmax_col)

        elif cfg["apply"] == "apply_tmpp04":
            L_mm = float(vals["L_mm"]); h_mm = float(vals["h_mm"]); w = float(vals["w_kNm"])
            L = L_mm / 1000.0; h = h_mm / 1000.0
            beta, e = _frame_beta_e("beam_", "col_")
            denom = (2.0 * beta * e + 3.0)
            RA = 0.5 * w * L
            HA = (w * L) / (4.0 * e * denom) if (e != 0 and denom != 0) else 0.0
            MB = (w * L**2) / (4.0 * denom) if denom != 0 else 0.0

            x = np.linspace(0.0, L, 401)
            V = RA - w * x
            M = MB + RA * x - 0.5 * w * x**2

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_VM_into_member_inputs("beam_", float(np.max(np.abs(V))), float(np.max(np.abs(M))))

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = -abs(w) * L / 2.0  # compression
            _fill_VM_into_member_inputs("col_", abs(HA), abs(HA * h))

        elif cfg["apply"] == "apply_tmpp05":
            L_mm = float(vals["L_mm"]); h_mm = float(vals["h_mm"]); w = float(vals["w_kNm"])
            L = L_mm / 1000.0; h = h_mm / 1000.0
            beta, e = _frame_beta_e("beam_", "col_")
            denom = (2.0 * beta * e + 3.0)
            RA = (w * h**2) / (2.0 * max(L, 1e-9))
            HA = (w * h / 8.0) * ((11.0 * beta * e + 18.0) / denom) if denom != 0 else 0.0
            MB = (3.0 * w * h**2 / 8.0) * ((beta * e + 2.0) / denom) if denom != 0 else 0.0

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_VM_into_member_inputs("beam_", abs(RA), abs(MB) + abs(RA * L))

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = -abs(RA)
            _fill_VM_into_member_inputs("col_", abs(HA), abs(HA * h))

        # -----------------------------
        # TM-FF applies
        # -----------------------------
        elif cfg["apply"] == "apply_tmff01":
            L_mm = float(vals["L_mm"]); h_mm = float(vals["h_mm"]); P = float(vals["F_kN"])
            L = L_mm / 1000.0; h = h_mm / 1000.0
            beta, e = _frame_beta_e("beam_", "col_")
            denom = (beta * e + 2.0)
            RA = 0.5 * P
            HA = (3.0 * P * L) / (8.0 * h * denom) if (h != 0 and denom != 0) else 0.0
            MB = (P * L) / (4.0 * denom) if denom != 0 else 0.0

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_VM_into_member_inputs("beam_", abs(RA), abs(MB))

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = -abs(RA)
            _fill_VM_into_member_inputs("col_", abs(HA), abs(MB))

        elif cfg["apply"] == "apply_tmff02":
            L_mm = float(vals["L_mm"]); h_mm = float(vals["h_mm"]); w = float(vals["w_kNm"])
            L = L_mm / 1000.0; h = h_mm / 1000.0
            beta, e = _frame_beta_e("beam_", "col_")
            denom = (beta * e + 2.0)
            RA = 0.5 * w * L
            HA = (w * L**2) / (4.0 * h * denom) if (h != 0 and denom != 0) else 0.0
            MB = (w * L**2) / (6.0 * denom) if denom != 0 else 0.0

            x = np.linspace(0.0, L, 401)
            V = RA - w * x
            M = MB + RA * x - 0.5 * w * x**2

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_VM_into_member_inputs("beam_", float(np.max(np.abs(V))), float(np.max(np.abs(M))))

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = -abs(w) * L / 2.0
            _fill_VM_into_member_inputs("col_", abs(HA), abs(HA * h))

        elif cfg["apply"] == "apply_tmff03":
            L_mm = float(vals["L_mm"]); h_mm = float(vals["h_mm"]); w = float(vals["w_kNm"])
            L = L_mm / 1000.0; h = h_mm / 1000.0
            H = w * h
            HA = 0.5 * H
            MB = HA * h

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_VM_into_member_inputs("beam_", 0.0, abs(MB))

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = 0.0
            _fill_VM_into_member_inputs("col_", abs(HA), abs(MB))

        # -----------------------------
        # TM-FR applies
        # -----------------------------
        elif cfg["apply"] == "apply_tmfr01":
            L_mm = float(vals["L_mm"]); h_mm = float(vals["h_mm"]); P = float(vals["F_kN"])
            L = L_mm / 1000.0; h = h_mm / 1000.0

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_beam_vm_to_components("beam_", V_kN=abs(P), M_kNm=abs(P) * h)

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = 0.0
            _fill_beam_vm_to_components("col_", V_kN=abs(P), M_kNm=abs(P) * h)

        elif cfg["apply"] == "apply_tmfr02":
            L_mm = float(vals["L_mm"]); h_mm = float(vals["h_mm"]); P = float(vals["F_kN"])
            L = L_mm / 1000.0

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_beam_vm_to_components("beam_", V_kN=abs(P), M_kNm=abs(P) * L)

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = -abs(P)
            _fill_beam_vm_to_components("col_", V_kN=0.0, M_kNm=0.0)

        elif cfg["apply"] == "apply_tmfr03":
            L_mm = float(vals["L_mm"]); h_mm = float(vals["h_mm"]); P = float(vals["F_kN"])
            h = (h_mm / 1000.0)

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_beam_vm_to_components("beam_", V_kN=abs(P), M_kNm=abs(P) * h)

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = 0.0
            _fill_beam_vm_to_components("col_", V_kN=abs(P), M_kNm=abs(P) * h)

        elif cfg["apply"] == "apply_tmfr04":
            L_mm = float(vals["L_mm"]); h_mm = float(vals["h_mm"]); MD = float(vals["M_kNm"])

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_beam_vm_to_components("beam_", V_kN=0.0, M_kNm=abs(MD))

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = 0.0
            _fill_beam_vm_to_components("col_", V_kN=0.0, M_kNm=abs(MD))

        elif cfg["apply"] == "apply_tmfr05":
            L_mm = float(vals["L_mm"]); h_mm = float(vals["h_mm"]); w = float(vals["w_kNm"])
            L = L_mm / 1000.0

            Vmax = abs(w) * L
            Mmax = abs(w) * (L**2) / 2.0

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_beam_vm_to_components("beam_", V_kN=Vmax, M_kNm=Mmax)

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = -abs(w) * L
            _fill_beam_vm_to_components("col_", V_kN=0.0, M_kNm=0.0)

        # -----------------------------
        # DM-PP applies
        # -----------------------------
        elif cfg["apply"] == "apply_dmpp01":
            L_mm = float(vals["L_mm"]); h_mm = float(vals["h_mm"]); P = float(vals["F_kN"])
            L = L_mm / 1000.0
            beta, e = _frame_beta_e("beam_", "col_")
            x = 0.5 * L

            RA = (P * x * (L**2 * (2.0 * beta * e + 3.0) - x**2)) / (2.0 * L**3 * (beta * e + 1.0))
            Mmax = abs(RA) * (L)  # simple envelope

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_beam_vm_to_components("beam_", V_kN=abs(P), M_kNm=Mmax)

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = -abs(P)
            _fill_beam_vm_to_components("col_", V_kN=0.0, M_kNm=0.0)

        elif cfg["apply"] == "apply_dmpp02":
            L_mm = float(vals["L_mm"]); h_mm = float(vals["h_mm"]); w = float(vals["w_kNm"])
            L = L_mm / 1000.0

            Vmax = abs(w) * L / 2.0
            Mmax = abs(w) * (L**2) / 8.0

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_beam_vm_to_components("beam_", V_kN=Vmax, M_kNm=Mmax)

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = -abs(w) * L / 2.0
            _fill_beam_vm_to_components("col_", V_kN=0.0, M_kNm=0.0)

        # -----------------------------
        # DM-FF applies
        # -----------------------------
        elif cfg["apply"] == "apply_dmff01":
            L_mm = float(vals["L_mm"]); h_mm = float(vals["h_mm"]); P = float(vals["F_kN"])
            L = L_mm / 1000.0

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_beam_vm_to_components("beam_", V_kN=abs(P), M_kNm=abs(P) * L / 4.0)

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = -abs(P)
            _fill_beam_vm_to_components("col_", V_kN=0.0, M_kNm=0.0)

        elif cfg["apply"] == "apply_dmff02":
            L_mm = float(vals["L_mm"]); h_mm = float(vals["h_mm"]); w = float(vals["w_kNm"])
            L = L_mm / 1000.0

            Vmax = abs(w) * L / 2.0
            Mmax = abs(w) * (L**2) / 12.0

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_beam_vm_to_components("beam_", V_kN=Vmax, M_kNm=Mmax)

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = -abs(w) * L / 2.0
            _fill_beam_vm_to_components("col_", V_kN=0.0, M_kNm=0.0)
        # -----------------------------
        # DM-FP applies
        # -----------------------------
        elif cfg["apply"] == "apply_dmfp01":
            L_mm = float(vals["L_mm"]); h_mm = float(vals["h_mm"]); P = float(vals["F_kN"])
            L = L_mm / 1000.0

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_beam_vm_to_components("beam_", V_kN=abs(P), M_kNm=abs(P) * L / 4.0)

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = -abs(P) / 2.0
            _fill_beam_vm_to_components("col_", V_kN=abs(P) / 2.0, M_kNm=abs(P) * (h_mm / 1000.0))

        elif cfg["apply"] == "apply_dmfp02":
            L_mm = float(vals["L_mm"]); h_mm = float(vals["h_mm"]); P = float(vals["F_kN"])
            h = h_mm / 1000.0

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_beam_vm_to_components("beam_", V_kN=0.0, M_kNm=abs(P) * h)

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = 0.0
            _fill_beam_vm_to_components("col_", V_kN=abs(P), M_kNm=abs(P) * h)

        elif cfg["apply"] == "apply_dmfp03":
            L_mm = float(vals["L_mm"]); h_mm = float(vals["h_mm"]); w = float(vals["w_kNm"])
            L = L_mm / 1000.0

            Vmax = abs(w) * L / 2.0
            Mmax = abs(w) * (L**2) / 8.0

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_beam_vm_to_components("beam_", V_kN=Vmax, M_kNm=Mmax)

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = -abs(w) * L / 2.0
            _fill_beam_vm_to_components("col_", V_kN=0.0, M_kNm=0.0)

        elif cfg["apply"] == "apply_dmfp04":
            L_mm = float(vals["L_mm"]); h_mm = float(vals["h_mm"]); w = float(vals["w_kNm"])
            h = h_mm / 1000.0

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_beam_vm_to_components("beam_", V_kN=0.0, M_kNm=abs(w) * h**2 / 2.0)

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = 0.0
            _fill_beam_vm_to_components("col_", V_kN=abs(w) * h, M_kNm=abs(w) * h**2 / 2.0)

        # -----------------------------
        # DM-FR applies
        # -----------------------------
        elif cfg["apply"] == "apply_dmfr01":
            L_mm = float(vals["L_mm"]); h_mm = float(vals["h_mm"]); P = float(vals["F_kN"])
            L = L_mm / 1000.0

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_beam_vm_to_components("beam_", V_kN=abs(P), M_kNm=abs(P) * L)

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = -abs(P)
            _fill_beam_vm_to_components("col_", V_kN=0.0, M_kNm=abs(P) * L)

        elif cfg["apply"] == "apply_dmfr02":
            L_mm = float(vals["L_mm"]); h_mm = float(vals["h_mm"]); P = float(vals["F_kN"])
            h = h_mm / 1000.0

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_beam_vm_to_components("beam_", V_kN=0.0, M_kNm=abs(P) * h)

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = 0.0
            _fill_beam_vm_to_components("col_", V_kN=abs(P), M_kNm=abs(P) * h)

        elif cfg["apply"] == "apply_dmfr03":
            L_mm = float(vals["L_mm"]); h_mm = float(vals["h_mm"]); M = float(vals["M_kNm"])

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_beam_vm_to_components("beam_", V_kN=0.0, M_kNm=abs(M))

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = 0.0
            _fill_beam_vm_to_components("col_", V_kN=0.0, M_kNm=abs(M))

        elif cfg["apply"] == "apply_dmfr04":
            L_mm = float(vals["L_mm"]); h_mm = float(vals["h_mm"]); w = float(vals["w_kNm"])
            L = L_mm / 1000.0

            Vmax = abs(w) * L
            Mmax = abs(w) * (L**2) / 2.0

            st.session_state["beam_L_mm_in"] = L_mm
            st.session_state["beam_N_in"] = 0.0
            _fill_beam_vm_to_components("beam_", V_kN=Vmax, M_kNm=Mmax)

            st.session_state["col_L_mm_in"] = h_mm
            st.session_state["col_N_in"] = -abs(w) * L
            _fill_beam_vm_to_components("col_", V_kN=0.0, M_kNm=Mmax)


        else:
            _apply_ready_frame_case(case)

        for k in ["beam_df_rows", "beam_overall_ok", "beam_governing", "beam_extras",
                  "col_df_rows", "col_overall_ok", "col_governing", "col_extras"]:
            st.session_state.pop(k, None)

        st.toast("Case applied — loads transferred to inputs.", icon="✅")


def _swap_yz_in_sr_display(sr_display: dict) -> dict:
    """Return a shallow-copied sr_display with y/z (strong/weak) properties swapped.
    Use when the user selects 'Weak axis (zz)' but the load component is still entered as My / Vz etc.
    Your DB convention: strong axis = yy, weak axis = zz.
    """
    if not isinstance(sr_display, dict):
        return sr_display
    sr = dict(sr_display)  # shallow copy

    swap_pairs = [
        ("Iy_cm4", "Iz_cm4"),
        ("I_y_cm4", "I_z_cm4"),
        ("iy_mm", "iz_mm"),
        ("Wel_y_cm3", "Wel_z_cm3"),
        ("Wpl_y_cm3", "Wpl_z_cm3"),
        ("Av_y_mm2", "Av_z_mm2"),
        ("Avy_mm2", "Avz_mm2"),
        ("alpha_y", "alpha_z"),
        ("buckling_curve_y", "buckling_curve_z"),
        # precomputed resistances if present
        ("Mel_Rd_y_kNm", "Mel_Rd_z_kNm"),
        ("Mpl_Rd_y_kNm", "Mpl_Rd_z_kNm"),
        ("Vpl_Rd_y_kN", "Vpl_Rd_z_kN"),
    ]

    for a, b in swap_pairs:
        if a in sr or b in sr:
            sr[a], sr[b] = sr.get(b), sr.get(a)

    # Some families may store alternate spellings
    # Keep other keys untouched (torsion/warping etc.).
    return sr


def _apply_member_bending_axis(member_prefix: str, sr_display: dict) -> dict:
    """Apply the member's 'Bending axis in frame plane' setting to sr_display."""
    sel = st.session_state.get(f"{member_prefix}bend_axis_sel", "Strong axis (yy)")
    if isinstance(sel, str) and sel.lower().startswith("weak"):
        return _swap_yz_in_sr_display(sr_display)
    return sr_display

def _run_member_checks(member_prefix: str, inputs_key: str, out_prefix: str):
    sr_display_raw = st.session_state.get(f"{member_prefix}sr_display", None)
    sr_display = _apply_member_bending_axis(member_prefix, sr_display_raw)
    material = st.session_state.get(f"{member_prefix}material", st.session_state.get(f"{member_prefix}mat_sel", "S355"))
    fy, fu = get_material_props(material)

    if sr_display is None:
        st.warning("No section selected yet.")
        return

    inputs = st.session_state.get(inputs_key, {})
    torsion_supported = supports_torsion_and_warping(sr_display.get("family", ""))

    calc_section_classes_ec3(sr_display, fy)

    df_rows, overall_ok, governing, extras = compute_checks(sr_display, fy, inputs, torsion_supported)

    st.session_state[f"{out_prefix}df_rows"] = df_rows
    st.session_state[f"{out_prefix}overall_ok"] = overall_ok
    st.session_state[f"{out_prefix}governing"] = governing
    st.session_state[f"{out_prefix}extras"] = extras

    # REMOVE THIS LINE (it causes duplicate tables):
    # render_results(df_rows, overall_ok, governing, show_footer=True)

def _render_report_member(member_prefix: str, inputs_key: str, title: str):
    """
    Reuse render_report_tab() for beam/column by temporarily mapping
    member-specific session_state keys to the global ones that report expects.
    """
    st.markdown(f"## {title}")

    out_prefix = member_prefix  # "beam_" or "col_"

    # Snapshot old globals
    old = {
        "sr_display":  st.session_state.get("sr_display"),
        "inputs":      st.session_state.get("inputs"),
        "material":    st.session_state.get("material"),
        "mat_sel":     st.session_state.get("mat_sel"),
        "df_rows":     st.session_state.get("df_rows"),
        "overall_ok":  st.session_state.get("overall_ok"),
        "governing":   st.session_state.get("governing"),
        "extras":      st.session_state.get("extras"),
    }

    try:
        _sr_raw = st.session_state.get(f"{member_prefix}sr_display")
        st.session_state["sr_display"] = _apply_member_bending_axis(member_prefix, _sr_raw)

        st.session_state["inputs"] = st.session_state.get(inputs_key, {})

        st.session_state["material"] = st.session_state.get(f"{member_prefix}material")
        st.session_state["mat_sel"]  = st.session_state.get(f"{member_prefix}mat_sel", "S355")

        # ---- THIS WAS MISSING IN YOUR CODE ----
        st.session_state["df_rows"]    = st.session_state.get(f"{out_prefix}df_rows")
        st.session_state["overall_ok"] = st.session_state.get(f"{out_prefix}overall_ok")
        st.session_state["governing"]  = st.session_state.get(f"{out_prefix}governing")
        st.session_state["extras"]     = st.session_state.get(f"{out_prefix}extras") or {}
        # --------------------------------------

        render_report_tab(key_prefix=f"{member_prefix}rpt")

    finally:
        # Restore
        for k, v in old.items():
            if v is None:
                st.session_state.pop(k, None)
            else:
                st.session_state[k] = v

def _render_member_load_form(member_prefix: str, title: str, family_for_torsion: str, read_only: bool):
    """Loads form (ULS) — isolated by prefix.
    IMPORTANT: Do NOT pass value=... when using session_state keys (avoids Streamlit conflict).
    NOTE: No torsion in this app (Tx removed completely).
    """
    st.markdown(f"#### {title}")
    st.caption("Positive N = tension.")

    dis = bool(read_only)

    # Length label: Column uses h, Beam uses L (but keep key as .L_mm_in)
    len_label = "h (mm)" if member_prefix == "col_" else "L (mm)"

    # Initialize defaults ONLY if missing
    st.session_state.setdefault(f"{member_prefix}L_mm_in", 3000.0)
    st.session_state.setdefault(f"{member_prefix}N_in", 0.0)
    st.session_state.setdefault(f"{member_prefix}Vy_in", 0.0)
    st.session_state.setdefault(f"{member_prefix}Vz_in", 0.0)
    st.session_state.setdefault(f"{member_prefix}My_in", 0.0)
    st.session_state.setdefault(f"{member_prefix}Mz_in", 0.0)

    st.session_state.setdefault(f"{member_prefix}Ky_in", 1.0)
    st.session_state.setdefault(f"{member_prefix}Kz_in", 1.0)
    st.session_state.setdefault(f"{member_prefix}KLT_in", 1.0)
    st.session_state.setdefault(f"{member_prefix}KT_in", 1.0)

    # --- main inputs ---
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        st.number_input(
            len_label, min_value=1.0,
            step=100.0, disabled=dis, key=f"{member_prefix}L_mm_in"
        )
        st.number_input(
            "N (kN)",
            step=1.0, disabled=dis, key=f"{member_prefix}N_in"
        )

    with r1c2:
        st.number_input(
            "Vy (kN)",
            step=1.0, disabled=dis, key=f"{member_prefix}Vy_in"
        )
        st.number_input(
            "Vz (kN)",
            step=1.0, disabled=dis, key=f"{member_prefix}Vz_in"
        )

    with r1c3:
        st.number_input(
            "My (kN·m)",
            step=1.0, disabled=dis, key=f"{member_prefix}My_in"
        )
        st.number_input(
            "Mz (kN·m)",
            step=1.0, disabled=dis, key=f"{member_prefix}Mz_in"
        )

    # --- buckling factors ---
    st.markdown("##### Buckling / stability factors")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.number_input("Ky", step=0.05, disabled=dis, key=f"{member_prefix}Ky_in")
    with b2:
        st.number_input("Kz", step=0.05, disabled=dis, key=f"{member_prefix}Kz_in")
    with b3:
        st.number_input("KLT", step=0.05, disabled=dis, key=f"{member_prefix}KLT_in")
    with b4:
        st.number_input("KT", step=0.05, disabled=dis, key=f"{member_prefix}KT_in")

    # Keep stable key so downstream never crashes (Tx not used)
    st.session_state.setdefault(f"{member_prefix}Tx_in", 0.0)

def _render_instability_length_ratios_member(member_prefix: str, title: str):
    st.markdown(f"##### {title}")

    # Beam ratios are relative to span L, Column ratios are relative to height h
    is_col = (member_prefix == "col_")
    base_sym = "h" if is_col else "L"

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.number_input(
            f"Lcr,y / {base_sym} (flexural y–y)",
            min_value=0.05,
            value=float(st.session_state.get(f"{member_prefix}Lcr_y_ratio", 1.0)),
            step=0.05,
            key=f"{member_prefix}Lcr_y_ratio",
            help=f"Effective buckling length ratio about strong axis y–y. 1.0 means Lcr,y = {base_sym}."
        )
    with c2:
        st.number_input(
            f"Lcr,z / {base_sym} (flexural z–z)",
            min_value=0.05,
            value=float(st.session_state.get(f"{member_prefix}Lcr_z_ratio", 1.0)),
            step=0.05,
            key=f"{member_prefix}Lcr_z_ratio",
            help=f"Effective buckling length ratio about weak axis z–z. 1.0 means Lcr,z = {base_sym}."
        )
    with c3:
        st.number_input(
            f"L_LT / {base_sym} (lateral–torsional)",
            min_value=0.05,
            value=float(st.session_state.get(f"{member_prefix}L_LT_ratio", 1.0)),
            step=0.05,
            key=f"{member_prefix}L_LT_ratio",
            help=f"Effective lateral–torsional buckling length ratio. 1.0 means L_LT = {base_sym}."
        )
    with c4:
        st.number_input(
            f"L_TF / {base_sym} (torsional / flexural–torsional)",
            min_value=0.05,
            value=float(st.session_state.get(f"{member_prefix}L_TF_ratio", 1.0)),
            step=0.05,
            key=f"{member_prefix}L_TF_ratio",
            help=f"Effective torsional / flexural–torsional buckling length ratio. 1.0 means L_TF = {base_sym}."
        )

def _render_member_load_form(member_prefix: str, title: str, family_for_torsion: str, read_only: bool = False):
    """
    Renders ONLY member design forces (ULS) in a 3-columns-per-row grid,
    like the Beam app. Buckling ratios are rendered separately in the expander.
    """
    dis = bool(read_only)

    st.subheader(title)
    st.caption("Positive N = tension.")

    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        len_label = "h (mm)" if member_prefix == "col_" else "L (mm)"
        st.number_input(len_label,
                        value=float(st.session_state.get(f"{member_prefix}L_mm_in", 3000.0)),
                        step=100.0, disabled=dis, key=f"{member_prefix}L_mm_in")

        st.number_input("N (kN)", value=float(st.session_state.get(f"{member_prefix}N_in", 0.0)),
                        step=1.0, disabled=dis, key=f"{member_prefix}N_in")

    with r1c2:
        st.number_input("Vy (kN)", value=float(st.session_state.get(f"{member_prefix}Vy_in", 0.0)),
                        step=1.0, disabled=dis, key=f"{member_prefix}Vy_in")
        st.number_input("Vz (kN)", value=float(st.session_state.get(f"{member_prefix}Vz_in", 0.0)),
                        step=1.0, disabled=dis, key=f"{member_prefix}Vz_in")

    with r1c3:
        st.number_input("My (kN·m)", value=float(st.session_state.get(f"{member_prefix}My_in", 0.0)),
                        step=1.0, disabled=dis, key=f"{member_prefix}My_in")
        st.number_input("Mz (kN·m)", value=float(st.session_state.get(f"{member_prefix}Mz_in", 0.0)),
                        step=1.0, disabled=dis, key=f"{member_prefix}Mz_in")

    # Beam torsion only (and only if section supports it)
    torsion_ok = supports_torsion_and_warping(family_for_torsion or "")
    if member_prefix == "beam_" and torsion_ok:
        st.number_input("Tx (kN·m)", value=float(st.session_state.get(f"{member_prefix}Tx_in", 0.0)),
                        step=0.5, disabled=dis, key=f"{member_prefix}Tx_in")
    else:
        # Keep stable key so downstream never crashes
        if f"{member_prefix}Tx_in" not in st.session_state:
            st.session_state[f"{member_prefix}Tx_in"] = 0.0

# ----------------------------
# MAIN TABS (Frame app)
# ----------------------------
tab_p, tab_b, tab_c, tab_l, tab_br, tab_cr, tab_brep, tab_crep = st.tabs(
    [
        "Project info",
        "Beam",
        "Column",
        "Loads",
        "Beam results",
        "Column results",
        "Beam report",
        "Column report",
    ]
)

with tab_p:
    render_project_data()

with tab_b:
    _render_member_section_panel("beam_", "Beam section selection")

with tab_c:
    _render_member_section_panel("col_", "Column section selection")

with tab_l:
    # -----------------------------
    # Loads settings — TOP (like beam app)
    # -----------------------------
    st.markdown("### Loads settings")
    _render_design_settings()

    # Auto-apply factoring when user changes γF or Characteristic/Design mode
    prev_g = st.session_state.get("_prev_gamma_F_frame", None)
    prev_t = st.session_state.get("_prev_manual_forces_type_frame", None)
    
    cur_g = float(st.session_state.get("gamma_F", 1.35))
    cur_t = str(st.session_state.get("manual_forces_type", "Characteristic"))
    
    # First run: initialize only (DO NOT clear results)
    if (prev_g is None) or (prev_t is None):
        st.session_state["_prev_gamma_F_frame"] = cur_g
        st.session_state["_prev_manual_forces_type_frame"] = cur_t
    
    else:
        changed = (abs(prev_g - cur_g) > 1e-9) or (prev_t != cur_t)
        if changed:
            st.session_state["_prev_gamma_F_frame"] = cur_g
            st.session_state["_prev_manual_forces_type_frame"] = cur_t
    
            _store_design_forces_from_state_member("beam_", "beam_inputs")
            _store_design_forces_from_state_member("col_",  "col_inputs")

    # -----------------------------
    # Instability length ratios (expander)
    # -----------------------------
    with st.expander("Instability length ratios (Beam: relative to L, Column: relative to h)", expanded=False):
        _render_instability_length_ratios_member("beam_", "Beam")
        st.markdown("---")
        _render_instability_length_ratios_member("col_", "Column")

    st.markdown("---")

    # -----------------------------
    # Ready frame cases gallery
    # -----------------------------
    _render_ready_frame_cases()

    st.markdown("---")

    # -----------------------------
    # Beam + Column forces (editable)
    # -----------------------------
    cL, cR = st.columns(2)
    beam_sr = st.session_state.get("beam_sr_display", {}) or {}
    col_sr  = st.session_state.get("col_sr_display", {}) or {}

    with cL:
        _render_member_load_form("beam_", "Beam design forces (ULS)", beam_sr.get("family", ""), read_only=False)
    with cR:
        _render_member_load_form("col_", "Column design forces (ULS)", col_sr.get("family", ""), read_only=False)

# ----------------------------
# Beam results tab
# ----------------------------
with tab_br:
    run_col, _ = st.columns([1, 3])
    with run_col:
        if st.button("Run beam check", key="run_check_beam"):
            _store_design_forces_from_state_member("beam_", "beam_inputs")
            try:
                _run_member_checks("beam_", "beam_inputs", "beam_")
            except Exception as e:
                st.error(f"Beam computation error: {e}")

    df_rows = st.session_state.get("beam_df_rows", None)
    if df_rows is None:
        st.info("Set up **Loads** and select a **Beam** section, then press **Run beam check**.")
    else:
        old_extras = st.session_state.get("extras", None)
        try:
            # IMPORTANT: render_results reads st.session_state["extras"]
            st.session_state["extras"] = st.session_state.get("beam_extras") or {}

            render_results(
                df_rows,
                st.session_state.get("beam_overall_ok"),
                st.session_state.get("beam_governing"),
                show_footer=True,
                key_prefix="beam_res_",
            )
        finally:
            if old_extras is None:
                st.session_state.pop("extras", None)
            else:
                st.session_state["extras"] = old_extras


# ----------------------------
# Column results tab
# ----------------------------
with tab_cr:
    run_col, _ = st.columns([1, 3])
    with run_col:
        if st.button("Run column check", key="run_check_col"):
            _store_design_forces_from_state_member("col_", "col_inputs")
            try:
                _run_member_checks("col_", "col_inputs", "col_")
            except Exception as e:
                st.error(f"Column computation error: {e}")

    df_rows = st.session_state.get("col_df_rows", None)
    if df_rows is None:
        st.info("Set up **Loads** and select a **Column** section, then press **Run column check**.")
    else:
        old_extras = st.session_state.get("extras", None)
        try:
            # IMPORTANT: render_results reads st.session_state["extras"]
            st.session_state["extras"] = st.session_state.get("col_extras") or {}

            render_results(
                df_rows,
                st.session_state.get("col_overall_ok"),
                st.session_state.get("col_governing"),
                show_footer=True,
                key_prefix="col_res_",
            )
        finally:
            if old_extras is None:
                st.session_state.pop("extras", None)
            else:
                st.session_state["extras"] = old_extras


# ----------------------------
# Beam report tab
# ----------------------------
with tab_brep:
    if not st.session_state.get("beam_sr_display"):
        st.info("Select a **Beam** section first.")
    elif st.session_state.get("beam_df_rows", None) is None:
        st.info("Run **Beam results** first, then open **Beam report**.")
    else:
        _render_report_member("beam_", "beam_inputs", "Beam report")


# ----------------------------
# Column report tab
# ----------------------------
with tab_crep:
    if not st.session_state.get("col_sr_display"):
        st.info("Select a **Column** section first.")
    elif st.session_state.get("col_df_rows", None) is None:
        st.info("Run **Column results** first, then open **Column report**.")
    else:
        _render_report_member("col_", "col_inputs", "Column report")
