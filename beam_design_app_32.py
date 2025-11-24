# beam_design_app.py
# EngiSnap — Standard steel beam checks (DB-backed, compact UI)

import streamlit as st
import pandas as pd
import math
from datetime import datetime, date
import traceback
import re
import numbers
import numpy as np

# -------------------------
# Optional: install psycopg2-binary:
# pip install psycopg2-binary
# -------------------------
try:
    import psycopg2
    HAS_PG = True
except Exception:
    psycopg2 = None
    HAS_PG = False

# -------------------------
# get_conn() using st.secrets (UNCHANGED)
# -------------------------
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


# -------------------------
# SQL runner
# -------------------------
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
# Fallback sample rows
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

# -------------------------
# DB helpers (unchanged)
# -------------------------
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

# -------------------------
# Robust numeric parser & pick helper
# -------------------------
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

# -------------------------
# Helpers
# -------------------------
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

# -------------------------
# Ready cases (RESTORED)
# -------------------------
def ss_udl(span_m: float, w_kN_per_m: float):
    Mmax = w_kN_per_m * span_m**2 / 8.0
    Vmax = w_kN_per_m * span_m / 2.0
    return (0.0, float(Mmax), 0.0, float(Vmax), 0.0)

def ss_point_center(span_m: float, P_kN: float):
    Mmax = P_kN * span_m / 4.0
    Vmax = P_kN / 2.0
    return (0.0, float(Mmax), 0.0, float(Vmax), 0.0)

def ss_point_at_a(span_m: float, P_kN: float, a_m: float):
    Mmax = P_kN * a_m * (span_m - a_m) / span_m
    Vmax = P_kN
    return (0.0, float(Mmax), 0.0, float(Vmax), 0.0)

READY_CATALOG = {
    "Beam": {
        "Simply supported (examples)": {
            "SS-UDL": {"label": "SS-01: UDL (w on L)", "inputs": {"L": 6.0, "w": 10.0}, "func": ss_udl},
            "SS-Point-Centre": {"label": "SS-02: Point at midspan (P)", "inputs": {"L": 6.0, "P": 20.0}, "func": ss_point_center},
            "SS-Point-a": {"label": "SS-03: Point at distance a (P at a)", "inputs": {"L": 6.0, "P": 20.0, "a": 2.0}, "func": ss_point_at_a},
        }
    },
    "Frame": {
        "Simple frame examples": {
            "F-01": {"label": "FR-01: Simple 2-member frame (placeholder)",
                     "inputs": {"L": 4.0, "P": 5.0},
                     "func": lambda L, P: (0.0, float(P*L/4.0), 0.0, float(P/2.0), 0.0)},
        }
    }
}

def render_ready_cases_panel():
    with st.expander("Ready beam & frame cases (optional)", expanded=False):
        st.write("Pick a ready case to auto-fill typical max forces/moments.")
        use_ready = st.checkbox("Use ready case", key="ready_use_case")

        if not use_ready:
            return

        chosen_type = st.selectbox("Type", options=["-- choose --", "Beam", "Frame"], key="ready_type")
        if chosen_type == "-- choose --":
            st.info("Select Beam or Frame to show categories.")
            return

        categories = sorted(READY_CATALOG.get(chosen_type, {}).keys())
        chosen_cat = st.selectbox("Category", options=["-- choose --"] + categories, key="ready_category")
        if chosen_cat == "-- choose --":
            return

        cases_dict = READY_CATALOG[chosen_type][chosen_cat]
        case_keys = list(cases_dict.keys())

        cols = st.columns(3)
        selected_case_key = None
        for i, ck in enumerate(case_keys):
            col = cols[i % 3]
            lbl = cases_dict[ck]["label"]
            col.markdown(
                f"<div style='border:2px solid #bbb;border-radius:10px;padding:12px;text-align:center;background:#fbfbfb;margin-bottom:6px;min-height:68px;display:flex;align-items:center;justify-content:center;font-weight:600;'>{lbl}</div>",
                unsafe_allow_html=True
            )
            if col.button(f"Select {ck}", key=f"ready_select_{chosen_type}_{chosen_cat}_{i}"):
                selected_case_key = ck

        if selected_case_key:
            st.session_state["ready_selected_type"] = chosen_type
            st.session_state["ready_selected_category"] = chosen_cat
            st.session_state["ready_selected_case"] = selected_case_key
            st.rerun()

        sel_case = st.session_state.get("ready_selected_case")
        sel_type = st.session_state.get("ready_selected_type")
        sel_cat = st.session_state.get("ready_selected_category")

        if sel_case and sel_type == chosen_type and sel_cat == chosen_cat:
            scase_info = READY_CATALOG[sel_type][sel_cat].get(sel_case)
            st.markdown(f"**Selected case:** {scase_info['label']}")
            inputs = scase_info.get("inputs", {})
            input_vals = {}
            for k, v in inputs.items():
                input_vals[k] = st.number_input(f"{k}", value=float(v), key=f"ready_input_{sel_case}_{k}")

            col_apply, col_clear = st.columns([1, 1])
            with col_apply:
                if st.button("Apply case to load inputs", key=f"ready_apply_{sel_case}"):
                    func = scase_info.get("func")
                    try:
                        args = [input_vals[k] for k in inputs.keys()]
                        N_case, My_case, Mz_case, Vy_case, Vz_case = func(*args)
                    except Exception:
                        N_case = My_case = Mz_case = Vy_case = Vz_case = 0.0

                    st.session_state["prefill_from_case"] = True
                    st.session_state["prefill_N_kN"] = float(N_case)
                    st.session_state["prefill_My_kNm"] = float(My_case)
                    st.session_state["prefill_Mz_kNm"] = float(Mz_case)
                    st.session_state["prefill_Vy_kN"] = float(Vy_case)
                    st.session_state["prefill_Vz_kN"] = float(Vz_case)
                    if "L" in input_vals:
                        st.session_state["case_L"] = float(input_vals["L"])

                    st.success("Case applied — open Loads form below and click Run check.")

            with col_clear:
                if st.button("Clear selected case", key=f"ready_clear_{sel_case}"):
                    for k in ("ready_selected_type","ready_selected_category","ready_selected_case",
                              "prefill_from_case","prefill_N_kN","prefill_My_kNm",
                              "prefill_Mz_kNm","prefill_Vy_kN","prefill_Vz_kN","case_L"):
                        st.session_state.pop(k, None)
                    st.success("Selected case cleared.")
                    st.rerun()

# -------------------------
# UI renderers
# -------------------------
def render_sidebar_guidelines():
    st.sidebar.title("Guidelines")
    st.sidebar.markdown("""
1. Member & Section  
2. Loads  
3. Run check  
4. Results  
5. Report  

DB sections read-only.
""")

def render_project_data():
    with st.expander("Project data", expanded=False):
        meta_col1, meta_col2, meta_col3 = st.columns([1,1,1])
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

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        material = st.selectbox("Material", ["S235","S275","S355"], index=2, key="mat_sel")
    with c2:
        family = st.selectbox("Section family / Type (DB)", ["-- choose --"] + types if types else ["-- choose --"], key="fam_sel")
    with c3:
        selected_name = None
        selected_row = None
        if family and family != "-- choose --":
            names = sizes_map.get(family, [])
            selected_name = st.selectbox("Section size (DB)", ["-- choose --"] + names if names else ["-- choose --"], key="size_sel")
            if selected_name and selected_name != "-- choose --":
                selected_row = get_section_row_db(family, selected_name, detected_table if detected_table != "sample" else None)

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

# ✅ FIX: add key_prefix so same UI can be rendered in multiple tabs safely
def render_section_summary_like_props(material, sr_display, key_prefix="sum"):
    fy = material_to_fy(material)
    st.markdown("### Selected section summary")

    s1, s2, s3 = st.columns(3)
    s1.text_input("Material", value=material, disabled=True, key=f"{key_prefix}_mat")
    s2.text_input("fy (MPa)", value=f"{fy:.0f}", disabled=True, key=f"{key_prefix}_fy")
    s3.text_input("Family / Type", value=str(sr_display.get("family","")), disabled=True, key=f"{key_prefix}_fam")

    s4, s5, s6 = st.columns(3)
    s4.text_input("Size", value=str(sr_display.get("name","")), disabled=True, key=f"{key_prefix}_size")
    s5.number_input("A (cm²)", value=float(sr_display.get("A_cm2",0.0)), disabled=True, key=f"{key_prefix}_A")
    s6.number_input("c_max (mm)", value=float(sr_display.get("c_max_mm",0.0)), disabled=True, key=f"{key_prefix}_c")

    s7, s8, s9 = st.columns(3)
    s7.number_input("Iy (cm⁴)", value=float(sr_display.get("I_y_cm4",0.0)), disabled=True, key=f"{key_prefix}_Iy")
    s8.number_input("Iz (cm⁴)", value=float(sr_display.get("I_z_cm4",0.0)), disabled=True, key=f"{key_prefix}_Iz")
    s9.number_input("J (cm⁴)", value=float(sr_display.get("J_cm4",0.0)), disabled=True, key=f"{key_prefix}_J")

    s10, s11, s12 = st.columns(3)
    s10.number_input("Wpl_y (cm³)", value=float(sr_display.get("Wpl_y_cm3",0.0)), disabled=True, key=f"{key_prefix}_Wply")
    s11.number_input("Wpl_z (cm³)", value=float(sr_display.get("Wpl_z_cm3",0.0)), disabled=True, key=f"{key_prefix}_Wplz")
    s12.text_input("Buckling α", value=str(sr_display.get("alpha_curve","n/a")), disabled=True, key=f"{key_prefix}_alpha")

    s13, s14, s15 = st.columns(3)
    s13.text_input("Flange class (DB)", value=str(sr_display.get("flange_class_db","n/a")), disabled=True, key=f"{key_prefix}_fc")
    s14.text_input("Web class (bending, DB)", value=str(sr_display.get("web_class_bending_db","n/a")), disabled=True, key=f"{key_prefix}_wb")
    s15.text_input("Web class (compression, DB)", value=str(sr_display.get("web_class_compression_db","n/a")), disabled=True, key=f"{key_prefix}_wc")

def render_section_properties_readonly(sr_display):
    c1, c2, c3 = st.columns(3)
    c1.number_input("A (cm²)", value=float(sr_display.get('A_cm2', 0.0)), disabled=True, key="db_A")
    c2.number_input("S_y (cm³) about y", value=float(sr_display.get('S_y_cm3', 0.0)), disabled=True, key="db_Sy")
    c3.number_input("S_z (cm³) about z", value=float(sr_display.get('S_z_cm3', 0.0)), disabled=True, key="db_Sz")

    c4, c5, c6 = st.columns(3)
    c4.number_input("I_y (cm⁴) about y", value=float(sr_display.get('I_y_cm4', 0.0)), disabled=True, key="db_Iy")
    c5.number_input("I_z (cm⁴) about z", value=float(sr_display.get('I_z_cm4', 0.0)), disabled=True, key="db_Iz")
    c6.number_input("J (cm⁴) (torsion const)", value=float(sr_display.get('J_cm4', 0.0)), disabled=True, key="db_J")

    c7, c8, c9 = st.columns(3)
    c7.number_input("c_max (mm)", value=float(sr_display.get('c_max_mm', 0.0)), disabled=True, key="db_c")
    c8.number_input("Wpl_y (cm³)", value=float(sr_display.get('Wpl_y_cm3', 0.0)), disabled=True, key="db_Wply")
    c9.number_input("Wpl_z (cm³)", value=float(sr_display.get('Wpl_z_cm3', 0.0)), disabled=True, key="db_Wplz")

    c10, c11, c12 = st.columns(3)
    c10.number_input("Iw (cm⁶) (warping)", value=float(sr_display.get("Iw_cm6", 0.0)), disabled=True, key="db_Iw")
    c11.number_input("It (cm⁴) (torsion inertia)", value=float(sr_display.get("It_cm4", 0.0)), disabled=True, key="db_It")
    c12.empty()

    cls1, cls2, cls3 = st.columns(3)
    cls1.text_input("Flange class (DB)", value=str(sr_display.get('flange_class_db', "n/a")), disabled=True, key="db_fc")
    cls2.text_input("Web class (bending, DB)", value=str(sr_display.get('web_class_bending_db', "n/a")), disabled=True, key="db_wc_b")
    cls3.text_input("Web class (compression, DB)", value=str(sr_display.get('web_class_compression_db', "n/a")), disabled=True, key="db_wc_c")

    a1, a2, a3 = st.columns(3)
    alpha_db_val = sr_display.get('alpha_curve', 0.49)
    alpha_label_db = next((lbl for lbl, val in [
        ("0.13 (a)",0.13),("0.21 (b)",0.21),("0.34 (c)",0.34),("0.49 (d)",0.49),("0.76 (e)",0.76)
    ] if abs(val - float(alpha_db_val)) < 1e-8), f"{alpha_db_val}")
    a1.text_input("Buckling α (DB)", value=str(alpha_label_db), disabled=True, key="db_alpha")
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
        return [color]*len(row)

    st.write(df_rows.style.apply(highlight_row, axis=1))






