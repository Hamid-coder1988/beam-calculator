# beam_design_app.py
# EngiSnap — Beam design Eurocode Checker (DB-backed, robust column mapping)
# SECURITY NOTE: use st.secrets["postgres"] for deployment (do NOT commit credentials).

import streamlit as st
import pandas as pd
import math
from io import BytesIO
from datetime import datetime, date
import psycopg2

st.set_page_config(page_title="EngiSnap Beam Design Eurocode Checker", layout="wide")
st.title("EngiSnap — Design of steel members (Eurocode)")

st.markdown("""
**Disclaimer (read carefully)**  
This prototype performs simplified Eurocode-style screening checks. It is **not** a full EN1993 design package.
Use for screening/prototyping only — always have final results verified by a licensed structural engineer.
""")

# -------------------------
# DB helper (use st.secrets["postgres"])
# -------------------------
@st.cache_resource(show_spinner=False)
def get_conn():
    pg = st.secrets.get("postgres", None)
    if not pg:
        raise RuntimeError('st.secrets["postgres"] not found. Add DB connection under st.secrets["postgres"].')
    conn = psycopg2.connect(
        host=pg["host"],
        port=str(pg["port"]),
        database=pg["database"],
        user=pg["user"],
        password=pg["password"]
    )
    return conn

@st.cache_resource(show_spinner=False)
def load_beam_db():
    # try common table names and fall back to sample if not available
    attempts = ["beam", "Beam", "beam_sections", "sections", "steel_sections"]
    for t in attempts:
        try:
            conn = get_conn()
            df = pd.read_sql(f'SELECT * FROM "{t}";', conn)
            st.sidebar.info(f"Detected table: {t}")
            return df
        except Exception:
            try:
                conn = get_conn()
                df = pd.read_sql(f"SELECT * FROM {t};", conn)
                st.sidebar.info(f"Detected table: {t}")
                return df
            except Exception:
                continue
    # final attempt: try reading first table name from information_schema
    try:
        conn = get_conn()
        q = """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type='BASE TABLE';
        """
        tables = pd.read_sql(q, conn)
        if not tables.empty:
            t0 = tables.iloc[0,0]
            try:
                df = pd.read_sql(f'SELECT * FROM "{t0}" LIMIT 1000;', conn)
                st.sidebar.info(f"Using table: {t0}")
                return df
            except Exception:
                pass
    except Exception:
        pass
    # fallback sample
    sample_rows = [
        {"family": "IPE", "name": "IPE 200",
         "A_cm2": 31.2, "S_y_cm3": 88.0, "S_z_cm3": 16.0,
         "I_y_cm4": 1760.0, "I_z_cm4": 64.0, "J_cm4": 14.0, "c_max_mm": 100.0,
         "Wpl_y_cm3": 96.8, "Wpl_z_cm3": 18.0, "alpha_curve": 0.49,
         "flange_class_db": "Class 1/2", "web_class_bending_db": "Class 2", "web_class_compression_db": "Class 3",
         "Iw_cm6": 2500.0, "It_cm4": 14.0
        },
        {"family": "RHS", "name": "RHS 150x100x6.3",
         "A_cm2": 25.0, "S_y_cm3": 60.0, "S_z_cm3": 40.0,
         "I_y_cm4": 900.0, "I_z_cm4": 450.0, "J_cm4": 10.0, "c_max_mm": 75.0,
         "Wpl_y_cm3": 48.0, "Wpl_z_cm3": 32.0, "alpha_curve": 0.34,
         "flange_class_db": "N/A (RHS)", "web_class_bending_db": "N/A (RHS)", "web_class_compression_db": "N/A (RHS)",
         "Iw_cm6": 0.0, "It_cm4": 0.0
        }
    ]
    st.sidebar.warning("Could not detect DB table — using internal sample.")
    return pd.DataFrame(sample_rows)

df_db = load_beam_db()

# -------------------------
# helper: try find a column from list of candidates
# -------------------------
def find_column(df, candidates):
    # return first candidate that exists in df.columns (case-insensitive matching)
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        cl = cand.lower()
        if cl in cols_lower:
            return cols_lower[cl]
    return None

# --------------------------------------------------------------------------------
# Sidebar UI: material, then section family/size mapping (robust)
# --------------------------------------------------------------------------------
st.sidebar.header("Material & section selection")

material = st.sidebar.selectbox("Material", ("S235", "S275", "S355"),
                                help="Select steel grade (typical EN names).")
fy = 235.0 if material=="S235" else (275.0 if material=="S275" else 355.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Auto-detect section family & size columns")

# candidate names
family_candidates = ["family", "type", "section_type", "category", "Family", "Type"]
name_candidates = ["name", "size", "section", "profile", "designation", "Name", "Size"]

family_col_auto = find_column(df_db, family_candidates)
name_col_auto = find_column(df_db, name_candidates)

st.sidebar.markdown("If detection is wrong, pick columns manually:")
family_col = st.sidebar.selectbox("Column for section family", options=["-- auto detect --"] + list(df_db.columns), index=0)
if family_col == "-- auto detect --":
    family_col = family_col_auto or (df_db.columns[0] if len(df_db.columns)>0 else None)

size_col = st.sidebar.selectbox("Column for section size / name", options=["-- auto detect --"] + list(df_db.columns), index=0)
if size_col == "-- auto detect --":
    size_col = name_col_auto or (df_db.columns[1] if len(df_db.columns)>1 else (df_db.columns[0] if len(df_db.columns)>0 else None))

# show mapping
st.sidebar.markdown(f"**Using family column:** `{family_col}`  — **Using size column:** `{size_col}`")

# -------------------------
# Safely get families & sizes preserving DB order
# -------------------------
families_list = []
if family_col and family_col in df_db.columns:
    families_list = df_db[family_col].dropna().astype(str).tolist()
    # preserve order as-is while removing duplicates (first occurrence kept)
    seen = set(); fams = []
    for v in families_list:
        if v not in seen:
            seen.add(v); fams.append(v)
    families_list = fams
else:
    # fallback to a single placeholder family
    families_list = ["IPE"]

family = st.sidebar.selectbox("Section family", options=["-- choose --"] + families_list)

selected_row = None
selected_name = None
if family and family != "-- choose --":
    # filter rows preserving original order
    df_f = df_db[df_db[family_col].astype(str) == str(family)]
    # list of sizes in db order
    if size_col and size_col in df_db.columns:
        names = df_f[size_col].dropna().astype(str).tolist()
    else:
        # if no size_col, use index as a fallback
        names = df_f.index.astype(str).tolist()
    # remove duplicates, keep order
    seen = set(); names_u = []
    for n in names:
        if n not in seen:
            seen.add(n); names_u.append(n)
    selected_name = st.sidebar.selectbox("Section size", options=["-- choose --"] + names_u)
    if selected_name and selected_name != "-- choose --":
        # pick first matching row (keep original row order)
        df_sel = df_f[df_f[size_col].astype(str) == str(selected_name)]
        if df_sel.empty:
            # fallback: try exact match ignoring leading/trailing spaces
            df_sel = df_f[df_f[size_col].astype(str).str.strip() == str(selected_name).strip()]
        if not df_sel.empty:
            selected_row = df_sel.iloc[0].to_dict()
            st.sidebar.success(f"Loaded: {selected_name}")
        else:
            st.sidebar.error("Could not find selected row — check mapping.")

# Buckling factors after section selection
st.sidebar.markdown("---")
st.sidebar.markdown("Buckling effective length factors (K):")
K_z = st.sidebar.number_input("K_z — flexural buckling about z–z", value=1.0, min_value=0.1, step=0.05)
K_y = st.sidebar.number_input("K_y — flexural buckling about y–y", value=1.0, min_value=0.1, step=0.05)
K_LT = st.sidebar.number_input("K_LT — lateral–torsional buckling", value=1.0, min_value=0.1, step=0.05)
K_T = st.sidebar.number_input("K_T — torsional buckling (reserved)", value=1.0, min_value=0.1, step=0.05)

alpha_default_val = 0.49

# -------------------------
# Project metadata top
# -------------------------
st.markdown("## Project data")
meta_col1, meta_col2, meta_col3 = st.columns([1,1,1])
with meta_col1:
    doc_name = st.text_input("Document title", value="Beam check")
    project_name = st.text_input("Project name", value="")
with meta_col2:
    position = st.text_input("Position / Location (Beam ID)", value="")
    requested_by = st.text_input("Requested by", value="")
with meta_col3:
    revision = st.text_input("Revision", value="0.1")
    run_date = st.date_input("Date", value=date.today())

st.markdown("---")

# -------------------------
# Cross-section schematic
# -------------------------
st.markdown("### Cross-section schematic")
center_cols = st.columns([1,2,1])
center_cols[1].markdown(
    """
<div style="border:2px dashed #999;border-radius:6px;width:100%;max-width:420px;height:220px;display:flex;align-items:center;justify-content:center;background:#fafafa;color:#666;font-weight:600;margin:8px auto;">
    Cross-section image placeholder
</div>
""", unsafe_allow_html=True)

# -------------------------
# Section properties area
# -------------------------
st.markdown("---")
st.header("Section properties")
use_custom = st.checkbox("Use custom section (enter resistances & buckling/classes only)", help="When checked, you enter resistances and classes manually — no profile dims required.")

# small helpers
def pick(d, *keys, default=None):
    if d is None:
        return default
    for k in keys:
        if k in d and pd.notnull(d[k]):
            return d[k]
    # case-insensitive
    lk = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        kl = str(k).lower()
        if kl in lk and pd.notnull(lk[kl]):
            return lk[kl]
    return default

def to_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def box_html(label, value, unit=""):
    return f"""
    <div style="border:1px solid #ddd;border-radius:8px;padding:10px;min-height:64px;display:flex;flex-direction:column;justify-content:center;background:#fbfbfb;">
      <div style="font-size:13px;color:#444;font-weight:600">{label}</div>
      <div style="font-size:15px;color:#111">{value} {unit}</div>
    </div>
    """

selected_row_mapped = None
if selected_row is not None and not use_custom:
    sr = selected_row
    # map many possible names robustly
    selected_row_mapped = {
        "family": pick(sr, "family", "type", "section_type"),
        "name": pick(sr, "name", "size", "profile", "designation"),
        "A_cm2": to_float(pick(sr, "A_cm2", "area_cm2", "area", "A", "Area"), 0.0),
        "I_y_cm4": to_float(pick(sr, "I_y_cm4", "Iy", "I_y", "Iyy"), 0.0),
        "I_z_cm4": to_float(pick(sr, "I_z_cm4", "Iz", "I_z", "Izz"), 0.0),
        "S_y_cm3": to_float(pick(sr, "S_y_cm3", "Sy", "S_y"), 0.0),
        "S_z_cm3": to_float(pick(sr, "S_z_cm3", "Sz", "S_z"), 0.0),
        "Wpl_y_cm3": to_float(pick(sr, "Wpl_y_cm3", "Wpl_y", "W_pl_y"), 0.0),
        "Wpl_z_cm3": to_float(pick(sr, "Wpl_z_cm3", "Wpl_z", "W_pl_z"), 0.0),
        "J_cm4": to_float(pick(sr, "J_cm4", "J"), 0.0),
        "Iw_cm6": to_float(pick(sr, "Iw_cm6", "Iw"), 0.0),
        "It_cm4": to_float(pick(sr, "It_cm4", "It"), 0.0),
        "c_max_mm": to_float(pick(sr, "c_max_mm", "c_max", "c"), 0.0),
        "alpha_curve": pick(sr, "alpha_curve", "alpha"),
        "flange_class_db": pick(sr, "flange_class_db", "flange_class", "flangeClass"),
        "web_class_bending_db": pick(sr, "web_class_bending_db", "web_class_bending"),
        "web_class_compression_db": pick(sr, "web_class_compression_db", "web_class_compression"),
        # possible DB resistances (optional)
        "N_Rd_kN_db": pick(sr, "N_Rd_kN", "N_Rd", "Nrd_kN"),
        "V_Rd_kN_db": pick(sr, "V_Rd_kN", "V_Rd", "Vrd_kN"),
        "M_Rd_y_kNm_db": pick(sr, "M_Rd_y_kNm", "M_Rd_y", "Mrd_y_kNm")
    }

# UI: display mapped DB properties or custom inputs
if use_custom:
    st.markdown("### Custom resistances & buckling/classes")
    c1, c2, c3 = st.columns(3)
    N_Rd_kN_custom = c1.number_input("N_Rd (kN) — axial resistance", value=0.0, key="custom_Nrd")
    V_Rd_kN_custom = c2.number_input("V_Rd (kN) — shear resistance", value=0.0, key="custom_Vrd")
    M_Rd_y_kNm_custom = c3.number_input("M_Rd,y (kN·m)", value=0.0, key="custom_Mrd_y")
    b1, b2, b3 = st.columns(3)
    alpha_choice = b1.selectbox("Buckling curve α (custom)", ["0.13 (a)","0.21 (b)","0.34 (c)","0.49 (d)","0.76 (e)"], index=3)
    alpha_map = {"0.13 (a)":0.13,"0.21 (b)":0.21,"0.34 (c)":0.34,"0.49 (d)":0.49,"0.76 (e)":0.76}
    alpha_val_custom = alpha_map[alpha_choice]
    flange_class_custom = b2.selectbox("Flange class", ["Auto","Class 1","Class 2","Class 3","Class 4"], index=0)
    web_class_bending_custom = b3.selectbox("Web class (bending)", ["Auto","Class 1","Class 2","Class 3","Class 4"], index=0)
    web_class_compression_custom = st.selectbox("Web class (compression)", ["Auto","Class 1","Class 2","Class 3","Class 4"], index=0)
    use_props = {
        "family":"CUSTOM","name":"CUSTOM",
        "A_cm2":0.0,"I_y_cm4":0.0,"I_z_cm4":0.0,"S_y_cm3":0.0,"S_z_cm3":0.0,
        "J_cm4":0.0,"Iw_cm6":0.0,"It_cm4":0.0,"c_max_mm":0.0,
        "N_Rd_kN_db": N_Rd_kN_custom if N_Rd_kN_custom>0 else None,
        "V_Rd_kN_db": V_Rd_kN_custom if V_Rd_kN_custom>0 else None,
        "M_Rd_y_kNm_db": M_Rd_y_kNm_custom if M_Rd_y_kNm_custom>0 else None,
        "alpha_curve": alpha_val_custom, "flange_class_db": flange_class_custom,
        "web_class_bending_db": web_class_bending_custom, "web_class_compression_db": web_class_compression_custom
    }
else:
    st.markdown("### Section properties (from DB — read only)")
    if selected_row_mapped is None:
        st.info("Select a section family & size in the sidebar (or enable custom).")
        use_props = {
            "family":"N/A","name":"N/A","A_cm2":0.0,"I_y_cm4":0.0,"I_z_cm4":0.0,"S_y_cm3":0.0,"S_z_cm3":0.0,
            "J_cm4":0.0,"Iw_cm6":0.0,"It_cm4":0.0,"c_max_mm":0.0,
            "N_Rd_kN_db":None,"V_Rd_kN_db":None,"M_Rd_y_kNm_db":None,
            "alpha_curve":alpha_default_val,"flange_class_db":"n/a","web_class_bending_db":"n/a","web_class_compression_db":"n/a"
        }
    else:
        sr = selected_row_mapped
        # display in 3-per-row rectangular boxes (profile dims, area, inertia, torsion, resistances, buckling/classes)
        st.write("**Profile dimensions**")
        p1, p2, p3 = st.columns(3)
        p1.markdown(box_html("Area A (cm²)", f"{sr.get('A_cm2',0.0):.3f}"), unsafe_allow_html=True)
        p2.markdown(box_html("I_y (cm⁴)", f"{sr.get('I_y_cm4',0.0):.3f}"), unsafe_allow_html=True)
        p3.markdown(box_html("I_z (cm⁴)", f"{sr.get('I_z_cm4',0.0):.3f}"), unsafe_allow_html=True)
        st.write("**Section modulus & W_pl**")
        s1, s2, s3 = st.columns(3)
        s1.markdown(box_html("S_y (cm³)", f"{sr.get('S_y_cm3',0.0):.3f}"), unsafe_allow_html=True)
        s2.markdown(box_html("S_z (cm³)", f"{sr.get('S_z_cm3',0.0):.3f}"), unsafe_allow_html=True)
        s3.markdown(box_html("W_pl,y (cm³)", f"{sr.get('Wpl_y_cm3',0.0):.3f}"), unsafe_allow_html=True)
        st.write("**Torsion & warping**")
        t1, t2, t3 = st.columns(3)
        t1.markdown(box_html("J (cm⁴)", f"{sr.get('J_cm4',0.0):.3f}"), unsafe_allow_html=True)
        t2.markdown(box_html("It (cm⁴)", f"{sr.get('It_cm4',0.0):.3f}"), unsafe_allow_html=True)
        t3.markdown(box_html("Iw (cm⁶)", f"{sr.get('Iw_cm6',0.0):.3f}"), unsafe_allow_html=True)
        st.write("**Resistances (DB values if present)**")
        r1, r2, r3 = st.columns(3)
        r1.markdown(box_html("N_Rd (kN) — DB", f"{sr.get('N_Rd_kN_db','n/a')}"), unsafe_allow_html=True)
        r2.markdown(box_html("V_Rd (kN) — DB", f"{sr.get('V_Rd_kN_db','n/a')}"), unsafe_allow_html=True)
        r3.markdown(box_html("M_Rd,y (kN·m) — DB", f"{sr.get('M_Rd_y_kNm_db','n/a')}"), unsafe_allow_html=True)
        st.write("**Buckling curves & section classes**")
        b1, b2, b3 = st.columns(3)
        b1.markdown(box_html("α (buckling)", f"{sr.get('alpha_curve', alpha_default_val)}"), unsafe_allow_html=True)
        b2.markdown(box_html("Flange class", f"{sr.get('flange_class_db','n/a')}"), unsafe_allow_html=True)
        b3.markdown(box_html("Web class (bending)", f"{sr.get('web_class_bending_db','n/a')}"), unsafe_allow_html=True)
        bb1, bb2, bb3 = st.columns(3)
        bb1.markdown(box_html("Web class (compression)", f"{sr.get('web_class_compression_db','n/a')}"), unsafe_allow_html=True)
        bb2.empty(); bb3.empty()

        use_props = sr.copy()

# -------------------------
# Material design props (read-only)
# -------------------------
st.markdown("---")
st.markdown("### Design properties of material (read-only)")
mp1, mp2, mp3 = st.columns(3)
mp1.text_input("Modulus of elasticity E (MPa)", value="210000", disabled=True)
mp1.text_input("Yield strength fy (MPa)", value=f"{fy}", disabled=True)
mp2.text_input("Shear modulus G (MPa)", value="80769", disabled=True)
mp2.text_input("Partial factor γ_M0 (DB assumed)", value="1.0", disabled=True)
mp3.text_input("Partial factor γ_M1 (DB assumed)", value="1.0", disabled=True)
st.markdown("---")

# -------------------------
# Ready cases (kept small)
# -------------------------
st.markdown("---")
st.markdown("### Ready beam & frame cases (optional)")
use_ready = st.checkbox("Use ready case (prefill loads)")
def ss_udl(L,w): return (0.0, w*L*L/8.0, 0.0, w*L/2.0, 0.0)
ready_catalog = {"Beam":{"SS-UDL":{"label":"UDL on span","inputs":{"L":6.0,"w":10.0},"func":ss_udl}}}
if use_ready:
    chosen_type = st.selectbox("Type", options=["-- choose --", "Beam", "Frame"])
    if chosen_type and chosen_type!="-- choose --":
        categories = sorted(ready_catalog.get(chosen_type, {}).keys())
        if categories:
            chosen_cat = st.selectbox("Category", options=["-- choose --"] + categories)
            if chosen_cat and chosen_cat!="-- choose --":
                cases_dict = ready_catalog[chosen_type][chosen_cat]
                ck = list(cases_dict.keys())[0]
                scase = cases_dict[ck]
                st.markdown(f"**Selected case**: {scase['label']}")
                inputs = scase["inputs"]
                vals = {}
                for k,v in inputs.items():
                    vals[k] = st.number_input(k, value=float(v), key=f"case_{k}")
                if st.button("Apply case"):
                    try:
                        args = [vals[k] for k in inputs.keys()]
                        N_case, My_case, Mz_case, Vy_case, Vz_case = scase["func"](*args)
                        st.session_state["prefill_N_kN"] = float(N_case)
                        st.session_state["prefill_My_kNm"] = float(My_case)
                        st.session_state["prefill_Vy_kN"] = float(Vy_case)
                        st.success("Prefilled loads from case.")
                    except Exception:
                        st.error("Case apply failed.")

# -------------------------
# Loads & inputs
# -------------------------
st.header("Design forces and moments (ultimate state) - INPUT")
r1c1, r1c2, r1c3 = st.columns(3)
with r1c1:
    L = st.number_input("Element length L (m)", value=6.0, min_value=0.0)
with r1c2:
    N_kN = st.number_input("Axial force N (kN) (positive = compression)", value=st.session_state.get("prefill_N_kN",0.0))
with r1c3:
    Vy_kN = st.number_input("Shear V_y (kN)", value=st.session_state.get("prefill_Vy_kN",0.0))
r2c1, r2c2, r2c3 = st.columns(3)
with r2c1:
    Vz_kN = st.number_input("Shear V_z (kN)", value=0.0)
with r2c2:
    My_kNm = st.number_input("Bending M_y (kN·m) (about y)", value=st.session_state.get("prefill_My_kNm",0.0))
with r2c3:
    Mz_kNm = st.number_input("Bending M_z (kN·m) (about z)", value=0.0)
r3c1, r3c2, r3c3 = st.columns(3)
with r3c1:
    Tx_kNm = st.number_input("Torsion T_x (kN·m)", value=0.0)
with r3c2:
    st.write("")
with r3c3:
    st.write("")

axis_check = "both (y & z)"
st.sidebar.header("Output mode")
output_mode = st.sidebar.radio("Results output mode", ("Concise (no formulas)", "Full (with formulas & steps)"))
want_pdf = st.sidebar.checkbox("Enable PDF download (requires reportlab)")

# -------------------------
# Use properties -> compute resistances (prefer DB/custom)
# -------------------------
# ensure use_props exists
if 'use_props' not in locals():
    use_props = {
        "family":"N/A","name":"N/A","A_cm2":0.0,"I_y_cm4":0.0,"I_z_cm4":0.0,
        "S_y_cm3":0.0,"S_z_cm3":0.0,"J_cm4":0.0,"Iw_cm6":0.0,"It_cm4":0.0,"c_max_mm":0.0,
        "N_Rd_kN_db":None,"V_Rd_kN_db":None,"M_Rd_y_kNm_db":None,"alpha_curve":alpha_default_val,
        "flange_class_db":"n/a","web_class_bending_db":"n/a","web_class_compression_db":"n/a"
    }

N_N = N_kN * 1e3
Vy_N = Vy_kN * 1e3
Vz_N = Vz_kN * 1e3
My_Nm = My_kNm * 1e3
Mz_Nm = Mz_kNm * 1e3
T_Nm = Tx_kNm * 1e3

A_m2 = use_props.get("A_cm2",0.0) / 1e4
S_y_m3 = use_props.get("S_y_cm3",0.0) * 1e-6
S_z_m3 = use_props.get("S_z_cm3",0.0) * 1e-6
I_y_m4 = use_props.get("I_y_cm4",0.0) * 1e-8
I_z_m4 = use_props.get("I_z_cm4",0.0) * 1e-8
J_m4 = use_props.get("J_cm4",0.0) * 1e-8
c_max_m = use_props.get("c_max_mm",0.0) / 1000.0
Wpl_y_m3 = (use_props.get("Wpl_y_cm3",0.0)*1e-6) if use_props.get("Wpl_y_cm3",0.0)>0 else (1.1*S_y_m3 if S_y_m3>0 else 0.0)
alpha_curve_db = use_props.get("alpha_curve", alpha_default_val)

# require area for DB sections
if (not use_custom) and A_m2 <= 0:
    st.error("DB section missing area. Pick another DB section or use custom mode.")
    st.stop()

# Resistances: prefer DB/custom input; else compute simply
if use_props.get("N_Rd_kN_db") is not None:
    N_Rd_N = float(use_props.get("N_Rd_kN_db")) * 1e3
else:
    N_Rd_N = A_m2 * fy * 1e6 / 1.0  # gamma assumed 1 (DB holds partials)

if use_props.get("V_Rd_kN_db") is not None:
    V_Rd_N = float(use_props.get("V_Rd_kN_db")) * 1e3
else:
    Av_m2 = 0.6 * A_m2
    V_Rd_N = Av_m2 * fy * 1e6 / (math.sqrt(3) * 1.0) if Av_m2>0 else 0.0

if use_props.get("M_Rd_y_kNm_db") is not None:
    M_Rd_y_Nm = float(use_props.get("M_Rd_y_kNm_db")) * 1e3
else:
    M_Rd_y_Nm = Wpl_y_m3 * fy * 1e6 / 1.0 if Wpl_y_m3>0 else 0.0

# custom override (if using custom we already wrote into use_props earlier)
if use_custom:
    if use_props.get("N_Rd_kN_db") is not None:
        N_Rd_N = float(use_props.get("N_Rd_kN_db")) * 1e3
    if use_props.get("V_Rd_kN_db") is not None:
        V_Rd_N = float(use_props.get("V_Rd_kN_db")) * 1e3
    if use_props.get("M_Rd_y_kNm_db") is not None:
        M_Rd_y_Nm = float(use_props.get("M_Rd_y_kNm_db")) * 1e3

# stresses & shear/torsion
sigma_axial_Pa = N_N / A_m2 if A_m2>0 else 0.0
sigma_by_Pa = My_Nm / S_y_m3 if S_y_m3>0 else 0.0
sigma_bz_Pa = Mz_Nm / S_z_m3 if S_z_m3>0 else 0.0
tau_y_Pa = Vz_N / (0.6*A_m2) if A_m2>0 else 0.0
tau_z_Pa = Vy_N / (0.6*A_m2) if A_m2>0 else 0.0
tau_torsion_Pa = 0.0
if J_m4>0 and c_max_m>0:
    tau_torsion_Pa = T_Nm * c_max_m / J_m4
tau_total_Pa = math.sqrt(tau_y_Pa**2 + tau_z_Pa**2 + tau_torsion_Pa**2)
tau_total_MPa = tau_total_Pa / 1e6
sigma_eq_MPa = math.sqrt((abs((sigma_axial_Pa + sigma_by_Pa + sigma_bz_Pa)/1e6))**2 + 3.0*(tau_total_MPa**2))

tau_allow_Pa = 0.6 * (0.6 * fy) * 1e6  # approx

# utilizations
util_axial = abs(N_N)/N_Rd_N if (N_Rd_N and N_Rd_N>0) else None
util_My = abs(My_Nm)/M_Rd_y_Nm if (M_Rd_y_Nm and M_Rd_y_Nm>0) else None
util_shear = math.sqrt(Vy_N**2 + Vz_N**2)/V_Rd_N if (V_Rd_N and V_Rd_N>0) else None
util_torsion = tau_torsion_Pa / tau_allow_Pa if tau_allow_Pa>0 else None

# Buckling simplified
E = 210e9
buck_results = []
I_check_list = []
if axis_check in ("both (y & z)", "y (weak)"):
    I_check_list.append(("y", I_y_m4, K_y))
if axis_check in ("both (y & z)", "z (strong)"):
    I_check_list.append(("z", I_z_m4, K_z))

for axis_label, I_axis, K_axis in I_check_list:
    if I_axis is None or I_axis <= 0:
        buck_results.append((axis_label, None, None, None, None, "No I"))
        continue
    Leff_axis = K_axis * L
    Ncr = (math.pi**2 * E * I_axis) / (Leff_axis**2)
    lambda_bar = math.sqrt((A_m2 * fy * 1e6) / Ncr) if Ncr>0 else float('inf')
    alpha_use = use_props.get("alpha_curve", alpha_default_val)
    phi = 0.5 * (1.0 + float(alpha_use) * (lambda_bar**2))
    sqrt_term = max(phi**2 - lambda_bar**2, 0.0)
    chi = 1.0 / (phi + math.sqrt(sqrt_term)) if (phi + math.sqrt(sqrt_term))>0 else 0.0
    N_b_Rd_N = chi * A_m2 * fy * 1e6 / 1.0
    status = "OK" if abs(N_N) <= N_b_Rd_N else "EXCEEDS"
    buck_results.append((axis_label, Ncr, lambda_bar, chi, N_b_Rd_N, status))

N_b_Rd_candidates = [r[4] for r in buck_results if r[4] not in (None,)]
N_b_Rd_min_N = min(N_b_Rd_candidates) if N_b_Rd_candidates else None
compression_resistance_N = N_b_Rd_min_N if N_b_Rd_min_N is not None else N_Rd_N

# -------------------------
# Build checks & output table
# -------------------------
def status_and_util(applied, resistance):
    if resistance is None or resistance == 0:
        return ("n/a", None)
    util = abs(applied)/resistance
    return ("OK" if util<=1.0 else "EXCEEDS", util)

rows = []
applied_N = N_N if N_N>=0 else 0.0
res_comp_N = compression_resistance_N
status_comp, util_comp = status_and_util(applied_N, res_comp_N)
rows.append({"Check":"Compression (N≥0)","Applied":f"{applied_N/1e3:.3f} kN","Resistance":(f"{res_comp_N/1e3:.3f} kN" if res_comp_N else "n/a"),"Utilization":(f"{util_comp:.3f}" if util_comp else "n/a"),"Status":status_comp})
applied_tension_N = -N_N if N_N<0 else 0.0
res_tension_N = N_Rd_N
status_ten, util_ten = status_and_util(applied_tension_N, res_tension_N)
rows.append({"Check":"Tension (N<0)","Applied":f"{applied_tension_N/1e3:.3f} kN","Resistance":f"{res_tension_N/1e3:.3f} kN","Utilization":(f"{util_ten:.3f}" if util_ten else "n/a"),"Status":status_ten})
applied_shear_N = math.sqrt(Vy_N**2 + Vz_N**2)
res_shear_N = V_Rd_N
status_shear, util_shear_val = status_and_util(applied_shear_N, res_shear_N)
rows.append({"Check":"Shear (resultant Vy & Vz)","Applied":f"{applied_shear_N/1e3:.3f} kN","Resistance":(f"{res_shear_N/1e3:.3f} kN" if res_shear_N else "n/a"),"Utilization":(f"{util_shear_val:.3f}" if util_shear_val else "n/a"),"Status":status_shear})
applied_tau_Pa = tau_torsion_Pa
res_tau_allow_Pa = tau_allow_Pa
util_torsion = applied_tau_Pa / res_tau_allow_Pa if res_tau_allow_Pa>0 else None
status_tors = "OK" if util_torsion is not None and util_torsion <=1.0 else ("EXCEEDS" if util_torsion is not None else "n/a")
rows.append({"Check":"Torsion (τ = T·c/J)","Applied":(f"{applied_tau_Pa/1e6:.6f} MPa" if isinstance(applied_tau_Pa,(int,float)) else "n/a"),"Resistance":(f"{res_tau_allow_Pa/1e6:.6f} MPa (approx)" if res_tau_allow_Pa else "n/a"),"Utilization":(f"{util_torsion:.3f}" if util_torsion else "n/a"),"Status":status_tors})
rows.append({"Check":"Bending y-y (σ_by)","Applied":f"{sigma_by_Pa/1e6:.3f} MPa","Resistance":f"{0.6*fy:.3f} MPa","Utilization":(f"{(abs(sigma_by_Pa/1e6)/(0.6*fy)):.3f}" if (0.6*fy)>0 else "n/a"),"Status":"OK" if abs(sigma_by_Pa/1e6)/(0.6*fy)<=1.0 else "EXCEEDS"})
rows.append({"Check":"Bending z-z (σ_bz)","Applied":f"{sigma_bz_Pa/1e6:.3f} MPa","Resistance":f"{0.6*fy:.3f} MPa","Utilization":(f"{(abs(sigma_bz_Pa/1e6)/(0.6*fy)):.3f}" if (0.6*fy)>0 else "n/a"),"Status":"OK" if abs(sigma_bz_Pa/1e6)/(0.6*fy)<=1.0 else "EXCEEDS"})

for axis_label, Ncr, lambda_bar, chi, N_b_Rd_N, status in buck_results:
    if N_b_Rd_N:
        util_buck = abs(N_N)/N_b_Rd_N
        rows.append({"Check":f"Flexural buckling {axis_label}","Applied":f"{abs(N_N)/1e3:.3f} kN","Resistance":f"{N_b_Rd_N/1e3:.3f} kN","Utilization":f"{util_buck:.3f}","Status":"OK" if util_buck<=1.0 else "EXCEEDS"})
    else:
        rows.append({"Check":f"Flexural buckling {axis_label}","Applied":"n/a","Resistance":"n/a","Utilization":"n/a","Status":"n/a"})

st.markdown("---")
df_rows = pd.DataFrame(rows).set_index("Check")
overall_ok = not any(df_rows["Status"]=="EXCEEDS")
if overall_ok:
    st.success("Result summary: DESIGN OK — no checks exceed capacity (preliminary).")
else:
    st.error("Result summary: DESIGN NOT OK — one or more checks exceed capacity (preliminary).")

def highlight_row(row):
    s = row["Status"]
    if s == "OK":
        color = "background-color: #e6f7e6"
    elif s == "EXCEEDS":
        color = "background-color: #fde6e6"
    else:
        color = "background-color: #f0f0f0"
    return [color]*len(row)

styled = df_rows.style.apply(highlight_row, axis=1)
st.subheader("Cross-section & member checks (detailed)")
st.write("Legend: OK — within capacity; EXCEEDS — capacity exceeded; n/a — not applicable/missing data.")
st.write(styled)

# Save results
if "saved_results" not in st.session_state:
    st.session_state["saved_results"] = []

def build_result_record():
    return {
        "timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "doc_title": doc_name,
        "project_name": project_name,
        "position": position,
        "requested_by": requested_by,
        "revision": revision,
        "date": run_date.isoformat(),
        "section_type": use_props.get("family"),
        "section_name": use_props.get("name"),
        "A_cm2": use_props.get("A_cm2"),
        "L_m": L, "N_kN": N_kN, "My_kNm": My_kNm,
        "N_Rd_kN": N_Rd_N/1e3 if N_Rd_N else None,
        "M_Rd_y_kNm": M_Rd_y_Nm/1e3 if M_Rd_y_Nm else None,
        "V_Rd_kN": V_Rd_N/1e3 if V_Rd_N else None,
        "overall_ok": overall_ok
    }

svc1, svc2 = st.columns([1,1])
with svc1:
    if st.button("Save results"):
        st.session_state["saved_results"].append(build_result_record())
        st.success("Saved to session.")
with svc2:
    st.info("Saved runs are kept in this browser session (temporary).")

# End notes
st.markdown("---")
st.subheader("Notes & limitations")
st.write("""
- This tool is a screening calculator. For full EN1993 checks use dedicated design software / engineer verification.
- Column mapping: if families/sizes appear incorrect, change the selected columns in the sidebar mapping.
- DO NOT commit DB credentials into a public repo. Use Streamlit secrets for deployment.
""")
