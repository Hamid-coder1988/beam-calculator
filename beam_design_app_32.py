# beam_type_size_connect.py
# Self-contained: connect, detect table/columns, show Type -> Size selectors.
import streamlit as st
import pandas as pd
import traceback

# ---- Try import DB driver ----
try:
    import psycopg2
    HAS_PG = True
except Exception:
    psycopg2 = None
    HAS_PG = False
    st.error("psycopg2 not installed. Run: pip install psycopg2-binary")

# ---- get_conn() - hard-coded credentials (LOCAL TESTING ONLY) ----
def get_conn():
    if not HAS_PG:
        raise RuntimeError("psycopg2 not available")
    return psycopg2.connect(
        host="yamanote.proxy.rlwy.net",
        port=15500,
        database="railway",
        user="postgres",
        password="KcMoXOMMbbOQITUHrdJMOiwyNBDGyrFy",
        sslmode="require"
    )

# ---- fallback sample if DB not accessible ----
SAMPLE_ROWS = [
    {"Type": "IPE", "Size": "IPE 200", "A_cm2": 31.2},
    {"Type": "IPE", "Size": "IPE 300", "A_cm2": 45.0},
    {"Type": "RHS", "Size": "100x50", "A_cm2": 20.1},
]

# ---- helper: run a query safely and return DataFrame or error ----
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
        except:
            pass
        return None, f"{e}\n\n{tb}"

# ---- helper: list user tables in public schema ----
def list_user_tables():
    sql = """
      SELECT schemaname, tablename
      FROM pg_tables
      WHERE schemaname NOT IN ('pg_catalog','information_schema')
      ORDER BY schemaname, tablename;
    """
    df, err = run_sql(sql)
    if err:
        return None, err
    return df, None

# ---- helper: get columns for a table ----
def table_columns(table_name):
    sql = """
      SELECT column_name, data_type, ordinal_position
      FROM information_schema.columns
      WHERE table_schema = 'public' AND table_name = %s
      ORDER BY ordinal_position;
    """
    df, err = run_sql(sql, params=(table_name,))
    return df, err

# ---- detect candidate table that contains Type/Size-like columns ----
def detect_table_and_columns():
    # If DB not available, return sample markers
    if not HAS_PG:
        return None, None, "no-db"
    tables_df, err = list_user_tables()
    if err:
        return None, None, f"list-tables-error: {err}"

    # normalize list of table names
    tables = tables_df['tablename'].astype(str).tolist()

    # search priority: exact "Beam" or beam or any table name containing 'beam'
    priorities = []
    if "Beam" in tables: priorities.append("Beam")
    if "beam" in tables and "beam" not in priorities: priorities.append("beam")
    # any containing 'beam'
    for t in tables:
        if 'beam' in t.lower() and t not in priorities:
            priorities.append(t)
    # then append all remaining tables (so we inspect a few)
    for t in tables:
        if t not in priorities:
            priorities.append(t)

    # inspect tables and find one that has columns named Type/Size (case-insensitive)
    for tbl in priorities:
        cols_df, err = table_columns(tbl)
        if err:
            # skip this table if error
            continue
        cols = cols_df['column_name'].astype(str).tolist()
        cols_lower = [c.lower() for c in cols]
        if 'type' in cols_lower and 'size' in cols_lower:
            # found a good candidate
            return tbl, cols, None
    # if not found, return first table as a fallback (so user can inspect)
    return priorities[0] if priorities else None, None, "no-type-size-found"

# -------------------------
# Replacement: fetch types & sizes preserving table order and hide diagnostics
# -------------------------

@st.cache_data(show_spinner=False)
def fetch_types_and_sizes():
    """
    Read rows from the detected table in physical insertion order (ORDER BY ctid),
    then build:
      - types: ordered list of Type values in the order they first appear
      - sizes_map: dict Type -> ordered list of Size values in the order they first appear
    Falls back to SAMPLE_ROWS if DB is not available.
    """
    # sample fallback
    if not HAS_PG:
        df = pd.DataFrame(SAMPLE_ROWS)
        types = []
        sizes_map = {}
        for _, r in df.iterrows():
            t = str(r["Type"])
            s = str(r["Size"])
            if t not in types:
                types.append(t)
                sizes_map[t] = []
            if s not in sizes_map[t]:
                sizes_map[t].append(s)
        return types, sizes_map, "sample"

    # detect candidate table
    tbl, cols, detect_err = detect_table_and_columns()
    if not tbl:
        return [], {}, f"detect-error: {detect_err}"

    # read Type and Size preserving table order using ctid
    # (ctid is a physical row identifier — gives row insertion order until VACUUM/updates move tuples,
    #  but in practice it preserves order for simple imports)
    sql = f'SELECT "Type", "Size" FROM "{tbl}" ORDER BY ctid;'
    rows_df, err = run_sql(sql)
    if err:
        # try unquoted fallback
        sql2 = f"SELECT type, size FROM {tbl} ORDER BY ctid;"
        rows_df, err2 = run_sql(sql2)
        if err2:
            return [], {}, f"could-not-read-rows: {err}\n{err2}"
    if rows_df is None or rows_df.empty:
        return [], {}, f"no-rows-read-from-{tbl}"

    types = []
    sizes_map = {}
    for _, row in rows_df.iterrows():
        t = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
        s = str(row.iloc[1]) if pd.notna(row.iloc[1]) else ""
        if t == "":
            continue
        if t not in types:
            types.append(t)
            sizes_map[t] = []
        if s and s not in sizes_map[t]:
            sizes_map[t].append(s)

    return types, sizes_map, tbl


# -------------------------
# UI: simplified (no diagnostic messages)
# -------------------------
# (replace previous UI block that showed detection messages)
use_custom = st.checkbox("Use custom section (enter properties manually)", value=False)

selected_row = None

types, sizes_map, _detected_table = fetch_types_and_sizes()

# Only show DB selectors when not using custom
if not use_custom:
    if not types:
        # silent fallback: show sample table only when nothing found
        st.info("No Types found in DB. Use custom section or check DB column names.")
    else:
        type_sel = st.selectbox("Type (family)", options=["-- choose --"] + types)
        if type_sel and type_sel != "-- choose --":
            sizes = sizes_map.get(type_sel, [])
            if not sizes:
                st.info(f"No sizes found for Type '{type_sel}'.")
            else:
                size_sel = st.selectbox("Size", options=["-- choose --"] + sizes)
                if size_sel and size_sel != "-- choose --":
                    # fetch the full row for display (robust SQL)
                    if not HAS_PG:
                        # sample
                        df = pd.DataFrame(SAMPLE_ROWS)
                        sel = df[(df['Type']==type_sel) & (df['Size']==size_sel)]
                        selected_row = sel.iloc[0].to_dict() if not sel.empty else None
                    else:
                        tbl_detected = _detected_table if _detected_table and _detected_table != "sample" else None
                        row = None
                        if tbl_detected:
                            q_variants = [
                                (f'SELECT * FROM "{tbl_detected}" WHERE "Type" = %s AND "Size" = %s LIMIT 1;', (type_sel, size_sel)),
                                (f"SELECT * FROM {tbl_detected} WHERE type = %s AND size = %s LIMIT 1;", (type_sel, size_sel)),
                            ]
                            for q,p in q_variants:
                                df_row, err = run_sql(q, params=p)
                                if err:
                                    continue
                                if df_row is not None and not df_row.empty:
                                    row = df_row.iloc[0].to_dict()
                                    break
                        selected_row = row

                    if selected_row:
                        # show only the section JSON (no diagnostic banners)
                        st.write(f"Loaded: {type_sel} / {size_sel}")
                        st.json(selected_row)
                    else:
                        st.warning("Could not fetch the selected full row from DB (check column names).")
else:
    st.info("Custom section mode ON — you will enter properties manually.")

