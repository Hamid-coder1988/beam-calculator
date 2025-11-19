# db_debug.py
import streamlit as st
import pandas as pd
import traceback

# ---- Replace or ensure you have get_conn() defined in this file or imported ----
# If you already have a get_conn() in your app, import it. Otherwise uncomment and edit below:
import psycopg2

def get_conn():
    # direct credentials (for local debugging only) - replace if different
    return psycopg2.connect(
        host="yamanote.proxy.rlwy.net",
        port=15500,
        database="railway",
        user="postgres",
        password="KcMoXOMMbbOQITUHrdJMOiwyNBDGyrFy",
        sslmode="require"
    )

st.set_page_config(page_title="DB debug", layout="wide")
st.title("DB Debug — Show tables, types, sizes, preview row")

# helper to safely run SQL and return dataframe or error
def run_sql(sql, params=None, safe_table=None):
    try:
        conn = get_conn()
    except Exception as e:
        return None, f"Could not connect: {e}"
    try:
        # optional safety: if safe_table provided, ensure it's in identifier list
        if safe_table:
            # nothing here, just placeholder for safety policy in production
            pass
        df = pd.read_sql(sql, conn, params=params)
        conn.close()
        return df, None
    except Exception as e:
        tb = traceback.format_exc()
        return None, f"{e}\n\n{tb}"

# 1) list tables in public schema
sql_tables = """
    SELECT schemaname, tablename
    FROM pg_tables
    WHERE schemaname NOT IN ('pg_catalog','information_schema')
    ORDER BY schemaname, tablename;
"""
tables_df, err = run_sql(sql_tables)
if err:
    st.error("Error reading table list: " + str(err))
    st.stop()
else:
    st.subheader("User tables (public schema)")
    st.dataframe(tables_df)

# 2) try to find any table name that matches 'beam' (case-insensitive)
sql_like = """
    SELECT schemaname, tablename
    FROM pg_tables
    WHERE tablename ILIKE %s
    ORDER BY schemaname, tablename;
"""
tbl_like_df, err = run_sql(sql_like, params=("%beam%",))
if err:
    st.warning("Error searching for 'beam' tables: " + str(err))
else:
    st.subheader("Tables matching '%beam%' (case-insensitive)")
    st.dataframe(tbl_like_df)

# 3) If 'Beam' exists exactly, attempt to read Type distinct values and Size values.
# We'll try several variants to be robust: exact "Beam", lowercase beam, any matching table.
candidate_tables = []
if not tbl_like_df.empty:
    candidate_tables = tbl_like_df['tablename'].astype(str).tolist()

# also include exactly "Beam" and 'beam' as priority candidates
if "Beam" not in candidate_tables:
    candidate_tables.insert(0, "Beam")
if "beam" not in candidate_tables:
    candidate_tables.insert(1, "beam")

st.subheader("Candidate tables we will inspect (priority order)")
st.write(candidate_tables)

# inspect up to first 6 candidate tables
inspected = {}
for t in list(dict.fromkeys(candidate_tables))[:6]:
    st.markdown(f"---\n### Inspecting table: `{t}`")
    # columns
    sql_cols = """
        SELECT column_name, data_type, ordinal_position
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position;
    """
    cols_df, err = run_sql(sql_cols, params=(t,))
    if err:
        st.write(f"Could not get columns for `{t}`: {err}")
        continue
    if cols_df.empty:
        st.write("(no columns found or table not present)")
        continue
    st.write("Columns:")
    st.dataframe(cols_df)

    # preview first 5 rows (use quoting)
    preview_sql = f'SELECT * FROM "{t}" LIMIT 5;'
    preview_df, err = run_sql(preview_sql)
    if err:
        st.write(f"Could not preview `{t}`: {err}")
    else:
        st.write("Preview (first 5 rows):")
        st.dataframe(preview_df)

    inspected[t] = {"columns": cols_df, "preview": preview_df}

# 4) If table 'Beam' (any case) exists, try to read Type & Size distinct values and show them
found = None
for candidate in inspected.keys():
    # check if this candidate has columns named Type and Size (case-sensitive)
    cols = inspected[candidate]["columns"]['column_name'].astype(str).tolist()
    cols_lower = [c.lower() for c in cols]
    if "type" in cols_lower and "size" in cols_lower:
        found = candidate
        break

if not found:
    st.warning("No table with columns named 'Type' and 'Size' found in inspected candidates. Check column names above.")
else:
    st.success(f"Using table `{found}` for Type/Size reads.")
    # read distinct Type values (try quoted exact, then lower-case)
    sql_types_quoted = f'SELECT DISTINCT "Type" FROM "{found}" ORDER BY "Type";'
    types_df, err = run_sql(sql_types_quoted)
    if err:
        st.write("Could not read quoted 'Type' column, trying lowercase identifier...")
        sql_types_unquoted = f"SELECT DISTINCT type FROM {found} ORDER BY type;"
        types_df, err = run_sql(sql_types_unquoted)
    if err:
        st.error("Could not read Type values: " + str(err))
    else:
        st.write("Distinct Type values:")
        st.write(types_df)

    # ask user to select one type to preview sizes
    if types_df is not None and not types_df.empty:
        # get first Type value as default
        first_type = types_df.iloc[0,0]
        sel_type = st.selectbox("Pick a Type to preview sizes", options=[str(x[0]) for x in types_df.values])
        # get sizes for chosen type
        try:
            sql_sizes = f'SELECT DISTINCT "Size" FROM "{found}" WHERE "Type" = %s ORDER BY "Size";'
            sizes_df, err = run_sql(sql_sizes, params=(sel_type,))
            if err:
                # try unquoted fallback
                sql_sizes2 = f"SELECT DISTINCT size FROM {found} WHERE type = %s ORDER BY size;"
                sizes_df, err = run_sql(sql_sizes2, params=(sel_type,))
            if err:
                st.error("Could not read sizes: " + str(err))
            else:
                st.write("Sizes for selected Type:")
                st.write(sizes_df)
                # preview one selected size row
                if not sizes_df.empty:
                    sel_size = st.selectbox("Pick a Size to preview full row", options=[str(x[0]) for x in sizes_df.values])
                    # fetch full row
                    try:
                        sql_row = f'SELECT * FROM "{found}" WHERE "Type" = %s AND "Size" = %s LIMIT 1;'
                        row_df, err = run_sql(sql_row, params=(sel_type, sel_size))
                        if err:
                            # try unquoted fallback
                            sql_row2 = f"SELECT * FROM {found} WHERE type = %s AND size = %s LIMIT 1;"
                            row_df, err = run_sql(sql_row2, params=(sel_type, sel_size))
                        if err:
                            st.error("Could not read full row: " + str(err))
                        else:
                            st.write("Full row preview (the dict you can map to your app):")
                            st.dataframe(row_df.T)  # transpose so columns show as rows
                    except Exception as e:
                        st.error("Error fetching row: " + str(e))
        except Exception as e:
            st.error("Failed to query sizes: " + str(e))
