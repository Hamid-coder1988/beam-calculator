# Beam Type -> Size selection (self-contained)
# Paste this block into your Streamlit app (or run as a small test file).

import streamlit as st
import pandas as pd
import traceback

# -------------------------
# Try DB driver
# -------------------------
try:
    import psycopg2
    HAS_PG = True
except Exception:
    psycopg2 = None
    HAS_PG = False

# -------------------------
# Simple get_conn() (hard-coded) - LOCAL TESTING ONLY
# Replace host/port/database/user/password if needed
# -------------------------
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

# -------------------------
# Sample fallback data (used if DB can't be read)
# -------------------------
SAMPLE_ROWS = [
    {"Type": "IPE", "Size": "IPE 200", "A_cm2": 31.2},
    {"Type": "IPE", "Size": "IPE 300", "A_cm2": 45.0},
    {"Type": "RHS", "Size": "100x50", "A_cm2": 20.1},
]

# -------------------------
# Cached DB helpers
# -------------------------
@st.cache_data(show_spinner=False)
def get_types():
    """Return sorted list of distinct Type values from DB or sample."""
    if not HAS_PG:
        df = pd.DataFrame(SAMPLE_ROWS)
        return sorted(df["Type"].dropna().unique().tolist())
    try:
        conn = get_conn()
        sql = 'SELECT DISTINCT "Type" FROM "Beam" ORDER BY "Type";'
        df = pd.read_sql(sql, conn)
        conn.close()
        return df["Type"].astype(str).tolist()
    except Exception as e:
        # return sample on error and log
        st.sidebar.warning(f"DB Error reading Types (using sample): {e}")
        return sorted(pd.DataFrame(SAMPLE_ROWS)["Type"].unique().tolist())

@st.cache_data(show_spinner=False)
def get_sizes(type_value):
    """Return list of Size values for given Type from DB or sample."""
    if not HAS_PG:
        df = pd.DataFrame(SAMPLE_ROWS)
        return df[df["Type"] == type_value]["Size"].astype(str).tolist()
    try:
        conn = get_conn()
        sql = 'SELECT "Size" FROM "Beam" WHERE "Type" = %s ORDER BY "Size";'
        df = pd.read_sql(sql, conn, params=(type_value,))
        conn.close()
        return df["Size"].astype(str).tolist()
    except Exception as e:
        st.sidebar.warning(f"DB Error reading Sizes (using sample): {e}")
        return pd.DataFrame(SAMPLE_ROWS)[pd.DataFrame(SAMPLE_ROWS)["Type"] == type_value]["Size"].astype(str).tolist()

@st.cache_data(show_spinner=False)
def get_section_row(type_value, size_value):
    """Return full row dict for given Type+Size or None."""
    if not HAS_PG:
        df = pd.DataFrame(SAMPLE_ROWS)
        row = df[(df["Type"] == type_value) & (df["Size"] == size_value)]
        return row.iloc[0].to_dict() if not row.empty else None
    try:
        conn = get_conn()
        sql = 'SELECT * FROM "Beam" WHERE "Type" = %s AND "Size" = %s LIMIT 1;'
        df = pd.read_sql(sql, conn, params=(type_value, size_value))
        conn.close()
        if df.empty:
            return None
        return df.iloc[0].to_dict()
    except Exception as e:
        st.sidebar.warning(f"DB Error reading section row (using sample): {e}")
        df = pd.DataFrame(SAMPLE_ROWS)
        row = df[(df["Type"] == type_value) & (df["Size"] == size_value)]
        return row.iloc[0].to_dict() if not row.empty else None

# -------------------------
# UI: Section selection
# -------------------------
st.header("Section selection (Type → Size)")

# Allow the user to choose custom mode (so variable use_custom is always defined)
use_custom = st.checkbox("Use custom section (enter properties manually)", value=False,
                         help="Tick to enter section properties manually instead of selecting from DB.")

selected_row = None  # ensure defined before use

# Only show DB selector if not using custom
if not use_custom:
    # Load available Types
    try:
        types = get_types()
    except Exception as e:
        types = []
        st.error(f"Error loading Types: {e}")

    if not types:
        st.info("No Types available from DB or sample. Tick 'Use custom section' to input properties manually.")
    else:
        type_sel = st.selectbox("Type (family)", options=["-- choose --"] + types, index=0)
        if type_sel and type_sel != "-- choose --":
            # Load sizes for chosen type
            try:
                sizes = get_sizes(type_sel)
            except Exception as e:
                sizes = []
                st.error(f"Error loading sizes for Type '{type_sel}': {e}")

            if not sizes:
                st.info(f"No sizes found for Type '{type_sel}'.")
            else:
                size_sel = st.selectbox("Size", options=["-- choose --"] + sizes, index=0)
                if size_sel and size_sel != "-- choose --":
                    # load the full row
                    try:
                        selected_row = get_section_row(type_sel, size_sel)
                    except Exception as e:
                        selected_row = None
                        st.error(f"Error loading section row: {e}")

                    if selected_row:
                        st.success(f"Loaded section: {size_sel}")
                        # show the raw row so you can see available column names/values
                        st.write("Selected row (raw):")
                        st.json(selected_row)
                    else:
                        st.warning("Selected row not found in DB/sample.")
else:
    st.info("Custom section mode enabled — enter properties manually in the custom inputs below (not included in this snippet).")

# -------------------------
# Optional: map selected_row to use_props (minimal)
# -------------------------
# If you want to pass DB values to downstream code, build a normalized dict here:
if selected_row is not None and not use_custom:
    # minimal mapping (extend keys as needed)
    use_props = {
        "family": selected_row.get("Type") or selected_row.get("family"),
        "name": selected_row.get("Size") or selected_row.get("name"),
        "A_cm2": selected_row.get("A_cm2") or selected_row.get("Area") or None,
        # add further mappings your app requires...
    }
    st.write("Mapped use_props (minimal):")
    st.write(use_props)
else:
    # no DB row selected — downstream code should handle custom inputs
    use_props = None
