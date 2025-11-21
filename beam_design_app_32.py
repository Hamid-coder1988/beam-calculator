# -------------------------
# Sidebar: material & defaults (simplified)
# -------------------------
st.sidebar.header("Material & section selection")

# Only standard steels (no custom)
material = st.sidebar.selectbox("Material", ("S235", "S275", "S355"),
                                help="Select steel grade (typical EN names).")
if material == "S235":
    fy = 235.0
elif material == "S275":
    fy = 275.0
else:
    fy = 355.0

# Partial factors removed from UI (assumed handled in DB). Use defaults here.
# If your DB contains gamma values, they'll be used later via use_props.
gamma_M0 = 1.0
gamma_M1 = 1.0

# Buckling K factors (these are still useful to let user change for the run)
st.sidebar.markdown("Buckling effective length factors (K):")
K_z = st.sidebar.number_input("K_z — flexural buckling about z–z", value=1.0, min_value=0.1, step=0.05)
K_y = st.sidebar.number_input("K_y — flexural buckling about y–y", value=1.0, min_value=0.1, step=0.05)
K_LT = st.sidebar.number_input("K_LT — lateral–torsional buckling", value=1.0, min_value=0.1, step=0.05)
K_T = st.sidebar.number_input("K_T — torsional buckling factor (reserved)", value=1.0, min_value=0.1, step=0.05)

alpha_default_val = 0.49

# -------------------------
# Section selection moved to SIDEBAR (under Material)
# -------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Section selection (DB)")

# Use the dataframe `df_sample_db` that you load earlier as df_sample_db
# If you are using real DB, df_sample_db should be the DataFrame returned by load_beam_db()
try:
    families_list = df_sample_db['family'].dropna().unique().tolist()
except Exception:
    families_list = []

# keep DB ordering for names (do NOT sorted) — user asked to preserve DB order
family = st.sidebar.selectbox("Section family", options=["-- choose --"] + families_list,
                              help="Choose section family (database).")

selected_row = None
selected_name = None
if family and family != "-- choose --":
    # preserve DB order for sizes (use the df_f order)
    df_f = df_sample_db[df_sample_db['family'] == family]
    names = df_f['name'].dropna().tolist()
    selected_name = st.sidebar.selectbox("Section size", options=["-- choose --"] + names,
                                         help="Choose section size (database). Selecting a size loads read-only properties.")
    if selected_name and selected_name != "-- choose --":
        # pick the first matching row (if multiple rows share the same name)
        selected_row = df_f[df_f['name'] == selected_name].iloc[0].to_dict()
        # optional: store raw DB row in session for debugging / later mapping
        st.session_state.setdefault("last_loaded_section_row", selected_row)
        # show a small confirmation (no big debug lines)
        st.sidebar.success(f"Loaded section: {selected_name}")

# Allow custom section toggle (keep on main page as before)
# Note: keep the 'use_custom' checkbox in the main canvas as earlier code expects it.
