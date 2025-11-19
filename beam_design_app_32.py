import psycopg2
import pandas as pd
import traceback

# ----- EDIT credentials if needed (you said you hard-coded) -----
DB_KW = dict(
    host="yamanote.proxy.rlwy.net",
    port=15500,
    database="railway",
    user="postgres",
    password="KcMoXOMMbbOQITUHrdJMOiwyNBDGyrFy",
    sslmode="require"
)

def run():
    try:
        conn = psycopg2.connect(**DB_KW)
    except Exception as e:
        print("Could not connect to DB:", e)
        return

    try:
        # 1) list all tables in public schema
        sql_tables = """
            SELECT schemaname, tablename
            FROM pg_tables
            WHERE schemaname NOT IN ('pg_catalog','information_schema')
            ORDER BY schemaname, tablename;
        """
        print("\n--- All user tables ---")
        tables = pd.read_sql(sql_tables, conn)
        print(tables.to_string(index=False))

        # 2) do a case-insensitive search for 'beam' in table names
        print("\n--- Tables matching 'beam' (case-insensitive) ---")
        sql_like = """
            SELECT schemaname, tablename
            FROM pg_tables
            WHERE tablename ILIKE %s
            ORDER BY schemaname, tablename;
        """
        tbl_like = pd.read_sql(sql_like, conn, params=("%beam%",))
        if tbl_like.empty:
            print("No tables matched '%beam%'.")
        else:
            print(tbl_like.to_string(index=False))

        # 3) for each candidate table (all tables and the matches), show columns and preview
        candidates = tbl_like['tablename'].tolist()
        # If no matches, take all user tables as cautious fallback
        if not candidates:
            candidates = tables['tablename'].tolist()

        for t in candidates:
            print(f"\n--- Inspecting table: {t} ---")
            # column list
            sql_cols = """
                SELECT column_name, data_type, ordinal_position
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position;
            """
            cols = pd.read_sql(sql_cols, conn, params=(t,))
            if cols.empty:
                print("(no columns returned)")
            else:
                print("Columns:")
                print(cols.to_string(index=False))

            # preview first 5 rows (use safe quoting)
            try:
                preview = pd.read_sql(f'SELECT * FROM "{t}" LIMIT 5;', conn)
                if preview.empty:
                    print("Preview: (no rows)")
                else:
                    print("Preview rows (first 5):")
                    print(preview.to_string(index=False))
            except Exception as ex:
                print("Could not preview table (maybe quoting/schema issue):", ex)
                traceback.print_exc()

    finally:
        conn.close()

if __name__ == "__main__":
    run()
