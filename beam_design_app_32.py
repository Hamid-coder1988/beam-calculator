import psycopg2

def get_conn():
    return psycopg2.connect(
        host="yamanote.proxy.rlwy.net",
        port=15500,
        database="railway",
        user="postgres",
        password="KcMoXOMMbbOQITUHrdJMOiwyNBDGyrFy",
        sslmode="require"
    )

conn = get_conn()
cur = conn.cursor()
cur.execute('SELECT * FROM "Beam" LIMIT 5;')
print(cur.fetchall())
