import sqlite3, os

db_path = 'data/trading.db'
conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print('Tables:', cur.fetchall())

try:
    cur.execute("SELECT COUNT(*) FROM trades")
    print('Trade count:', cur.fetchone()[0])
    cur.execute("SELECT * FROM trades LIMIT 2")
    rows = cur.fetchall()
    if rows:
        print('Sample row:', rows[0])
        print('Columns:', [d[0] for d in cur.description])
except Exception as e:
    print('Error:', e)

conn.close()
