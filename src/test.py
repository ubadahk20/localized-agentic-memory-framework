"""
import os
os.remove("data/memory.db")

"""
import sqlite3
conn = sqlite3.connect("data/memory.db")
conn.executescript(open("data/schema.sql").read())
conn.close()
