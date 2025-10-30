import streamlit as st
import pyodbc

def get_db_connection():
    config = st.secrets["sqlserver"]
    
    # Build connection string
    conn_str = (
        f"DRIVER={{{config['driver']}}};"
        f"SERVER={config['server']};"
        f"DATABASE={config['database']};"
    )
    
    if config.get("trusted_connection", "no").lower() == "yes":
        conn_str += "Trusted_Connection=yes;"
    else:
        conn_str += f"UID={config['username']};PWD={config['password']};"
    
    conn = pyodbc.connect(conn_str)
    return conn