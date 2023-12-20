import os

def get_dv_connection_string():
    username = os.getenv('STOCK_DB_USR')
    password = os.getenv('STOCK_DB_PWD')
    host = 'localhost'
    port = '5432'
    return f"postgresql+psycopg2://{username}:{password}@{host}:{port}/stock_dv"