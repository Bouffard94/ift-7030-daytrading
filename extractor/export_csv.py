from db import get_dv_connection_string
import sqlalchemy
import pandas as pd
from datetime import datetime
import os
import glob
import shutil

CSV_DIR = 'csv'

table_name = 'common_10s'

files = glob.glob(f"{CSV_DIR}/*")
for f in files:
    os.remove(f)

conn_str = get_dv_connection_string()
engine = sqlalchemy.create_engine(conn_str)

# SINGLE FILE
sql = sqlalchemy.text(f"""
select *
from stock_data.{table_name}
""")
df = pd.read_sql(sql, engine.connect())
df.to_csv(f"csv/{table_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv")

# SEPARATE FILES
# sql = sqlalchemy.text(f"""
# select cs.ticker as ticker
# from stock_data.{table_name} cs
# group by cs.ticker
# """)
# df = pd.read_sql(sql, engine.connect())
# for ticker in df['ticker']:
#     print(f"Getting data for {ticker}")
#     sql = sqlalchemy.text(f"""
#     select *
#     from stock_data.{table_name} cs
#     where cs.ticker = '{ticker}'
#     """)
#     df_ticker = pd.read_sql(sql, engine.connect())
#     df_ticker.to_csv(f"csv/{table_name}_{ticker}_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv")

# archive_name = f"{table_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
# shutil.make_archive(archive_name, 'zip', CSV_DIR)