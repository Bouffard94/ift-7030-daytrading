from ib_insync import *
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import sqlalchemy
from db import get_dv_connection_string
from util import stock_market_daterange, common_elements
from tradinghours import NYSETradingHours

INTERVAL_SEC = 12 # 50 calls/10min, max 60calls/10mins

ib = IB()
#ib.connect('127.0.0.1', 4001, clientId=1) # IB Gateway real
ib.connect('127.0.0.1', 4002, clientId=1) # IB Gateway paper
#ib.connect('127.0.0.1', 7497, clientId=1) # IB Workstation paper

def contract_exist(contract: Contract):
    details = ib.reqContractDetails(contract)
    if not details:
        return False
    return True

def fetch_stock_data(
        stock: Stock, 
        start: datetime, 
        end: datetime,
        duration: int,
        bar_size: int,
        table_name: str = 'common_10s'
    ):

    next_query_time = datetime.now()
    for start_date_time, end_date_time, frame_duration in stock_market_daterange(start, end, duration, NYSETradingHours()):
        bars = ib.reqHistoricalData(
            stock,
            endDateTime=end_date_time,
            durationStr=str(frame_duration) + " S",
            barSizeSetting=str(bar_size) + " secs",
            whatToShow='TRADES',
            useRTH=True
        )
        next_query_time += timedelta(seconds=INTERVAL_SEC)

        df = util.df(bars)
        df = df.rename(columns={"barCount": "barcount", "date": "datetime"})
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['datetime'] = df['datetime'].dt.tz_localize(None)
        df['ticker'] = stock.symbol

        # Keep only data after the max date (in case of missing market data)
        sql = sqlalchemy.text(f"""
        select max(cs."datetime") as max_datetime
        from stock_data.{table_name} cs
        where cs.ticker = '{stock.symbol}' and cs."datetime"::date between 
        '{df['datetime'].min().strftime('%Y-%m-%d %H:%M:%S')}' and '{df['datetime'].max().strftime('%Y-%m-%d %H:%M:%S')}'
        """)
        max_datetime = pd.read_sql(sql, engine.connect())['max_datetime'].iloc[0]
        if max_datetime: 
            max_date = pd.Timestamp(max_datetime)
            df = df[df['datetime'] > max_date]

        if not df.empty:
            df.to_sql(name=table_name, schema='stock_data', con=engine, if_exists='append', index=False)

        print(f"{stock.symbol}: {str(end_date_time.date())} {str(start_date_time.time())} - {str(end_date_time.time())}")
        while datetime.now() < next_query_time:
            pass

# PARAMS
new_stocks = []
start = (datetime.now() - relativedelta(months=6)).replace(hour=9, minute=30, second=0)
end = (datetime.now() - relativedelta(days=1)).replace(hour=16, minute=0, second=0)
bar_size = 10
duration = 14400
table_name = 'common_10s'

conn_str = get_dv_connection_string()
engine = sqlalchemy.create_engine(conn_str)
sql = sqlalchemy.text(f"""
select cs.ticker, max(cs."datetime") as max_date
from stock_data.{table_name} cs
group by cs.ticker
""")
df = pd.read_sql(sql, engine.connect())

common_stocks = common_elements(new_stocks, df['ticker'].to_list())
if common_stocks:
    raise Exception(f"New stocks already exists in database: {common_stocks}")

# UPDATE EXISTING TICKERS
for idx in df.index:
    stock = Stock(
        symbol=df['ticker'][idx],
        exchange='SMART', 
        currency='USD'
    )
    if contract_exist(stock):
        update_start = pd.to_datetime(df['max_date'][idx]) + pd.Timedelta(seconds=10)
        update_start = update_start.to_pydatetime()
        if update_start.date() < end.date():
            fetch_stock_data(stock=stock, start=update_start, end=end, duration=duration, bar_size=bar_size, table_name=table_name)

# ADD NEW TICKERS
for symbol in new_stocks:
    stock = Stock(
        symbol=symbol, 
        exchange='SMART', 
        currency='USD'
    )
    if contract_exist(stock):
        fetch_stock_data(stock=stock, start=start, end=end, duration=duration, bar_size=bar_size, table_name=table_name)