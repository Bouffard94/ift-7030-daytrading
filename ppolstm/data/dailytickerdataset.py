from torch.utils.data import Dataset
import torch
import pandas as pd
from datetime import date

class DailyTickerDataset(Dataset):
    def __init__(self, data: list[torch.Tensor], tickers: list[str], dates: list[date]):
        self.data = data
        self.tickers = tickers
        self.dates = dates
    
    def __len__(self):
        return len(self.tickers)
    
    def __getitem__(self, idx):
        day_data = self.data[idx]
        # Add 1 to volume and barcount for non zero div
        day_data[:, 5] += 1
        day_data[:, 6] += 1
        # Compute the delta percentage between each rows
        return (day_data[1:] - day_data[:-1]) / day_data[:-1]

    @staticmethod
    def from_csv(stock_file: str, nrows: int = None):
        df = pd.read_csv(stock_file, nrows=nrows)

        df['date'] = pd.to_datetime(df['datetime']).dt.date
        ticker_date = df[['ticker', 'date']].drop_duplicates()
        tickers = ticker_date['ticker'].to_list()
        dates = ticker_date['date'].to_list()
        
        data = []
        data_col = ['open', 'high', 'low', 'close', 'average', 'volume', 'barcount']
        for i in range(len(tickers)):
            daily_ticker_df = df[(df['date'] == dates[i]) & (df['ticker'] == tickers[i])]
            data.append(torch.tensor(daily_ticker_df[data_col].values, dtype=torch.float32))

        return DailyTickerDataset(data, tickers, dates)