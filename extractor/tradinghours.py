from abc import ABC, abstractmethod
from datetime import time, date

class TradingHours(ABC):
    @abstractmethod
    def holidays(self):
        pass

    @abstractmethod
    def open_close(self):
        pass

# https://www.nyse.com/markets/hours-calendars
class NYSETradingHours():
    def __init__(self):
        pass
    
    def holidays(self):
        return [
            # 2023
            date(year=2023, month=1, day=2), # New Years Day
            date(year=2023, month=1, day=16), # Martin Luther King, Jr. Day
            date(year=2023, month=2, day=20), # Washington's Birthday
            date(year=2023, month=4, day=7), # Good Friday
            date(year=2023, month=5, day=29), # Memorial Day
            date(year=2023, month=6, day=19), # Juneteenth National Independence Day
            date(year=2023, month=7, day=4), # Independence Day
            date(year=2023, month=9, day=4), # Labor Day
            date(year=2023, month=11, day=23), # Thanksgiving Day
            date(year=2023, month=12, day=25), # Christmas Day
            
            # 2024
            date(year=2023, month=1, day=1), # New Years Day
            date(year=2023, month=1, day=15), # Martin Luther King, Jr. Day
            date(year=2023, month=2, day=19), # Washington's Birthday
            date(year=2023, month=3, day=29), # Good Friday
            date(year=2023, month=5, day=27), # Memorial Day
            date(year=2023, month=6, day=19), # Juneteenth National Independence Day
            date(year=2023, month=7, day=4), # Independence Day
            date(year=2023, month=9, day=2), # Labor Day
            date(year=2023, month=11, day=28), # Thanksgiving Day
            date(year=2023, month=12, day=25), # Christmas Day

            # 2025
            date(year=2023, month=1, day=1), # New Years Day
            date(year=2023, month=1, day=20), # Martin Luther King, Jr. Day
            date(year=2023, month=2, day=17), # Washington's Birthday
            date(year=2023, month=4, day=18), # Good Friday
            date(year=2023, month=5, day=26), # Memorial Day
            date(year=2023, month=6, day=19), # Juneteenth National Independence Day
            date(year=2023, month=7, day=4), # Independence Day
            date(year=2023, month=9, day=1), # Labor Day
            date(year=2023, month=11, day=27), # Thanksgiving Day
            date(year=2023, month=12, day=25), # Christmas Day
        ]
    
    def open_close(self):
        return time(9,30,0), time(16,0,0)

        