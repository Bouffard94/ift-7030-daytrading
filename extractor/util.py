from datetime import datetime, timedelta
from tradinghours import TradingHours

def common_elements(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    common = set1.intersection(set2)
    return list(common)

def stock_market_daterange(start: datetime, end: datetime, duration: int, tradinghours: TradingHours):
    if start >= end:
        raise Exception("START plus grand que END dans stock_market_daterange()")
    m_open, m_close = tradinghours.open_close()
    
    current_datetime = start
    while current_datetime < end:
        if current_datetime.time() < m_open:
            current_datetime = datetime.combine(current_datetime, m_open)

        if current_datetime.weekday() < 5 and current_datetime.date() not in tradinghours.holidays() and not current_datetime.time() >= m_close:
            frame_start = current_datetime
            current_datetime += timedelta(seconds=duration)
            current_datetime = min(current_datetime, datetime.combine(current_datetime.date(), m_close))
            frame_duration = current_datetime - frame_start
            yield frame_start, current_datetime, int(frame_duration.total_seconds())
        else:
            current_datetime = datetime.combine((current_datetime + timedelta(days=1)).date(), m_open)