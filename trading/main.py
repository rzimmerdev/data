import json
from datetime import datetime

import numpy as np
import pandas as pd

from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager

api_key = 'eW4p0oktgWfTP5ycSABmfuYreNwBy9Cu7UtVSbV8LMpsR0Ik0XVJJs4bzEjjD759'
api_secret = 'hVDpfjDK3hRmGiVsXLPhYhetLjuTwgWw7TP0YVawWM3y9sOkIQn18jvM2wiAwL57'

client = Client(api_key, api_secret)
depth = client.get_order_book(symbol='ETHBRL')

order = client.create_test_order(
    symbol='BNBBTC',
    side=Client.SIDE_BUY,
    type=Client.ORDER_TYPE_MARKET,
    quantity=100)

prices = client.get_all_tickers()

print(next((asset for asset in prices if asset['symbol'] == 'ETHBRL'), None))

symbols = [asset['symbol'] for asset in prices]

symbol = 'ETHBRL'

klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, "365 day ago UTC")

candles = []

for line_index in range(0, len(klines)):
    line = klines[line_index]
    candle = {'open_time': datetime.utcfromtimestamp(line[0] / 1000),
              'open': float(line[1]), 'high': float(line[2]), 'low': float(line[3]), 'close': float(line[4]),
              'volume': float(line[5]), 'asset_volume': float(line[7]),
              'n_trades': float(line[8])}

    if candle['open_time'] and candle['close']:
        candles.append(candle)

df = pd.DataFrame(candles)
date_time = pd.to_datetime(df.pop('open_time'), format='%d/%m/%Y')

df.describe().transpose()
