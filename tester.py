import json
import time
import logging

import market_maker.bitmex as bm
from market_maker.settings import settings
import market_maker.auth.poloniex_keys as polo_api

from market_maker.binance import Binance
from market_maker.poloniex import PoloniexWebsocket

from market_maker.hyperliquid import HyperLiquidWebsocket
from market_maker.hyperliquid import HyperLiquidConnector

from market_maker.difx import DIFXWebsocket

# Test the code


logging.basicConfig(level=logging.INFO)

symbol = "ETH"

hl = HyperLiquidWebsocket(symbol=symbol, testnet=False)
hl.connect()
time.sleep(2)
hl.subscribe("l2Book", coin=symbol)  # Example subscription method for l2Book channel

while True:
    book = hl.data.get('l2Book')
    if book:
        print(book)
    else:
        logging.warning("No book data received.")
    time.sleep(2)



exit()
symbol = "BTCUSDT"
difx = DIFXWebsocket(symbol=symbol)
difx.connect()
time.sleep(2)

try:
    while difx.is_alive:
        book = difx.get_order_book()
        print(book)
        time.sleep(5)  # Let's also give it some sleep time to not spam the server
except KeyboardInterrupt:
    difx.exit()




exit()
api_key = polo_api.API_KEY
api_secret = polo_api.API_SECRET

poloniex = PoloniexWebsocket("BTC_USDT", api_key, api_secret)
poloniex.connect()
time.sleep(2)


while True:
    book = poloniex.get_order_book()
    if book['data']['action'] == 'snapshot':
        print(book['data'])
        time.sleep(2)




exit()
binance = Binance("BTCUSDT")
time.sleep(5)
i = 0
while True:
    market_buy_orders = binance.get_market_buy_orders()
    market_sell_orders = binance.get_market_sell_orders()
    print(f'Market_buys: {len(market_buy_orders)}\nMarket_sells: {len(market_sell_orders)}')
    time.sleep(0.1)



exit()
bitmex = bm.BitMEX(base_url=settings.BASE_URL, symbol=settings.SYMBOL,
                                    apiKey=settings.API_KEY, apiSecret=settings.API_SECRET,
                                    orderIDPrefix=settings.ORDERID_PREFIX, postOnly=settings.POST_ONLY,
                                    timeout=settings.TIMEOUT)
try:
    previous_id = None
    while True:
        inst = bitmex.recent_trades()
        current_id = inst[0]["trdMatchID"]
        if current_id != previous_id:
            print(json.dumps(inst, indent=2))
            previous_id = current_id
        time.sleep(5)
except Exception as err:
    print(f'Error: {err}')



symbol = "BTC-USD"
hyper = HyperLiquidConnector(testnet=True, symbol=symbol)
state = hyper.position()

print(state)


logging.basicConfig(level=logging.INFO)

hl = HyperLiquidWebsocket(symbol='ETH-USD', testnet=True)
hl.connect()
time.sleep(2)

# Subscribe just once to l2 book
logging.info("Subscribing to L2 book...")
hl.subscribe_l2_book()

while True:
    logging.info("Getting L2 book data...")
    book = hl.get_l2_book_data()
    time.sleep(2)
    
    if book:
        print(book)
    else:
        logging.warning("No book data received.")
        
    time.sleep(2)
