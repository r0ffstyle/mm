import websocket
import json
import threading
import time
import numpy as np
from binance.client import Client
import os

api_key = os.getenv("binance_key")
api_secret = os.getenv("binance_secret")

class BinanceWebSocket():
    """Websocket"""
    def __init__(self, symbols, streams, callback):
        self.ws = None
        self.latest_data = {}
        self.latest_trade_data = None
        self.lock = threading.Lock()
        self.callback = callback

        stream_path = "/".join([f"{symbol.lower()}@{stream}" for symbol, stream in zip(symbols, streams)])

        self.connect(f"wss://stream.binance.com:9443/stream?streams={stream_path}")
        print(f"Binance initiated with {str(symbols[0]).upper()}")

    def connect(self, url):
        self.ws = websocket.WebSocketApp(url,
                                         on_message=self.on_message,
                                         on_error=self.on_error,
                                         on_close=self.on_close)

        self.ws.on_open = self.on_open
        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()

    def on_message(self, ws, message):
        data = json.loads(message)
        with self.lock:
            self.latest_data[data['stream']] = data['data']

        if 'ticker' in data['stream']:
            self.callback(self.ws, message)

        if 'trade' in data['stream']:
            self.latest_trade_data = data['data']

    def on_error(self, ws, error):
        print(f"Error: {error}")

    def on_close(self, ws):
        print("Connection closed")

    def on_open(self, ws):
        print("Connection opened")

    def get_latest_data(self, stream):
        with self.lock:
            return self.latest_data.get(stream)


class Binance():
    def __init__(self, symbol):
        self.symbol = symbol.lower()
        self.prices = []  # Store last 5 minutes prices. Assumes updates every second
        self.max_prices_length = 300
        self.market_buy_orders = []
        self.market_sell_orders = []
        self.historical_trades = []
        self.fetch_historical_trades_interval = 900  # Fetch every 15 minutes to deem the live trade significant
        self.last_historical_fetch_time = time.time()

        streams = ["depth20@100ms", "ticker", "trade"]
        self.ws = BinanceWebSocket([self.symbol]*len(streams), streams, self.on_ticker_message)
        self.client = Client(api_key, api_secret)

    def on_ticker_message(self, ws, message):
        price = self.get_ticker_price()
        if price:  # 'c' is the field for the latest price in Binance's ticker data
            if len(self.prices) == self.max_prices_length:
                self.prices.pop(0)  # Remove the oldest price
            self.prices.append((time.time(), price))
        
        # Check if time to fetch historical trades
        current_time = time.time()
        if current_time - self.last_historical_fetch_time >= self.fetch_historical_trades_interval:
            self.get_historical_trades()
            self.last_historical_fetch_time = current_time
        
        trade = self.get_latest_trade()
        if trade:
            if trade['m']:  # If the buyer is the market maker, it's a market sell order
                self.market_sell_orders.append(trade)
            else:  # If the buyer is the market taker, it's a market buy order
                self.market_buy_orders.append(trade)
    
    """
    Getters
    """
    def get_historical_trades(self):
        """Returns an array of qty's for the past 1000 trades"""
        try:
            historical_trades = self.client.get_historical_trades(symbol=self.symbol.upper(), limit=1000)
            return historical_trades
        except Exception as e:
            print(f"Error fetching historical trades: {e}")

    def get_order_book(self):
        """Fetch the orderbook"""
        return self.ws.get_latest_data(f"{self.symbol}@depth20@100ms")

    def get_ticker_price(self):
        """Fetch the latest ticker price"""
        ticker = self.ws.get_latest_data(f"{self.symbol}@ticker")
        price =  float(ticker["c"])
        return price
    
    def get_latest_trade(self):
        """Fetch the latest trade"""
        return self.ws.latest_trade_data

    def get_volatility(self):
        """
        Calculate the volatility of a series of prices.

        :param prices: A list or array of prices.
        :return: The estimated volatility.
        """
        # Get last 1000 historical trade prices
        trades = self.get_historical_trades()
        prices = [float(trade['price']) for trade in trades]

        if len(prices) < 2:
            raise ValueError("At least two prices are required to calculate volatility.")

        log_returns = np.diff(np.log(prices))
        volatility = np.std(log_returns)

        if np.isnan(volatility) or np.isinf(volatility):
            raise ValueError("Calculated volatility is not a valid number.")
        return volatility
    
def main():
    symbol = "WLDUSDT"
    binance = Binance(symbol)

    # Print the volatility
    volatility = binance.get_volatility()
    print(volatility)
    print(f"Volatiltiy: {round(volatility, 4) * 100}%")

    # Print historical trades
    historical_trades = binance.get_historical_trades()
    # Stream live trades
    print("\nStreaming Live Trades:")
    try:
        while True:
            book = binance.get_order_book()
            latest_trade = binance.get_latest_trade()
            if latest_trade:
                print(latest_trade['p'])
                print(type(latest_trade['p'][0]))
            if book:
                mid_price = (float(book['bids'][0][0]) + float(book['asks'][0][0])) / 2
                print(mid_price)
            time.sleep(1) 
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == "__main__":
    main()