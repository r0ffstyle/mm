import websocket
import json
import threading
import time
import numpy as np


class BinanceWebSocket():
    def __init__(self, symbols, streams, callback):
        self.ws = None
        self.latest_data = {}
        self.latest_trade_data = None
        self.lock = threading.Lock()
        self.callback = callback

        stream_path = "/".join([f"{symbol.lower()}@{stream}" for symbol, stream in zip(symbols, streams)])

        self.connect(f"wss://stream.binance.com:9443/stream?streams={stream_path}")

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
        streams = ["depth20@100ms", "ticker", "trade"]
        self.ws = BinanceWebSocket([self.symbol]*len(streams), streams, self.on_ticker_message)

    def on_ticker_message(self, ws, message):
        price = self.get_ticker_price()
        if price:  # 'c' is the field for the latest price in Binance's ticker data
            if len(self.prices) == self.max_prices_length:
                self.prices.pop(0)  # Remove the oldest price
            self.prices.append((time.time(), price))
        
        trade = self.get_latest_trade()
        if trade:
            if trade['m']:  # If the buyer is the market maker, it's a market sell order
                self.market_sell_orders.append(trade)
            else:  # If the buyer is the market taker, it's a market buy order
                self.market_buy_orders.append(trade)

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
        """Calculate the past 5min realized volatility"""
        five_min_ago = time.time() - 300
        recent_prices = [price for timestamp, price in self.prices if timestamp >= five_min_ago]
        if len(recent_prices) < 2:  # Not enough data to calculate volatility
            return None

        log_returns = [np.log(later / earlier) for earlier, later in zip(recent_prices[:-1], recent_prices[1:])]
        return np.std(log_returns) * 100 # Percentage

    def get_market_buy_orders(self):
        """Get recent market buy orders"""
        return self.market_buy_orders

    def get_market_sell_orders(self):
        """Get recent market sell orders"""
        return self.market_sell_orders
