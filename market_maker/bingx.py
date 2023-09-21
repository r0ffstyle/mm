"""
    The channel id determines what room we subscribe to (market depth, k-line, ...)
    See documentation, the channel is built differently for PERP
"""

import websocket
import json
import threading
import time
import gzip
import io
import os
import requests
import hmac
from hashlib import sha256

class BingXWebSocket():
    def __init__(self, symbol, data_type, callback):
        self.ws = None
        self.latest_data = {}
        self.lock = threading.Lock()
        self.callback = callback
        self.symbol = symbol
        self.data_type = data_type
        self.connect(f"wss://open-api-ws.bingx.com/market")

    def decompress_data(self, binary_data):
        buffer = io.BytesIO(binary_data)
        with gzip.GzipFile(fileobj=buffer, mode='rb') as f:
            return f.read().decode('utf-8')

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
        decompressed_data = self.decompress_data(message)
        data = json.loads(decompressed_data)

        # Handling Ping-Pong mechanism
        if decompressed_data == "Ping":
            ws.send("Pong")
            return

        # Handle listenKey expiration
        if data.get('e') == 'listenKeyExpired':
            print("ListenKey has expired. Need to refresh and reconnect!")
            # Add logic here to fetch a new listenKey and restart the WebSocket connection.
            return
        
        with self.lock:
            self.latest_data = data
        self.callback(data)

    def on_error(self, ws, error):
        print(f"Error: {error}")

    def on_close(self, ws):
        print("Connection closed")

    def on_open(self, ws):
        print("Connection opened")
        CHANNEL = {"id": "e745cd6d-d0f6-4a70-8d5a-043e4c741b40", "dataType": f"{self.symbol}@{self.data_type}"}
        subStr = json.dumps(CHANNEL)
        ws.send(subStr)
        print("Subscribed to :", subStr)

    def get_latest_data(self):
        with self.lock:
            return self.latest_data


class BingX():
    def __init__(self, symbol, data_type=None):
        self.BASE_URL = "https://open-api.bingx.com"
        self.api_key = os.getenv("bingx_key")
        self.api_secret = os.getenv("bingx_secret")
        self.symbol = symbol
        self.data_type = data_type

        if self.data_type:
            self.ws = BingXWebSocket(self.symbol, self.data_type, self.on_message)
        else:
            self.ws = None

    def on_message(self, data):
        # Process the data as per your needs. For now, just printing it.
        # print(data)
        pass

    def get_order_book(self):
        """Fetch the orderbook from the WebSocket"""
        if not self.ws:
            raise ValueError("Provide datatype to initialize WebSocket")
        return self.ws.get_latest_data()
    
    def get_latest_trade(self):
        """Fetch the latest trade details from the WebSocket
            Returns:
                e: Event type
                E: Event time
                s: Trading pair
                t: Transaction ID
                p: Transaction price
                q: Executed quantity
                T: Transaction time
                m: True is sell MO and False is buy MO
        """
        if not self.ws:
            raise ValueError("Provide datatype to initialize WebSocket")
        return self.ws.get_latest_data()
    

    #################
    # ORDER METHODS #
    #################
    

    @staticmethod
    def _generate_signature(secret_key, payload):
        return hmac.new(secret_key.encode("utf-8"), payload.encode("utf-8"), digestmod=sha256).hexdigest()

    def _send_order(self, order_params):
        path = '/openApi/swap/v2/trade/order'
        method = "POST"
        
        # Convert the order_params dictionary to a parameter string
        params_string = "&".join(f"{key}={value}" for key, value in order_params.items())
        
        # Append timestamp
        timestamp = int(time.time() * 1000)  # Current time in milliseconds
        params_string += f"&timestamp={timestamp}"
        
        # Generate the signature
        signature = self._generate_signature(self.api_secret, params_string)
        url = f"{self.BASE_URL}{path}?{params_string}&signature={signature}"  # Add signature as a query parameter
        
        headers = {
            "X-BX-APIKEY": self.api_key
        }

        response = requests.post(url, headers=headers, data=order_params)
        return response.json()
    
    """
    TEST ORDER METHODS
    """

    def _trade_order_test(self, side, order_type, **kwargs):
        position_side = "LONG" if side == "BUY" else "SHORT"
        params = {
            "symbol": self.symbol,
            "side": side,  # This should be either "BUY" or "SELL"
            "positionSide": position_side,  # This will be either "LONG" or "SHORT"
            "type": order_type,
            "timestamp": int(time.time() * 1000)
        }
        params.update(kwargs)
        return self._send_request("POST", "/openApi/swap/v2/trade/order/test", params)

    def test_limit_order(self, side, quantity, price, **kwargs):
        return self._trade_order_test(side, "LIMIT", quantity=quantity, price=price, **kwargs)

    def test_market_order(self, side, quantity, price, **kwargs):
        return self._trade_order_test(side, "MARKET", quantity=quantity, price=price, **kwargs)
    
    """
    LIVE ORDER METHODS
    """

    def place_limit_order(self, direction, price, quantity):
        # Here, direction should be either 'BID' or 'ASK'
        order_params = {
            "symbol": self.symbol,
            "side": direction,
            "positionSide": "LONG" if direction == 'BID' else 'SHORT',
            "type": "LIMIT",
            "price": price,
            "quantity": quantity
        }
        return self._send_order(order_params)

    def place_market_order(self, direction, quantity):
        # Here, direction should be either 'BID' or 'ASK'
        order_params = {
            "symbol": self.symbol,
            "side": direction,
            "positionSide": "LONG" if direction == 'BID' else 'SHORT',
            "type": "MARKET",
            "quantity": quantity
        }
        return self._send_order(order_params)

    def _parse_param(self, params):
        sorted_keys = sorted(params)
        params_str = "&".join(["%s=%s" % (k, params[k]) for k in sorted_keys])
        return params_str + "&timestamp=" + str(int(time.time() * 1000))

    def _send_request(self, method, path, params, payload={}):
        params_str = self._parse_param(params)
        signature = self._generate_signature(self.api_secret, params_str)
        
        url = f"{self.BASE_URL}{path}?{params_str}&signature={signature}"
        headers = {
            'X-BX-APIKEY': self.api_key,
        }

        response = requests.request(method, url, headers=headers, data=payload)
        return response.json()

    def cancel_order(self, order_id=None, client_order_id=None, recv_window=0):
        params = {
            "symbol": self.symbol,
            "recvWindow": recv_window
        }
        if order_id:
            params["orderId"] = order_id
        if client_order_id:
            params["clientOrderID"] = client_order_id

        return self._send_request("DELETE", "/openApi/swap/v2/trade/order", params)

    def cancel_all_orders(self, recv_window=0):
        params = {
            "symbol": self.symbol,
            "recvWindow": recv_window
        }

        return self._send_request("DELETE", "/openApi/swap/v2/trade/allOpenOrders", params)
    

    ###################
    # ACCOUNT METHODS #
    ###################


    def get_balance(self, recv_window=0):
        params = {
            "recvWindow": recv_window
        }
        return self._send_request("GET", "/openApi/swap/v2/user/balance", params)

    def get_positions(self, recv_window=0):
        params = {
            "symbol": self.symbol,
            "recvWindow": recv_window
        }
        return self._send_request("GET", "/openApi/swap/v2/user/positions", params)


# # Testing
# if __name__ == "__main__":
#     symbol = "BTC-USDT"
    
#     bingx = BingX(symbol=symbol)

#     # Testing the trade_order_test method for limit order
#     result = bingx.test_limit_order("BUY", 0.1, 26503)
#     print("Test Limit Order Result:", result)
    
#     # Testing the trade_order_test method for market order
#     result = bingx.test_market_order("SELL", 0.1, 26503)
#     print("Test Market Order Result:", result)

# Testing
if __name__ == "__main__":
    symbol = "BIFI-USDT"
    # data_type = "depth"
    data_type = "trade"

    bingx = BingX(symbol=symbol, data_type=data_type)
    time.sleep(2) # Give time to establish connection

    while True:
        #book = bingx.get_latest_trade()
        book = bingx.get_latest_trade()
        print(book)
        time.sleep(1) # Do not overflow the WebSocket