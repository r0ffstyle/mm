import websocket
import json
import threading


class Binance():
    def __init__(self, symbol):
        self.symbol = symbol.lower()
        self.ws = None
        self.latest_data = None
        self.lock = threading.Lock()

    def connect(self):
        self.ws = websocket.WebSocketApp(f"wss://stream.binance.com:9443/ws/{self.symbol}@depth20@100ms",
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
            self.latest_data = data

    def on_error(self, ws, error):
        print(f"Error: {error}")

    def on_close(self, ws):
        print("Connection closed")

    def on_open(self, ws):
        print("Connection opened")


    def get_order_book(self):
        with self.lock:
            return self.latest_data
