import logging
import time
import threading
import websocket
import json

class PoloniexWebsocket:
    """WebSocket object for Poloniex."""
    def __init__(self, symbol, api_key=None, api_secret=None):
        self.logger = logging.getLogger('root')
        self.endpoint = "wss://ws.poloniex.com/ws/public"  # If using private data, use private endpoint
        self.api_key = api_key
        self.api_secret = api_secret
        self.ws = None
        self.thread = None
        self.symbol = symbol
        self.data = {}

    def connect(self):
        """Connect to the websocket in a thread."""
        self.logger.debug("Starting WebSocket thread.")
        self.ws = websocket.WebSocketApp(self.endpoint,
                                         on_message=self.on_message,
                                         on_close=self.on_close,
                                         on_open=self.on_open,
                                         on_error=self.on_error)

        self.thread = threading.Thread(target=lambda: self.ws.run_forever())
        self.thread.daemon = True
        self.thread.start()

    def exit(self):
        """Call this to exit - will close WebSocket."""
        self.exited = True
        self.ws.close()

    def send_command(self, command):
        """Send a raw command."""
        self.ws.send(json.dumps(command))

    def on_message(self, ws, message):
        """Handler for parsing WS messages."""
        message = json.loads(message)
        self.data[message['channel']] = message  # Store the data by channel
        self.logger.info(message)

    def on_error(self, ws, error):
        """Called on fatal websocket errors. We exit on these."""
        if not self.exited:
            self.logger.error("Error : %s" % error)
            raise websocket.WebSocketException(error)

    def on_close(self, ws):
        """Called on websocket close."""
        self.logger.info('Websocket Closed')

    def on_open(self, ws):
        """Called when the WS opens."""
        self.logger.info("Websocket Opened.")
        
    def ping(self):
        """Ping the server to keep the connection alive"""
        self.send_command({"event": "ping"})

    def subscribe(self, channel, symbols):
        """Subscribe to a channel for given symbols"""
        self.send_command({
            "event": "subscribe",
            "channel": [channel],
            "symbols": symbols
        })

    def unsubscribe(self, channel, symbols):
        """Unsubscribe from a channel for given symbols"""
        self.send_command({
            "event": "unsubscribe",
            "channel": [channel],
            "symbols": symbols
        })

    def unsubscribe_all(self):
        """Unsubscribe from all channels"""
        self.send_command({
            "event": "unsubscribe_all"
        })

    def list_subscriptions(self):
        """List all current subscriptions"""
        self.send_command({
            "event": "list_subscriptions"
        })

    def reset(self):
        """Hard reset: close and reopen websocket."""
        self.exit()
        self.connect()

    def get_order_book(self):
        """Fetch the order book"""
        book = None
        while True:
            book =  self.subscribe(channel="book_lv2", symbols=[self.symbol])
            if book is not None and book['data']['action'] == 'snapshot':
                time.sleep(1)
                return book
            else:
                time.sleep(0.1)
                continue

    def get_ticker(self):
        """Fetch the ticker"""
        ticker = self.subscribe(channel="ticker", symbols=[self.symbol])
        time.sleep(1)
        if ticker is not None:
            return ticker