import logging
import time
import threading
import websocket
import json

### """WEBSOCKET DOES NOT WORK""" ###

class DIFXWebsocket:
    """WebSocket object for DIFX."""

    def __init__(self, symbol):
        self.logger = logging.getLogger('root')
        self.ws = None
        self.thread = None
        self.symbol = symbol
        self.data = {}
        self.endpoint = "wss://api-v2.difx.com"  # Main endpoint

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

    @property
    def is_alive(self):
        return self.ws and self.ws.sock and self.ws.sock.connected

    def exit(self):
        """Call this to exit - will close WebSocket."""
        if self.is_alive:
            self.ws.close()

    def send_command(self, command):
        """Send a raw command."""
        if self.is_alive:
            self.ws.send(json.dumps(command))
        else:
            self.logger.error("Socket is not connected.")

    def on_message(self, ws, message):
        """Handler for parsing WS messages."""
        message = json.loads(message)
        event = message.get('event')
        if event:
            self.data[event] = message  # Store the data by event type
        self.logger.info(message)

    def on_error(self, ws, error):
        """Called on fatal websocket errors."""
        self.logger.error(f"Error: {error}")

    def on_close(self, ws, *args, **kwargs):
        """Called on websocket close."""
        self.logger.info('Websocket Closed')

    def on_open(self, ws):
        """Called when the WS opens."""
        self.logger.info("Websocket Opened.")

    def join_room(self, room):
        """Join a specific room."""
        self.send_command({
            "event": "join",
            "room": room
        })

    def listen_event(self, event_name):
        """Listen to a specific event."""
        self.send_command({
            "event": event_name
        })

    def get_order_book(self):
        """Fetch the order book."""
        room = self.symbol
        self.join_room(room)
        self.listen_event("orderbook_limited")
        time.sleep(1)
        return self.data.get("orderbook_limited", {})

    def get_price_changes(self):
        """Fetch price changes."""
        self.listen_event("prices")
        time.sleep(1)
        return self.data.get("prices", [])

    def get_trades(self):
        """Fetch the trades."""
        room = self.symbol
        self.join_room(room)
        self.listen_event("trades")
        time.sleep(1)
        return self.data.get("trades", [])