import json
import logging
import threading
import websocket

import market_maker.auth.eth_keys as keys
import eth_account
from eth_account.signers.local import LocalAccount

from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants



class HyperLiquidConnector:
    """HyperLiquid API Connector."""
    def __init__(self, testnet=True, symbol=None):
        self.url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
        self.symbol = symbol
        self.logger = logging.getLogger('root')

        # Init account
        self.account: LocalAccount = eth_account.Account.from_key(keys.ETH_SECRET)
        print("Running with account address:", self.account.address)
        self.info = Info(constants.TESTNET_API_URL, skip_ws=True)
        # Init exchange
        self.ex = Exchange(self.account, self.url)

    """Account methods"""
    def position(self):
        user_state = self.info.user_state(self.account.address)
        positions = []
        for position in user_state["assetPositions"]:
            if float(position["position"]["szi"]) != 0:
                positions.append(position["position"])
        if len(positions) > 0:
            print("positions:")
            for position in positions:
                print(json.dumps(position, indent=2))
        else:
            print("no open positions")

    """Order methods"""
    def limit_order(self, is_buy, size, price):
        try:
            order_result = self.ex.order(self.symbol, is_buy, size, price, {"limit": {"tif": "Gtc"}}) # Gtc = Good till cancelled
            return order_result
        except Exception as err:
            print(f"Error: {err}")
    
    def market_order(self, is_buy, size, trigger_price):
        order_result = self.ex.order(self.symbol, is_buy, size, trigger_price, {"trigger": {"triggerPx": trigger_price, "isMarket": True, "tpsl": "tp"}}) # Trigger price should be best price at direction
        return order_result
    
    def cancel_order(self, order_id):
        if order_id:
            try:
                cancel_result = self.ex.cancel(self.symbol, order_id)
                return cancel_result
            except Exception as err:
                print(f"Invalid Order ID: {err}")

    """Info methods"""
    def get_exchange_metadata(self):
        return self.info.meta()

    def get_state(self):
        return self.info.user_state(self.account.address)

    def get_open_orders(self):
        return self.info.open_orders(self.account.address)

    def get_fills(self):
        return self.info.user_fills(self.account.address)
    
    def get_ticker(self):
        return self.info.all_mids()[self.symbol]
    
    def get_l2_snapshot(self):
        return self.info.l2_snapshot(self.symbol)


class HyperLiquidWebsocket(HyperLiquidConnector):
    def __init__(self, symbol, testnet=True):
        super().__init__(testnet, symbol)
        
        # Websocket URL
        self.endpoint = "wss://api.hyperliquid-testnet.xyz/ws" if testnet else "wss://api.hyperliquid.xyz/ws"
        self.ws = None
        self.thread = None
        self.data = {}
        self.exited = False

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
        """Exit and close WebSocket."""
        self.exited = True
        self.ws.close()

    def send_command(self, command):
        """Send a raw command."""
        self.ws.send(json.dumps(command))

    def on_message(self, ws, message):
        try:
            parsed_message = json.loads(message)
            # Handle the parsed_message as needed
        except json.JSONDecodeError:
            self.logger.error(f"Failed to decode message: {message}")
            # If you want to process non-JSON messages, you can do it here.

    def on_error(self, ws, error):
        """Handle WS errors."""
        if not self.exited:
            self.logger.error("Error : %s" % error)
            raise websocket.WebSocketException(error)

    def on_close(self, ws):
        """Handle WS close."""
        self.logger.info('Websocket Closed')

    def on_open(self, ws):
        """Handle WS open."""
        self.logger.info("Websocket Opened.")



    """Subscription Methods"""
    def subscribe_all_mids(self):
        message = {
            "method": "subscribe",
            "subscription": {
                "type": "allMids"
            }
        }
        self.send_command(message)

    def subscribe_notification(self, user_address):
        message = {
            "method": "subscribe",
            "subscription": {
                "type": "notification",
                "user": user_address
            }
        }
        self.send_command(message)

    def subscribe_web_data(self, user_address):
        message = {
            "method": "subscribe",
            "subscription": {
                "type": "webData",
                "user": user_address
            }
        }
        self.send_command(message)

    def subscribe_candle(self, candle_interval):
        message = {
            "method": "subscribe",
            "subscription": {
                "type": "candle",
                "coin": self.symbol,
                "interval": candle_interval
            }
        }
        self.send_command(message)

    def subscribe_l2_book(self):
        message = {
            "method": "subscribe",
            "subscription": {
                "type": "l2Book",
                "coin": self.symbol
            }
        }
        self.send_command(message)

    def subscribe_trades(self):
        message = {
            "method": "subscribe",
            "subscription": {
                "type": "trades",
                "coin": self.symbol
            }
        }
        self.send_command(message)

    def subscribe_order_updates(self, user_address):
        message = {
            "method": "subscribe",
            "subscription": {
                "type": "orderUpdates",
                "user": user_address
            }
        }
        self.send_command(message)

    # Unsubscribe Method

    def unsubscribe(self, subscription_type, **kwargs):
        message = {
            "method": "unsubscribe",
            "subscription": {
                "type": subscription_type,
                **kwargs
            }
        }
        self.send_command(message)

    # Handling Incoming Data (based on the data type)

    def handle_all_mids(self, data):
        # Process data of type AllMids
        pass

    def handle_notification(self, data):
        # Process data of type Notification
        pass

    def handle_web_data(self, data):
        # Process data of type WebData
        pass

    def handle_trade_data(self, data):
        # Process data of type WsTrade[]
        pass

    def handle_l2_book_data(self, data):
        # Store the data of type WsBook
        self.l2_book = data

    def get_l2_book_data(self):
        return getattr(self, 'l2_book', None)