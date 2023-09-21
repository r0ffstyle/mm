import json
import logging
import threading
import websocket
import eth_account
from eth_account.signers.local import LocalAccount
import market_maker.auth.eth_keys as keys
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants


TESTNET_WS_URL = "wss://api.hyperliquid-testnet.xyz/ws"
MAINNET_WS_URL = "wss://api.hyperliquid.xyz/ws"

class HyperLiquidConnector:
    """HyperLiquid API Connector."""
    
    def __init__(self, testnet=True, symbol=None):
        self.url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
        self.symbol = symbol
        self.logger = logging.getLogger('root')
        
        # Init account
        self.account: LocalAccount = eth_account.Account.from_key(keys.ETH_SECRET)
        print("Running with account address:", self.account.address)
        
        self.info = Info(self.url, skip_ws=True)
        self.ex = Exchange(self.account, self.url)
        
    # Account methods
    def position(self):
        user_state = self.info.user_state(self.account.address)
        positions = [position["position"] for position in user_state["assetPositions"] if float(position["position"]["szi"]) != 0]
        
        if positions:
            print("Positions:")
            print('\n'.join(json.dumps(pos, indent=2) for pos in positions))
        else:
            print("No open positions")

    # Order methods
    def limit_order(self, is_buy, size, price):
        try:
            order_result = self.ex.order(self.symbol, is_buy, size, price, {"limit": {"tif": "Gtc"}})
            return order_result
        except Exception as err:
            print(f"Error: {err}")

    def market_order(self, is_buy, size, trigger_price):
        try:
            order_result = self.ex.order(self.symbol, is_buy, size, trigger_price,
                                         {"trigger": {"triggerPx": trigger_price, "isMarket": True, "tpsl": "tp"}})
            return order_result
        except Exception as err:
            print(f"Error: {err}")

    def cancel_order(self, order_id):
        try:
            if order_id:
                cancel_result = self.ex.cancel(self.symbol, order_id)
                return cancel_result
        except Exception as err:
            print(f"Invalid Order ID: {err}")

    # Info methods
    def get_exchange_metadata(self):
        return self.info.meta()

    def get_state(self):
        return self.info.user_state(self.account.address)

    def get_open_orders(self):
        return self.info.open_orders(self.account.address)

    def get_fills(self):
        return self.info.user_fills(self.account.address)

    def get_ticker(self):
        return self.info.all_mids().get(self.symbol, {})

    def get_l2_snapshot(self):
        return self.info.l2_snapshot(self.symbol)


class HyperLiquidWebsocket(HyperLiquidConnector):
    def __init__(self, symbol, testnet=True):
        super().__init__(testnet, symbol)
        self.endpoint = TESTNET_WS_URL if testnet else MAINNET_WS_URL
        self.ws = None
        self.thread = None
        self.data = {}
        self.exited = False
        self.ws_open = False
        
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
            channel = parsed_message.get('channel', '')
            data = parsed_message.get('data', {})
            handler_method_name = f'handle_{channel}'
            handler_method = getattr(self, handler_method_name, None)
            if handler_method:
                handler_method(data)
            else:
                self.logger.warning(f"No handler for channel: {channel}")
        except json.JSONDecodeError:
            self.logger.error(f"Failed to decode message: {message}")

    def on_error(self, ws, error):
        """Handle WS errors."""
        if not self.exited:
            self.logger.error(f"Error: {error}")
            self.ws_open = False  # Update WebSocket status on error
            raise websocket.WebSocketException(error)

    def on_close(self, ws):
        """Handle WS close."""
        self.logger.info(f'Websocket Closed at {self.endpoint}')
        self.ws_open = False  # Update WebSocket status on close

    def on_open(self, ws):
        """Handle WS open."""
        self.logger.info(f"Websocket Opened at {self.endpoint}")
        self.ws_open = True  # Update WebSocket status on open

    """
    Handle methods
    """
    def handle_error(self, data):
        self.logger.error(f"Error Channel Data: {data}")

    def handle_all_mids(self, data):
        # Process data of type AllMids
        self.data['allMids'] = data.get('mids', {})
        self.logger.debug(f"Updated allMids: {self.data['allMids']}")

    def handle_notification(self, data):
        # Process data of type Notification
        self.data['notification'] = data.get('notification', '')
        self.logger.debug(f"Notification: {self.data['notification']}")

    def handle_web_data(self, data):
        # Process data of type WebData
        self.data['webData'] = data
        self.logger.debug(f"WebData: {self.data['webData']}")

    def handle_candle(self, data):
        # Process data of type WsTrade[]
        self.data['candle'] = data
        self.logger.debug(f"Candle Data: {self.data['candle']}")

    def handle_l2Book(self, data):
        """Handle data from the l2Book channel."""
        processed_data = []
        for level in data.get('levels', []):
            processed_level = []
            for order in level:
                processed_order = {
                    'price': order.get('price', order.get('px', 0)),
                    'size': order.get('size', order.get('sz', 0)),  
                    'number': order.get('number', order.get('n', 0))
                }
                processed_level.append(processed_order)
            processed_data.append(processed_level)
            
        self.data['l2Book'] = processed_data
        self.logger.debug(f"l2Book Data: {self.data['l2Book']}")


    def handle_user_events(self, data):
        # Process data of type UserEvents
        self.data['userEvents'] = data
        self.logger.debug(f"User Events Data: {self.data['userEvents']}")

    """
    Subscription methods
    """
    def subscribe(self, subscription_type, **kwargs):
        message = {"method": "subscribe", "subscription": {"type": subscription_type, **kwargs}}
        self.logger.debug(f"Sending Subscription Message: {message}")  # Added logging
        self.send_command(message)

    def subscribe_to_user_events(self, user_address):
        self.subscribe("userEvents", user=user_address)

    def subscribe(self, subscription_type, **kwargs):
        message = {"method": "subscribe", "subscription": {"type": subscription_type, **kwargs}}
        self.send_command(message)

    def subscribe_to_all_mids(self):
        self.subscribe("allMids")

    def subscribe_to_notification(self, user_address):
        self.subscribe("notification", user=user_address)

    def subscribe_to_web_data(self, user_address):
        self.subscribe("webData", user=user_address)

    def subscribe_to_candle(self, candle_interval):
        self.subscribe("candle", coin=self.symbol, interval=candle_interval)

    def subscribe_to_l2_book(self):
        self.subscribe("l2Book", coin=self.symbol)

    def subscribe_to_trades(self):
        self.subscribe("trades", coin=self.symbol)

    def subscribe_to_order_updates(self, user_address):
        self.subscribe("orderUpdates", user=user_address)

    def unsubscribe(self, subscription_type, **kwargs):
        message = {"method": "unsubscribe", "subscription": {"type": subscription_type, **kwargs}}
        self.send_command(message)



# def main():
#     symbol = "BTCUSD"  # replace with the desired symbol
#     ws_connector = HyperLiquidWebsocket(symbol, testnet=True)
#     ws_connector.connect()
    
#     import time
#     time.sleep(5)  # Wait for the connection to establish
    
#     ws_connector.subscribe_to_l2_book()
#     ws_connector.subscribe_to_trades()
#     time.sleep(30)  # Run for a while to receive messages
    
#     ws_connector.unsubscribe("l2Book", coin=symbol)
#     ws_connector.unsubscribe("trades", coin=symbol)
#     ws_connector.exit()


# if __name__ == "__main__":
#     main()
