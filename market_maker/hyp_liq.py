import json
import logging
import threading
import websocket
import time
import numpy as np
import pickle

import eth_account
from eth_account.signers.local import LocalAccount
import auth.eth_keys as keys

from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

MAINNET_WS_URL = 'wss://api.hyperliquid.xyz/ws'
TESTNET_WS_URL = 'wss://api.hyperliquid-testnet.xyz/ws'

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
            return positions
        else:
            return 0
        
    """
    Orders
    """
    def limit_order(self, is_buy, size, price):
        try:
            response = self.ex.order(self.symbol, is_buy, size, price, {"limit": {"tif": "Gtc"}})
            return response
        except Exception as err:
            print(f"Error: {err}")
            return None

    def market_order(self, is_buy, size, trigger_price):
        try:
            return self.ex.order(self.symbol, is_buy, size, trigger_price,
                                 {"trigger": {"triggerPx": trigger_price, "isMarket": True, "tpsl": "tp"}})
        except Exception as err:
            print(f"Error: {err}")

    def cancel_order(self, order_id):
        try:
            return self.ex.cancel(self.symbol, order_id)
        except Exception as err:
            print(f"Invalid Order ID: {err}")
    
    def cancel_all_orders(self):
        """Cancel all open orders."""
        try:
            open_orders = self.get_open_orders() # Fetch open orders
            cancel_requests = [{"coin": order['coin'], "oid": order['oid']} for order in open_orders]
            if cancel_requests:
                self.ex.bulk_cancel(cancel_requests) # Cancel all orders
                self.logger.info("All orders cancelled.")
            else:
                self.logger.info("No open orders to cancel.")
        except Exception as err:
            self.logger.error(f"Error cancelling all orders: {err}")
    
    def modify_order(self, order_id, is_buy, size, price, order_type, tif=None, trigger_price=None):
        """Amend order."""    
        # Set a default value for tif if None is provided
        if not tif:
            tif = "Gtc"  # Default to 'Good Until Canceled'

        # Prepare the order parameters
        if order_type == "limit":
            order_params = {"limit": {"tif": tif}}
        elif order_type == "trigger":
            order_params = {
                "trigger": {
                    "triggerPx": trigger_price,
                    "isMarket": True,
                    "tpsl": "tp"
                }
            }
        else:
            self.logger.error(f"Invalid order type: {order_type}")
            raise ValueError(f"Invalid order type: {order_type}")

        return self.ex.modify_order(order_id, self.symbol, is_buy, size, price, order_params)

    """
    Getters
    """
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
        self.open_orders = []
        # Lock for thread safety
        self.lock = threading.Lock()
        self.order_fill_callback = None
        self.command_queue = []
        
    def connect(self):
        """Connect to the websocket in a thread."""
        if self.ws_open:
            self.logger.debug("WebSocket is already open.")
            return

        self.logger.debug("Starting WebSocket thread.")
        self.ws = websocket.WebSocketApp(
            self.endpoint,
            on_message=self.on_message,
            on_close=self.on_close,
            on_open=self.on_open,
            on_error=self.on_error
        )
        self.thread = threading.Thread(target=lambda: self.ws.run_forever())
        self.thread.daemon = True
        self.thread.start()
        time.sleep(1)  # Give time for the connection to establish

    def exit(self):
        """Exit and close WebSocket."""
        self.exited = True
        self.ws.close()

    def send_command(self, command):
        """Send a raw command."""
        if not self.ws.sock or not self.ws_open:
            self.logger.debug("WebSocket is not open. Queuing command.")
            self.command_queue.append(command)
            return
        self.ws.send(json.dumps(command))

    def on_message(self, ws, message):
        try:
            parsed_message = json.loads(message)
            channel = parsed_message.get('channel', '')
            data = parsed_message.get('data', {})
            if channel == 'pong':
                self.logger.debug("Heartbeat received.")
            else:
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

    def on_close(self, ws, close_status_code, close_msg):
        """Handle WS close."""
        self.logger.info(f'Websocket Closed at {self.endpoint}')
        self.logger.info(f"Close status code: {close_status_code}")
        self.logger.info(f"Close message: {close_msg}")
        self.ws_open = False  # Update WebSocket status on close

    def on_open(self, ws):
        """Handle WS open."""
        self.logger.info(f"Websocket Opened at {self.endpoint}")
        self.ws_open = True  # Update WebSocket status on open

        # Send any queued commands
        while self.command_queue:
            command = self.command_queue.pop(0)
            self.ws.send(json.dumps(command))

        # Subscribe to channels
        self.subscribe_to_userFills(self.account.address)
        # Add other subscriptions if needed

    def set_order_fill_callback(self, callback):
        """Set the callback function to be called when an order is filled."""
        self.order_fill_callback = callback

    """
    Handlers
    """
    def handle_subscriptionResponse(self, data):
        # Process data of type subscriptionResponse
        self.logger.debug(f"Subscription Response: {data}")

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
        
    def handle_trades(self, data):
        """Handle data from the trades channel."""
        self.data['trades'] = data
        self.logger.debug(f"Trades Data: {self.data['trades']}")
    
    def handle_orderUpdates(self, data):
        """Handle data from the orderUpdates channel."""
        with self.lock:
            for order_update in data:
                order = order_update.get('order', {})
                status = order_update.get('status', '')
                oid = order.get('oid')

                # Remove the order if it is filled or canceled
                if status in ['filled', 'canceled', 'rejected', 'marginCanceled']:
                    self.open_orders = [o for o in self.open_orders if o['oid'] != oid]
                    # Check if the order was filled
                    if status == 'filled':
                        # If we have a callback set, call it
                        if self.order_fill_callback:
                            # Provide the order update info to the callback
                            self.order_fill_callback(order_update)
                else:
                    # Update existing order or add new order
                    existing_order = next((o for o in self.open_orders if o['oid'] == oid), None)
                    if existing_order:
                        existing_order.update(order)
                        existing_order['status'] = status
                        existing_order['statusTimestamp'] = order_update.get('statusTimestamp')
                    else:
                        # Add new order
                        order['status'] = status
                        order['statusTimestamp'] = order_update.get('statusTimestamp')
                        self.open_orders.append(order)

        self.logger.debug(f"Open Orders Updated: {self.open_orders}")

    def get_cached_open_orders(self):
        """Return the current list of open orders."""
        with self.lock:
            return list(self.open_orders)

    def handle_userFills(self, data):
        """Process data from the userFills channel."""
        with self.lock:
            fills = data.get('fills', [])
            for fill in fills:
                # Prepare the order update data structure
                order_update = {
                    'order': {
                        'oid': fill['oid'],
                        'limitPx': fill['px'],
                        'sz': fill['sz'],  # Size filled in this fill
                        'origSz': fill['sz'],  # Total size filled in this fill
                        'isBuy': fill['side'] == 'B',
                        'timestamp': fill['time']
                    },
                    'status': 'fill',  # Use 'fill' to indicate a fill event
                    'statusTimestamp': fill['time']
                }

                # If we have a callback set, call it
                if self.order_fill_callback:
                    # Provide the order update info to the callback
                    self.order_fill_callback(order_update)

        self.logger.debug(f"User Fills: {fills}")

    
    def send_heartbeat(self):
        self.ws.send(json.dumps({"method": "ping"}))

    def maintain_connection(self):
        # Periodically check and maintain the connection
        if not self.ws_open:
            self.connect()
        else:
            self.send_heartbeat()

    def handle_user_events(self, data):
        # Process data of type UserEvents
        for event in data:
            if event['type'] == 'order':
                self.logger.debug(f"Order Event: {event}")
            elif event['type'] == 'trade':
                self.logger.debug(f"Trade Event: {event}")
            # Add more event types as needed
        self.data['userEvents'] = data

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

    def subscribe_to_orderUpdates(self, user_address):
        self.subscribe("orderUpdates", user=user_address)

    def subscribe_to_userFills(self, user_address):
        self.subscribe("userFills", user=user_address)

    def unsubscribe(self, subscription_type, **kwargs):
        message = {"method": "unsubscribe", "subscription": {"type": subscription_type, **kwargs}}
        self.send_command(message)

    """
    Getters
    """
    def get_l2_book(self):
        """Return the latest L2 book data."""
        return self.data.get('l2Book', None)
    
    def get_trades(self):
        return self.data.get('trades', None)
    
    def get_volatility(self):
        """
        Calculate the volatility of a series of prices.
        """
        # Get the latest trade prices
        trades = self.get_trades()
        prices = [float(trade['px']) for trade in trades]

        if len(prices) < 2:
            raise ValueError("At least two prices are required to calculate volatility.")

        log_returns = np.diff(np.log(prices))
        volatility = np.std(log_returns)
        return volatility
    
    def get_orderUpdates(self):
        """Get the latest order updates."""
        return self.data.get('orderUpdates', None)

    def get_userFills(self):
        """Return the latest user fills data."""
        return self.data.get('userFills', None)


def main():
    symbol = "WLD" 
    testnet = True

    # Instantiate the HyperLiquidWebsocket class
    hl_websocket = HyperLiquidWebsocket(symbol, testnet)
    hl_websocket.connect()

    hl_websocket.subscribe_to_l2_book()
    hl_websocket.subscribe_to_trades()

    orders = hl_websocket.get_open_orders()
    meta = hl_websocket.get_exchange_metadata()
    try:
        while True:
            """L2 book"""
            # l2_book = hl_websocket.get_l2_book()
            # print(l2_book)
            # print(l2_book[0][0]['price'])
            # print(l2_book[0][0]['size'])
            # print(type(l2_book[0][0]['price'])) # String
            # print(type(l2_book[0][0]['size'])) # String

            # """Trades"""
            # trades = hl_websocket.get_trades()
            # print(trades)

            """Volatility"""
            # volatility = hl_websocket.get_volatility()
            # if volatility:
            #     print(volatility)
            #     print(f"Volatiltiy: {round(volatility, 4) * 100}%")
            print('Placing order...')
            order = {'is_buy': True, 'size': 10, 'price': 2.6270}
            order = hl_websocket.limit_order(order["is_buy"], order["size"], order['price'])
            print(order)

            """Metadata"""
            # print(meta)
            time.sleep(2)


    except KeyboardInterrupt:
        print("Exiting...")
        hl_websocket.exit()

if __name__ == "__main__":
    main()