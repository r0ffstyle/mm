from __future__ import absolute_import
from time import sleep
import sys
import os
from datetime import datetime
from os.path import getmtime
import random
import requests
import atexit
import signal
import numpy as np

import binance_api
import hyp_liq as HL

from settings import settings
from utils import log, constants, errors, math
import auth.eth_keys as keys

#
# Helpers
#
logger = log.setup_custom_logger('root')

symbol = "WLD"

class ExchangeInterface:
    def __init__(self, dry_run=False):
        """Connect to Binance"""
        self.binance_symbol = "WLDUSDT"
        self.binance = binance_api.Binance(self.binance_symbol)
        
        """Connect to HyperLiquid"""
        self.HL_symbol = symbol
        self.HL = HL.HyperLiquidWebsocket(self.HL_symbol, testnet=True)
        self.HL.connect()                                       # Establish WebSocket connection
        self.HL.subscribe_to_l2_book()                          # Subscribe to the l2Book channel
        self.HL.subscribe_to_trades()                           # Subscribe to the trades channel
        self.HL.subscribe_to_orderUpdates(keys.ETH_ADRESS)      # Subscribe to the orderUpdates channel
        self.HL.subscribe_to_userFills(keys.ETH_ADRESS)         # Subscribe to the userFills channel

        # Fetch initial open orders via HTTP API
        initial_open_orders = self.HL.get_open_orders()
        with self.HL.lock:
            self.HL.open_orders = initial_open_orders
        logger.info(f"Initial open orders fetched: {initial_open_orders}")

    
    """
    HyperLiquid Methods
    """
    def get_HL_order_book(self):
        """Fetch the order book on HyperLiquid"""
        try:
            return self.HL.get_l2_book()
        except Exception as e:
            logger.exception(f"Error fetching order book: {e}")
        
    def get_HL_volatility(self):
        """Fetch the volatility from HyperLiquid"""
        try:
            return self.HL.get_volatility()
        except Exception as e:
            logger.exception(f"Error fetching volatility: {e}")

    def get_HL_latest_trades(self):
        """Get the latest trades made on asset"""
        try:
            return self.HL.get_trades()
        except Exception as e:
            logger.exception(f"Error fetching trades: {e}")
    
    def get_HL_arrival_rate(self): # TODO
        """Get the arrival rate of market orders of HyperLiquid"""
        pass

    def get_open_HL_orders(self):
        """Get user's open orders from the cache."""
        try:
            return self.HL.get_cached_open_orders()
        except Exception as e:
            logger.error(f"Error fetching open orders from cache: {e}")
            return []
    
    def get_HL_order_updates(self):
        """Get users order updates."""
        try:
            return self.HL.get_orderUpdates()
        except Exception as e:
            logger.error(f"Error fetching order updates: {e}")
    
    def set_order_fill_callback(self, callback):
        """Set the order fill callback function."""
        try:
            self.HL.set_order_fill_callback(callback)
        except Exception as e:
            logger.error(f"Error setting order fill callback: {e}")

    def get_HL_user_fills(self):
        """Get users fills."""
        try:
            return self.HL.get_userFills()
        except Exception as e:
            logger.error(f"Error fetching users fills: {e}")

    def get_HL_positions(self):
        """Get user positions."""
        try:
            pos = self.HL.position()
            return float(pos[0]['szi']) if pos != 0 else 0
        except Exception as e:
            logger.error(f"Error fetching users positions: {e}")
    
    def place_HL_order(self, side, quantity, price):
        """Orderplacing"""
        try:
            is_buy = True if side == 'B' else False
            order = self.HL.limit_order(is_buy, quantity, price)
            logger.info(f"Placing order: {side} {quantity} @ {price}")
            return order
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None

    def cancel_HL_order(self, order_id):
        """Cancel single order"""
        try:
            return self.HL.cancel_order(order_id)
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")

    def cancel_all_HL_orders(self):
        """Cancel all orders on HyperLiquid."""
        try:
            # Call the cancel_all_orders method from the HyperLiquidConnector class
            self.HL.cancel_all_orders()
            logger.info("All HyperLiquid orders cancelled successfully.")
        except Exception as e:
            logger.error(f"Error cancelling all HyperLiquid orders: {e}")
    
    def amend_HL_order(self, order_id, is_buy, size, price, order_type):
        """Modify order on HyperLiquid"""
        try:
            if order_id is None or is_buy is None or size is None or price is None or order_type is None:
                raise ValueError("Amending failed because a None is returned")
            return self.HL.modify_order(order_id, is_buy, size, price, order_type)
        
        except Exception as e:
            logger.error(f"Error modifying order: {e}")
            return None

    def ensure_HL_ws_connected(self):
        """Make sure the connection to the HyperLiquid WebSocket stays open"""
        if not self.HL.ws_open:
            self.HL.connect()

            # Keep subscribing to WebSocket channels
            self.HL.subscribe_to_l2_book()
            self.HL.subscribe_to_trades()
            self.HL.subscribe_to_orderUpdates(keys.ETH_ADRESS)
            self.HL.subscribe_to_userFills(keys.ETH_ADRESS)

    """
    Binance methods
    """
    def get_binance_order_book(self):
        return self.binance.get_order_book()
    
    def get_binance_latest_trade(self):
        return self.binance.get_latest_trade()
    
    def get_binance_historical_trades(self):
        return self.binance.get_historical_trades()
    
    def get_5min_volatility(self):
        return self.binance.get_volatility()
    
    def get_market_buy_orders(self):
        return self.binance.get_market_buy_orders()
    
    def get_market_sell_orders(self):
        return self.binance.get_market_sell_orders()
    

class OrderManager:
    def __init__(self):
        self.exchange = ExchangeInterface(settings.DRY_RUN)
        atexit.register(self.exit)
        signal.signal(signal.SIGTERM, self.exit)

        logger.info("Using symbol %s." % self.exchange.HL_symbol)

        self.start_time = datetime.now()
        # self.reset()

    def reset(self):
        self.exchange.cancel_all_HL_orders()
        self.sanity_check() # TODO
        self.print_status()
        # Create orders and converge.
        self.place_orders()

    def print_status(self):
        """Print the current MM status."""
        pass

    ###
    # Orders
    ###

    def place_orders(self):
        """Create order items for use in convergence."""
        self.exchange.ensure_HL_ws_connected() # Ensure we are connected to the websocket
        buy_orders = []
        sell_orders = []
        # Orders are created/amended from outside-in. This approach minimizes amendments, as innermost orders (closer to market price and more likely to be filled) are adjusted last. It efficiently manages order placement, reducing unnecessary adjustments and aligning with market activity probabilities.
        for i in reversed(range(1, settings.ORDER_PAIRS + 1)):
            if not self.long_position_limit_exceeded():
                buy_orders.append(self.prepare_order(-i))
            if not self.short_position_limit_exceeded():
                sell_orders.append(self.prepare_order(i))

        return self.converge_orders(buy_orders, sell_orders)

    def prepare_order(self, index):
        """Create an order object."""
        if settings.RANDOM_ORDER_SIZE is True:
            quantity = random.randint(settings.MIN_ORDER_SIZE, settings.MAX_ORDER_SIZE)
            # Respect lot size
            quantity = round(quantity / settings.ORDER_STEP_SIZE) * settings.ORDER_STEP_SIZE
        else:
            quantity = settings.ORDER_START_SIZE + ((abs(index) - 1) * settings.ORDER_STEP_SIZE)

        price = self.get_price_offset(index)

        return {'price': price, 'orderQty': quantity, 'side': "B" if index < 0 else "A"}
    
    def converge_orders(self, buy_orders, sell_orders):
        """
        Converge the current orders with the desired orders.
        This method amends, creates, or cancels orders to match the desired state.

        Parameters:
        buy_orders (list): The list of desired buy orders.
        sell_orders (list): The list of desired sell orders.
        """
        # Initialize lists for order actions and match counters
        to_amend = []  # Orders to be amended
        to_create = []  # Orders to be created
        to_cancel = []  # Orders to be cancelled
        buys_matched = 0  # Counter for buy orders matched with existing orders
        sells_matched = 0  # Counter for sell orders matched with existing orders

        # Fetch the current open orders from HyperLiquid
        existing_orders = self.exchange.get_open_HL_orders()

        # Iterate through existing orders to check if they match the desired orders
        for order in existing_orders:
            try:
                # Determine if the existing order is a buy or sell order
                is_buy = order['side'] == 'B'
                desired_order = buy_orders[buys_matched] if is_buy else sell_orders[sells_matched]

                # Increment the matched order counter
                buys_matched += is_buy
                sells_matched += not is_buy

                # Check if the existing order needs to be amended (quantity or price change)
                if desired_order['orderQty'] != order['sz'] or \
                desired_order['price'] != order['limitPx']:
                    # Add the order to the amendment list
                    to_amend.append({
                        'orderID': order['oid'],
                        'is_buy': is_buy,
                        'quantity': desired_order['orderQty'],
                        'price': desired_order['price'],
                        'type': 'limit'  # Assuming a limit order type for amendments
                    })
            except IndexError:
                # Add orders to cancellation list if there's no matching desired order
                to_cancel.append(order)

        # Add any remaining unmatched desired orders to the creation list
        to_create += buy_orders[buys_matched:] + sell_orders[sells_matched:]

        # Amend orders if there are any in the amendment list
        if len(to_amend) > 0:
            for order in to_amend:
                print(f"AMEND THIS BITCH {order}")
                self.exchange.amend_HL_order(order['orderID'], order['is_buy'], order['quantity'], order['price'], order['type'])

        # Create new orders if there are any in the creation list
        if len(to_create) > 0:
            for order in to_create:
                self.exchange.place_HL_order(order['side'], order['orderQty'], order['price'])

        # Cancel orders if there are any in the cancellation list
        if len(to_cancel) > 0:
            for order in to_cancel:
                self.exchange.cancel_HL_order(order['orderID'])


    ###
    # Sanity
    ##

    def sanity_check(self):
        """Perform checks before placing orders."""
        pass

    ###
    # Running
    ###

    def check_connection(self):
        """
        Ensure the WS connections are still open.
        If any connection is closed, attempt to reconnect.
        """
        is_connected = True

        # Check HyperLiquid WebSocket connection
        if not self.exchange.HL.ws_open:
            logger.warning("HyperLiquid WebSocket connection closed. Attempting to reconnect...")
            try:
                self.exchange.HL.connect()
                self.exchange.HL.subscribe_to_l2_book()
                logger.info("Reconnected to HyperLiquid WebSocket and resubscribed to l2Book.")
            except Exception as e:
                logger.error(f"Failed to reconnect to HyperLiquid WebSocket: {e}")
                is_connected = False

        # Check Binance WebSocket connection (if applicable)
        # Assuming self.exchange.binance has a method or property to check connection status
        # if hasattr(self.exchange.binance, 'is_connected') and not self.exchange.binance.is_connected():
        #     logger.warning("Binance WebSocket connection closed. Attempting to reconnect...")
        #     try:
        #         # Reconnect logic for Binance
        #         # self.exchange.binance.reconnect()  # Example method
        #         logger.info("Reconnected to Binance WebSocket.")
        #     except Exception as e:
        #         logger.error(f"Failed to reconnect to Binance WebSocket: {e}")
        #         is_connected = False

        return is_connected

    def exit(self):
        logger.info("Shutting down. All open orders will be cancelled.")
        try:
            self.exchange.cancel_all_HL_orders()
        except Exception as e:
            logger.info("Unable to cancel orders: %s" % e)
        sys.exit()

    def run_loop(self):
        while True:
            sys.stdout.write("-----\n")
            sys.stdout.flush()

            sleep(settings.LOOP_INTERVAL)

            # Restart if any data connection is unexpectedly closed
            if not self.check_connection():
                logger.error("Realtime data connection unexpectedly closed, restarting.")
                self.restart()

            # self.sanity_check()  # Ensures health of mm - several cut-out points here
            # self.print_status()  # Print skew, delta, etc
            # self.place_orders()  # Creates desired orders and converges to existing orders

    def restart(self):
        logger.info("Restarting the market maker...")
        os.execv(sys.executable, [sys.executable] + sys.argv)

#
# Helpers
#

def cost(instrument, quantity, price):
    mult = instrument["multiplier"]
    P = mult * price if mult >= 0 else mult / price
    return abs(quantity * P)


def margin(instrument, quantity, price):
    return cost(instrument, quantity, price) * instrument["initMargin"]


def run():
    logger.info('BitMEX Market Maker Version: %s\n' % constants.VERSION)

    om = OrderManager()
    # Try/except just keeps ctrl-c from printing an ugly stacktrace
    try:
        om.run_loop()
    except (KeyboardInterrupt, SystemExit):
        sys.exit()