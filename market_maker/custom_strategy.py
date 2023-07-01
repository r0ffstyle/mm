import sys
from typing import Dict, List
import numpy as np

from market_maker.market_maker import OrderManager
from market_maker.settings import settings


class CustomOrderManager(OrderManager):
    """A custom order manager for a strategy based on Binance order book skew"""

    def get_binance_order_book(self):
        """Fetch the Binance order book"""
        return self.exchange.get_binance_order_book()

    def calculate_skew(self, order_book):
        """Calculate the skew of the order book"""
        top_bid_depth = float(order_book['bids'][0][1])
        top_ask_depth = float(order_book['asks'][0][1])
        skew = np.log(top_bid_depth) - np.log(top_ask_depth)

        return skew
    
    def calculate_imbalance(self, order_book):
        top_bid_depth = float(order_book['bids'][0][1])
        top_ask_depth = float(order_book['asks'][0][1])
        imbalance = (top_ask_depth - top_bid_depth) / (top_ask_depth + top_bid_depth)

        return imbalance
    
    def get_price(self, index):
        """Calculate the order price"""
        binance_order_book = self.get_binance_order_book()
        if binance_order_book is None:
            return
        
        skew = self.calculate_skew(binance_order_book)
        mid_price = (float(binance_order_book['bids'][0][0]) + float(binance_order_book['asks'][0][0])) / 2
        # Index indicates direction (negative = buy)
        price = round_to_tick(mid_price + index * skew)

        return price

    ###
    # Orders
    ###

    def place_orders(self) -> None:
        """Create the orders to converge"""
        buy_orders = []
        sell_orders = []

        # We make orders from the outer parts towards the middle. This way, if an order near the middle is filled,
        # we only need to create a new one there and adjust fewer existing orders.
        for i in reversed(range(1, settings.ORDER_PAIRS + 1)):
            if not self.long_position_limit_exceeded():
                buy_orders.append(self.prep_order(-i))
            if not self.short_position_limit_exceeded():
                sell_orders.append(self.prep_order(i))

        # Converge the orders
        self.converge_orders(buy_orders, sell_orders)

    
    def prep_order(self, index):
        quantity = settings.ORDER_START_SIZE + ((abs(index) - 1) * settings.ORDER_STEP_SIZE)
        price = self.get_price(index)

        return {'price': price, 'orderQty': quantity, 'side': "Buy" if index < 0 else "Sell"}

    ###
    # Running
    ###


#
# Helpers
#

def round_to_tick(price):
    """Ensure price is multiple of tick size"""
    return round(price * 2) / 2


def run() -> None:
    order_manager = CustomOrderManager()

    # Try/except just keeps ctrl-c from printing an ugly stacktrace
    try:
        order_manager.run_loop()
    except (KeyboardInterrupt, SystemExit):
        sys.exit()


# TODO
# Monitor the orders and cancel them if necessary (not currently implemented)
# This could be done based on whether the price has deviated a lot between Binance and BitMex
# Start from quoting from the back and cancel as a free option when we are at risk of
# getting filled/ adverse selected