import sys
from typing import Dict, List
import numpy as np

from market_maker.market_maker import OrderManager
from market_maker.settings import settings


class CustomOrderManager(OrderManager):
    """A custom order manager for a strategy based on Binance order book skew"""

    def get_binance_order_book(self) -> Dict[str, List[List[float]]]:
        """Fetch the Binance order book"""
        return self.exchange.get_binance_order_book()

    def calculate_skew(self, order_book: Dict[str, List[List[float]]]) -> float:
        """Calculate the skew of the order book"""
        top_bid_depth = float(order_book['bids'][0][1])
        top_ask_depth = float(order_book['asks'][0][1])
        skew = np.log(top_bid_depth) - np.log(top_ask_depth)

        return skew
    
    def get_price(self, index):
        """Function to fetch our order price"""
        binance_order_book = self.get_binance_order_book()
        # Sanity check
        if binance_order_book is None:
            return
        
        skew = self.calculate_skew(binance_order_book)
        mid_price = (float(binance_order_book['bids'][0][0]) + float(binance_order_book['asks'][0][0])) / 2
        # Index indicates direction
        price = round_to_tick(mid_price + index * skew)

        return price

    ###
    # Orders
    ###

    def place_orders(self) -> None:
        """Create the orders to converge"""
        buy_orders = []
        sell_orders = []

        for i in reversed(range(1, settings.ORDER_PAIRS + 1)):
            if not self.long_position_limit_exceeded():
                buy_orders.append(self.prep_order(-i))
            if not self.short_position_limit_exceeded():
                sell_orders.append(self.prep_order(i))

        # Place the orders on BitMEX
        self.converge_orders(buy_orders, sell_orders)

        # Monitor the orders and cancel them if necessary (not currently implemented)
        # This could be done based on whether the price has deviated a lot between Binance and BitMex
    
    def prep_order(self, index):
        quantity = settings.ORDER_START_SIZE + ((abs(index) - 1) * settings.ORDER_STEP_SIZE)
        price = self.get_price(index)

        return {'price': price, 'orderQty': quantity, 'side': "Buy" if index < 0 else "Sell"}


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



"""
Does not update on new price
"""