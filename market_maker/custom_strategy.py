import sys
from typing import Dict, List
import numpy as np

from market_maker.market_maker import OrderManager
from market_maker.settings import settings

from glft import GLFT

class CustomOrderManager(OrderManager):
    def __init__(self):
        super().__init__()
        self.glft = GLFT
        self.inventory = {'a' : 0, 'b' : 0}
        self.open_orders = {} # For storing open orders with their IDs

    def get_binance_order_book(self):
        """Fetch the Binance order book"""
        return self.exchange.get_binance_order_book()

    def get_top_skew(self, order_book):
        """Calculate the skew of the order book"""
        top_bid_depth = float(order_book['bids'][0][1])
        top_ask_depth = float(order_book['asks'][0][1])
        skew = np.log(top_bid_depth) - np.log(top_ask_depth)
        return skew
    
    def get_queue_imbalance(self, order_book):
        """Volume imbalance of the best bid and ask"""
        top_bid_depth = float(order_book['bids'][0][1])
        top_ask_depth = float(order_book['asks'][0][1])
        imbalance = top_bid_depth / (top_bid_depth + top_ask_depth)
        return imbalance
    
    def get_market_order_imbalance(self):
        """Market order imbalance"""
        market_buy_orders = self.exchange.get_market_buy_orders()
        market_sell_orders = self.exchange.get_market_sell_orders()

        order_flow_imbalance = (len(market_buy_orders) - len(market_sell_orders)) / ((len(market_buy_orders) + len(market_sell_orders)))
        return order_flow_imbalance

    def get_mid_price(self):
        """Calculate mid price"""
        binance_order_book = self.get_binance_order_book()
        if binance_order_book is None:
            return
    
        mid_price = (float(binance_order_book['bids'][0][0]) + float(binance_order_book['asks'][0][0])) / 2
        return mid_price

    def optimal_quotes(self):
        """Get the optimal quotes based on the GLFT formula"""
        return self.glft.optimal_bid_ask_quotes(S=self.get_mid_price(), q=self.inventory)


    ##########
    # Orders #
    ##########

    def initialize_orders(self):
        """Place orders at multiple levels away from the current price for queue positioning"""
        current_price = self.get_mid_price()
        for i in range(1, self.initial_levels + 1):
            self.place_orders('buy', current_price - i * self.distance_between_levels, quantity=1)  # Example
            self.place_orders('sell', current_price + i * self.distance_between_levels, quantity=1)  # Example


    def place_orders(self) -> None:
        buy_orders = []
        sell_orders = []

        for i in reversed(range(1, settings.ORDER_PAIRS + 1)):
            if not self.long_position_limit_exceeded():
                buy_orders.append(self.prep_order(-i))
            if not self.short_position_limit_exceeded():
                sell_orders.append(self.prep_order(i))

        self.converge_orders(buy_orders, sell_orders)

    def prep_order(self, index):
        quantity = settings.ORDER_START_SIZE + ((abs(index) - 1) * settings.ORDER_STEP_SIZE)
        price = self.get_price(index)

        return {'price': price, 'orderQty': quantity, 'side': "Buy" if index < 0 else "Sell"}
    
    def converge_orders(self, buy_orders, sell_orders):
        pass

    def check_executed_orders(self):
        current_orders = self.exchange.get_open_orders()
        executed_orders = set(self.open_orders.keys()) - set(o['orderID'] for o in current_orders)

        # Update the inventory as the orders are filled
        for order_id in executed_orders:
            order = self.open_orders[order_id]
            if order['side'] == 'Buy':
                self.inventory['a'] += order['orderQty']
            else:
                self.inventory['b'] += order['orderQty']
            del self.open_orders[order_id]


#
# Helpers
#

def round_to_tick(price):
    """Ensure price is multiple of tick size"""
    return round(price * 2) / 2

def run() -> None:
    order_manager = CustomOrderManager()
    try:
        order_manager.run_loop()
        if order_manager.orders_init: # Initialize the first orders
            order_manager.orders_init = True
    except (KeyboardInterrupt, SystemExit):
        sys.exit()