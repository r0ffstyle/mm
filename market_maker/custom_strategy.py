import sys
from typing import Dict, List
import numpy as np

from market_maker.market_maker import OrderManager


class CustomOrderManager(OrderManager):
    """A custom order manager for a strategy based on Binance order book skew"""

    def place_orders(self) -> None:
        # Get the Binance order book
        binance_order_book = self.get_binance_order_book()

        # Make sure there is data from Binance before proceeding
        if binance_order_book is None:
            return

        # Calculate the skew of the Binance order book
        skew = self.calculate_skew(binance_order_book)

        # Calculate the mid price based on the Binance order book
        mid_price = (float(binance_order_book['bids'][0][0]) + float(binance_order_book['asks'][0][0])) / 2
        buy_price = self.round_to_tick(mid_price - skew)
        sell_price = self.round_to_tick(mid_price + skew)

        
        # Create a larger number of buy and sell orders at a variety of price levels
        # The trading size could be based on volatility (not currently implemented)
        depth = 3
        buy_orders = [{'price': buy_price - i * 10, 'orderQty': 100, 'side': "Buy"} for i in range(depth)]
        sell_orders = [{'price': sell_price + i * 10, 'orderQty': 100, 'side': "Sell"} for i in range(depth)]


        # Place the orders on BitMEX
        self.converge_orders(buy_orders, sell_orders)


        # Monitor the orders and cancel them if necessary (not currently implemented)
        # This could be done based on whether the price has deviated a lot between Binance and BitMex



    def get_binance_order_book(self) -> Dict[str, List[List[float]]]:
        """Fetch the Binance order book"""
        return self.exchange.get_binance_order_book()

    def calculate_skew(self, order_book: Dict[str, List[List[float]]]) -> float:
        # Calculate the skew of an order book
        top_bid_depth = float(order_book['bids'][0][1])
        top_ask_depth = float(order_book['asks'][0][1])

        # Calculate the skew as the log difference between bid depth and ask depth
        # We use the log to scale these by their order of magnitude
        skew = np.log(top_bid_depth / top_ask_depth)

        return skew

    def round_to_tick(self, price):
        """To ensure price tick is fulfilled"""
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