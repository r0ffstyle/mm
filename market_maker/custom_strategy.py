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

    def get_alpha(self):
        """Calculate short-term alpha based on MO imbalance and market volatility"""
        alpha = self.get_market_order_imbalance()

        # Incorporate market volatility
        market_volatility = self.exchange.get_5min_volatility()
        alpha *= market_volatility

        return alpha

    def get_adverse_selection(self, order_book):
        # TODO
        """Calculate adverse selection risk based on the bid-ask spread"""
        # A larger spread increases the risk of being adversely selected
        best_bid = float(order_book['bids'][0][0])
        best_ask = float(order_book['asks'][0][0])
        spread = best_ask - best_bid

        midprice = (best_bid + best_ask) / 2
        adverse_selection = spread / midprice

        return adverse_selection

    def get_inventory_penalty(self):
        """Calculate inventory level"""
        pass

    def get_price(self, index):
        """Calculate the order price"""
        binance_order_book = self.get_binance_order_book()
        if binance_order_book is None:
            return

        skew = self.get_top_skew(binance_order_book)
        alpha = self.get_alpha()  # Calculate alpha

        # Incorporate inventory penalty
        inventory_penalty = self.get_inventory_penalty()

        mid_price = (float(binance_order_book['bids'][0][0]) + float(binance_order_book['asks'][0][0])) / 2
        price = round_to_tick(mid_price + index * (skew + alpha))

        return price

    ########
    # Orders
    ########


    def place_orders(self) -> None:
        """Create the orders to converge"""
        buy_orders = []
        sell_orders = []

        # We make orders from the outer parts towards the middle. This way, if an order near the middle is filled, we only need to create a new one there and adjust fewer existing orders.
        for i in reversed(range(1, settings.ORDER_PAIRS + 1)):
            if not self.long_position_limit_exceeded():
                buy_orders.append(self.prep_order(-i))
            if not self.short_position_limit_exceeded():
                sell_orders.append(self.prep_order(i))

        # Converge the orders
        self.converge_orders(buy_orders, sell_orders)

    def prep_order(self, index):
        """Prepare the order"""
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


# TODO
# Monitor the orders and cancel them if necessary (not currently implemented)
# This could be done based on whether the price has deviated a lot between Binance and BitMex. So that if the price moves up we would move the sell and buy before BitMex updates
# Start from quoting from the back and cancel as a free option when we are at risk of
# getting filled/ adverse selected
# IDEA: If level(s) above best bid/ask gets a lot of volume -> anticipate incoming market order and quote 
# to buy (if move is anticipated up) from slow participants and immediately sell when price moves



"""
@chameleon_jeff on Twitter
 
When getting started, only:
    * model
    * strategy 
    * latency 
should be part of the first problem class. 


Some important sounding things that I would recommend the 80/20 principle on:
1) inventory management
2) sizing
3) hedging

Starting with inventory management:

1. Query your positions periodically. 
2. Listen to fill feed or order/cancel responses. 
3. When your position changed, don't bother figuring out by how much. Instead, fade your top of book orders and don't take until you re-query.

Sounds hacky, but a low volume HFT strategy running with this system will get almost all the good volume compared to running with drop-copy and real time position reconciliation.

The volume you miss with this setup will be the more marginal fills anyway.


With market moves and hedging, just ignore it. Keep a small position limit, and bias towards flattening. 

Again, if you truly have short term alpha, your expected pnl will not suffer from this adjustment.

Compute markouts to determine if you have edge.

Once your strategy matures and you decide to productionize it, these issues slowly migrate towards things you must get "exactly right." 

But when starting out the proper implementation will bog you down and not contribute to the sign of your pnl.
"""