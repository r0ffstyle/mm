import sys
import numpy as np

from market_maker.market_maker import OrderManager
from market_maker.settings import settings
from glft import GLFT

from apricepred import AdvancedPricePredictor

from sklearn.linear_model import LogisticRegression


class CustomOrderManager(OrderManager):
    def __init__(self, exchange_api):
        super().__init__(exchange_api)
        self.inventory = 0
        self.short_term_alpha = 0
        self.ema_alpha = 0.1
        self.predictor = AdvancedPricePredictor()
        self.sigma = 0.1  # Placeholder for stock's volatility
        self.rho = 0.05  # Placeholder for risk-free rate
        self.execution_data = self.load_execution_data()
        self.execution_model = self.train_execution_model()
        self.glft = GLFT(A=self.mo_arrival_rate(), theta=0.5, sigma=0.1, Q=10, T=1)  # GLFT optimal bid/ask quotes

    def initialize_orders(self):
        """Place orders at multiple levels away from the current price for queue positioning."""
        current_price = self.exchange.get_last_price()
        for i in range(1, self.initial_levels + 1):
            # Place buy orders below the current price
            self.exchange.post_order('buy', current_price - i * self.distance_between_levels, quantity=1)  # Example quantity
            # Place sell orders above the current price
            self.exchange.post_order('sell', current_price + i * self.distance_between_levels, quantity=1)  # Example quantity
    
    def mo_arrival_rate(self):
        """Method to estimate market order arrival rate."""
        pass

    def load_execution_data(self):
        """Load historical data on order executions and their distances from the mid-price."""
        return []

    def train_execution_model(self):
        """Train a logistic regression model to estimate order execution probability."""
        X = np.array([data['distance_from_mid'] for data in self.execution_data]).reshape(-1, 1)
        y = np.array([data['was_executed'] for data in self.execution_data])
        model = LogisticRegression()
        model.fit(X, y)
        return model

    def order_execution_probability(self, distance_from_mid):
        """Estimate the probability of order execution based on its distance from the mid-price."""
        prob = self.execution_model.predict_proba(np.array([[distance_from_mid]]))[0][1]
        bid_depth, ask_depth, _ = self.analyze_order_book()
        liquidity_ratio = bid_depth / (bid_depth + ask_depth)
        volatility_adjustment = self.sigma
        adjusted_prob = prob * (1 + liquidity_ratio - volatility_adjustment)
        return min(max(adjusted_prob, 0), 1)

    def optimal_quotes(self):
        """Determine the optimal bid and ask quotes based on the Avellaneda-Stoikov model."""
        return self.glft.optimal_bid_ask_quotes(self.price, self.inventory)

    def extract_features(self, historical_data):
        """Extract relevant features (prices, volumes, spreads) from historical data."""
        prices = [data['price'] for data in historical_data]
        volumes = [data['volume'] for data in historical_data]
        spreads = [data['ask'] - data['bid'] for data in historical_data]
        return prices, volumes, spreads

    def net_inventory(self):
        """Update the net inventory."""
        pass

    def update_short_term_alpha(self):
        """Update the short-term alpha based on recent trades and volume-weighted price."""
        recent_trades = self.exchange.get_recent_trades()
        volume_weighted_price = sum([trade['price'] * trade['volume'] for trade in recent_trades]) / sum([trade['volume'] for trade in recent_trades])
        self.short_term_alpha = (1 - self.ema_alpha) * self.short_term_alpha + self.ema_alpha * volume_weighted_price

    def predict_price_movement(self):
        """Predict the next price movement using the AdvancedPricePredictor."""
        historical_data = self.exchange.get_historical_data()
        prices, volumes, spreads = self.extract_features(historical_data)
        features = [[prices[i], volumes[i], spreads[i]] for i in range(len(prices)-self.predictor.window_size)]
        targets = prices[self.predictor.window_size:]
        self.predictor.train(features, targets)

        recent_data = historical_data[-self.predictor.window_size:]
        recent_prices, recent_volumes, recent_spreads = self.extract_features(recent_data)
        predicted_price = self.predictor.predict([recent_prices[-1], recent_volumes[-1], recent_spreads[-1]])
        return predicted_price - recent_prices[-1]

    def dynamic_pricing(self, predicted_movement):
        """Determine the target price based on predicted movement and market liquidity."""
        bid_depth, ask_depth, spread = self.analyze_order_book()
        liquidity_ratio = bid_depth / (bid_depth + ask_depth)
        target_price = self.short_term_alpha + predicted_movement * liquidity_ratio
        return target_price

    def risk_management(self, proposed_order):
        """Evaluate if a proposed order meets the risk management criteria."""
        max_position = 100
        max_order_size = 10
        if abs(self.inventory + proposed_order['quantity']) > max_position or abs(proposed_order['quantity']) > max_order_size:
            return False
        return True

    def analyze_order_book(self):
        """Analyze the current order book to determine bid depth, ask depth, and spread."""
        order_book = self.exchange.get_order_book()
        bid_depth = sum([order['quantity'] for order in order_book['bids']])
        ask_depth = sum([order['quantity'] for order in order_book['asks']])
        spread = order_book['asks'][0]['price'] - order_book['bids'][0]['price']
        return bid_depth, ask_depth, spread

    def place_orders(self) -> None:
        """Create the orders to converge"""
        buy_orders = []
        sell_orders = []

        self.update_short_term_alpha()
        self.manage_inventory()

        # We make orders from the outer parts towards the middle.
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

    def manage_inventory(self):
        """Manage the current inventory by posting buy or sell orders based on predicted price movement."""
        predicted_movement = self.predict_price_movement()
        target_price = self.dynamic_pricing(predicted_movement)
        target_inventory = predicted_movement * 10

        if self.inventory > target_inventory:
            proposed_order = {'side': 'sell', 'price': target_price + 0.5, 'quantity': self.inventory - target_inventory}
            if self.risk_management(proposed_order):
                self.exchange.post_order(proposed_order['side'], proposed_order['price'], proposed_order['quantity'])
        elif self.inventory < target_inventory:
            proposed_order = {'side': 'buy', 'price': target_price - 0.5, 'quantity': target_inventory - self.inventory}
            if self.risk_management(proposed_order):
                self.exchange.post_order(proposed_order['side'], proposed_order['price'], proposed_order['quantity'])

    def manage_orders(self):
        """Manage the orders based on market movement and strategy logic."""
        # Check if the market price is close to one of the levels where we have an order
        current_price = self.exchange.get_last_price()
        for i in range(1, self.initial_levels + 1):
            buy_price = current_price - i * self.distance_between_levels
            sell_price = current_price + i * self.distance_between_levels

            # Decision logic for buy orders
            if abs(current_price - buy_price) < self.distance_between_levels / 2:  # Example threshold
                # Here, you can decide to cancel, adjust, or let the order execute based on other strategy logic
                pass

            # Decision logic for sell orders
            if abs(current_price - sell_price) < self.distance_between_levels / 2:  # Example threshold
                # Here, you can decide to cancel, adjust, or let the order execute based on other strategy logic
                pass

        

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
        self.update_short_term_alpha()
        self.manage_inventory()
        self.manage_orders()
    except (KeyboardInterrupt, SystemExit):
        sys.exit()
