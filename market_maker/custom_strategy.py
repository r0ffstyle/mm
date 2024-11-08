import sys
import traceback
import threading
import numpy as np
import pandas as pd
from time import sleep, time

from sklearn.linear_model import LinearRegression

from utils import log
from main_maker import OrderManager
from glft import GLFT, measure_trading_intensity


logger = log.setup_custom_logger('root')


Q = 75              # Maximum inventory level
order_size = 10     # Order size of the given instrument
tickSize = 0.0001   # WLD on HyperLiquid
loop_interval = 0.1 # In seconds, 100ms is lowest allowed
param_interval = 20 # How often to update parameters, in seconds
buffer_size = 1000  # In loop_interval ticks

class CustomOrderManager(OrderManager):
    def __init__(self):
        self.q = 0 # Inventory
        super().__init__()

        self.price_increment = tickSize * 10 # Increment to the next placed order
        self.order_pairs = 5     # Amount of pairs to place in the book (bid, ask)

        self.adj1 = 0.0015  # Arbitrary, lower => lower half spread
        self.adj2 = 0.5     # Arbitrary, affects skew which affects incentiveness to a neutral inventory
        self.delta = 0.5    # Higher delta => lower half-spread
        self.gamma = 1.0    # 1 <= gamma <= 10 is a typical range. Higher => wider spread
        self.xi = self.gamma
        
        # Parameters to be calibrated
        self.A = None
        self.kappa = 1e-8
        self.sigma = None

        self.glft = GLFT(self.A, self.gamma, self.xi, self.sigma, self.kappa, Q, self.adj1, self.adj2, self.delta)
        self.ticks = np.arange(500) + 0.5 # Tick range for linear regression

        # Buffers
        self.volatility_buffer = []
        self.trading_intensity_buffer = []
        self.buffer_size = buffer_size # In ticks

        self.arrival_depth = [] # For trading intensity
        self.mid_price_chg = [] # For volatiltiy
        self.prev_mid_price_tick = np.nan

        self.high_imbalance = 0.95
        self.low_imbalance = 0.05

        # Orderbook imbalanace parameters
        self.scale = 0.01 # Scaling factor for alpha signal
        self.oir_buffer = []
        self.voi_buffer = []
        self.price_change_buffer = []
        self.spread_buffer = []
        self.buffer_size_oir = 1000

        self.exchange.set_order_fill_callback(self.on_order_filled)

        sleep(1) # Allow the initialization to breathe

    """
    Binance
    """
    def get_binance_order_book(self):
        """Fetch the Binance order book"""
        return self.exchange.get_binance_order_book()

    def get_binance_latest_trade(self):
        """Fetch latest trades from Binance"""
        return self.exchange.get_binance_latest_trade()
    
    def get_binance_historical_trades(self):
        """Fetch latest trades from Binance"""
        return self.exchange.get_binance_historical_trades()
    
    def get_binance_mid_price(self):
        """Calculate mid price on Binance"""
        binance_order_book = self.get_binance_order_book()
        if binance_order_book is None:
            return
        mid_price = (float(binance_order_book['bids'][0][0]) + float(binance_order_book['asks'][0][0])) / 2
        return mid_price
    
    """
    HyperLiquid
    """
    def get_book(self):
        """Fetch the order book of the exchange"""
        return self.exchange.get_HL_order_book()

    def get_HL_volatility(self):
        "Get the volatiltiy based on the latest trades"
        return self.exchange.get_HL_volatility()

    def get_positions(self):
        """Fetch user positions"""
        return self.exchange.get_HL_positions()
    
    def on_order_filled(self, order_update):
        """Callback when an order is filled."""
        print(f"Order update: {order_update}")
        order = order_update.get('order', {})
        print(f"Order: {order}")
        order_id = order.get('oid')
        is_buy = order.get('isBuy', True)  # Default to True if not present
        side = 'B' if is_buy else 'A'
        self.handle_order_fill(order_id, side)

    def get_mid_price(self):
        """Calculate mid price on HyperLiquid."""
        book = self.get_book()
        if not book or not book[0] or not book[1]:
            logger.warning("Order book is empty or incomplete.")
            return None
        try:
            Pb = float(book[0][0]['price'])
            Pa = float(book[1][0]['price'])
            mid_price = (Pb + Pa) / 2.0
            self.record_mid_price_change(mid_price)
            self.record_arrival_depth(mid_price)
            return mid_price
        except (IndexError, KeyError, ValueError) as e:
            logger.error(f"Error calculating mid price: {e}")
            return None
    
    def get_queue_positions(self):
        """Compute the queue positions for our bid and ask orders"""
        book = self.get_book()
        if not book or not book[0] or not book[1]:
            logger.warning("Order book is incomplete")
            return None, None

        try:
            # Get our open orders
            existing_orders = self.exchange.get_open_HL_orders()

            # Initialize queue positions
            bid_queue_position = None
            ask_queue_position = None

            # Process bid and ask sides separately
            for side in ['B', 'A']:
                if side == 'B':
                    # Bids
                    book_side = book[0]
                else:
                    # Asks
                    book_side = book[1]

                # Get our order on this side
                our_orders = [order for order in existing_orders if order['side'] == side]

                if not our_orders:
                    # No order on this side
                    continue

                our_order = our_orders[0]  # Assuming only one order per side
                our_price = float(our_order['limitPx'])
                our_size = float(our_order['sz'])

                # Find total volume at our price level
                total_volume_at_price = 0.0
                for level in book_side:
                    level_price = float(level['price'])
                    level_size = float(level['size'])
                    if level_price == our_price:
                        total_volume_at_price = level_size
                        break

                # Estimate queue position assuming we are at the end of the queue
                volume_ahead = total_volume_at_price - our_size
                queue_position = volume_ahead / (total_volume_at_price + 1e-8)

                if side == 'B':
                    bid_queue_position = queue_position
                else:
                    ask_queue_position = queue_position

            return bid_queue_position, ask_queue_position

        except Exception as e:
            logger.error(f"Error computing queue positions: {e}")
            return None, None
        
    
    def compute_alpha(self):
        """Calculate alpha using regression based prediction of future price change"""
        # Ensure the regression model is available
        if not hasattr(self, 'regression_model'):
            logger.info("Regression model not available. Using default alpha of 0.")
            return 0

        if not hasattr(self, 'model_feature_names'):
            logger.error("Model feature names not found. Cannot compute alpha.")
            return 0

        # Prepare the current feature vector
        try:
            # Get the latest features
            spread = self.spread_buffer[-1] if self.spread_buffer else 0.0001

            # Initialize features dictionary with zeros
            features = {feature: 0 for feature in self.model_feature_names}

            # Update features with available data
            features['VOI0'] = self.voi_buffer[-1]
            features['OIR0'] = self.oir_buffer[-1]
            features['spread'] = spread

            for lag in range(1, 6):
                features[f'VOI{lag}'] = self.voi_buffer[-(lag+1)] if len(self.voi_buffer) > lag else 0
                features[f'OIR{lag}'] = self.oir_buffer[-(lag+1)] if len(self.oir_buffer) > lag else 0

            # Create DataFrame for prediction using the model's feature names
            X_pred = pd.DataFrame([features], columns=self.model_feature_names)

            # Predict price change
            predicted_price_change = self.regression_model.predict(X_pred)[0]

            # Compute alpha
            alpha = self.scale * predicted_price_change
            logger.info(f"Predicted price change: {predicted_price_change}, alpha: {alpha}")
            return alpha

        except Exception as e:
            logger.error(f"Error computing alpha: {e}")
            return 0

    def optimal_quotes(self):
        """Get the optimal quotes based on the GLFT model."""
        mid_price = self.get_mid_price()
        logger.info(f"HyperLiquid mid: {mid_price}")
        mid_binance = self.get_binance_mid_price()
        logger.info(f"Binance mid: {mid_binance}")

        if not self.buffers_filled():
            logger.info("Buffers not filled. Waiting for sufficient data...")
            return None, None
        
        if mid_binance is not None:
            alpha = self.compute_alpha()
            bid, ask = self.glft.optimal_bid_ask_quotes(mid_binance, self.q, alpha)
            logger.info(f"Bid Distance: {mid_price - bid}")
            logger.info(f"Ask Distance: {ask - mid_price}")
            return round_to_tick(bid), round_to_tick(ask)
        else:
            logger.warning("Mid price is None. Unable to get optimal quotes.")
            return None, None

    def get_imbalance(self, book):
        """Volume imbalance of the best bid and ask"""
        Qb = float(book[0][0]['size'])
        Qa = float(book[1][0]['size'])
        I = np.where(Qb + Qa != 0, Qb / (Qb + Qa), 0)
        return I
    
    def get_market_order_imbalance(self):
        """Market order imbalance"""
        market_buy_orders = self.exchange.get_market_buy_orders()
        market_sell_orders = self.exchange.get_market_sell_orders()
        order_flow_imbalance = (len(market_buy_orders) - len(market_sell_orders)) / ((len(market_buy_orders) + len(market_sell_orders)))
        return order_flow_imbalance

    def update_order_book_indicators(self):
        """Compute and store OIR, VOI, and related indicators."""
        with threading.Lock():
            book = self.get_book()
            if not book or not book[0] or not book[1]:
                logger.warning("Order book is empty or incomplete.")
                # Append np.nan to maintain buffer alignment
                self.spread_buffer.append(np.nan)
                self.oir_buffer.append(np.nan)
                self.voi_buffer.append(np.nan)
                self.price_change_buffer.append(np.nan)
                return

            try:
                # Best bid and ask prices and quantities
                Pb = float(book[0][0]['price'])
                Qb = float(book[0][0]['size'])
                Pa = float(book[1][0]['price'])
                Qa = float(book[1][0]['size'])
                spread = Pa - Pb
                self.spread_buffer.append(spread)

                # Mid-price
                mid_price = (Pb + Pa) / 2.0

                # Compute Order Imbalance Ratio
                oir = (Qb - Qa) / (Qb + Qa) if (Qb + Qa) != 0 else 0
                self.oir_buffer.append(oir)

                # Compute Volume Order Imbalance
                delta_Qb = Qb - self.prev_Qb if hasattr(self, 'prev_Qb') else 0
                delta_Qa = Qa - self.prev_Qa if hasattr(self, 'prev_Qa') else 0

                # VOI calculation
                voi = delta_Qb - delta_Qa
                self.voi_buffer.append(voi)

                # Record previous bid and ask quantities
                self.prev_Qb = Qb
                self.prev_Qa = Qa

                # Compute price change
                if hasattr(self, 'prev_mid_price'):
                    price_change = mid_price - self.prev_mid_price
                    self.price_change_buffer.append(price_change)
                else:
                    self.price_change_buffer.append(0)
                self.prev_mid_price = mid_price


            except (IndexError, KeyError, ValueError) as e:
                logger.error(f"Error updating order book indicators: {e}")
                # Append np.nan to maintain buffer alignment
                self.spread_buffer.append(np.nan)
                self.oir_buffer.append(np.nan)
                self.voi_buffer.append(np.nan)
                self.price_change_buffer.append(np.nan)

            # Trim buffers
            max_buffer_length = self.buffer_size_oir
            self.spread_buffer = self.spread_buffer[-max_buffer_length:]
            self.oir_buffer = self.oir_buffer[-max_buffer_length:]
            self.voi_buffer = self.voi_buffer[-max_buffer_length:]
            self.price_change_buffer = self.price_change_buffer[-max_buffer_length:]
            
    def update_regression_model(self):
        """Update the regression model using the historical data."""
        # Ensure we have enough data
        MIN_DATA_POINTS = 50
        buffer_lengths = {
            'price_change_buffer': len(self.price_change_buffer),
            'voi_buffer': len(self.voi_buffer),
            'oir_buffer': len(self.oir_buffer),
            'spread_buffer': len(self.spread_buffer)
        }
        logger.info(f"Buffer lengths: {buffer_lengths}")

        # Calculate the minimum length across all buffers
        min_length = min(buffer_lengths.values())

        if min_length < MIN_DATA_POINTS:
            logger.info("Not enough data for regression. Skipping model update.")
            return

        # Prepare the data
        data = {
            'VOI0': self.voi_buffer[-min_length:],
            'spread': self.spread_buffer[-min_length:],
            'PriceChange': self.price_change_buffer[-min_length:]
        }

        # Convert data to DataFrame
        df = pd.DataFrame(data)
        df.dropna(inplace=True)

        # Check if we have enough data after dropping NaNs
        if len(df) < MIN_DATA_POINTS:
            logger.info("Not enough valid data after dropping NaNs. Skipping model update.")
            return

        # Create lagged features
        for lag in range(1, 6):
            df[f'VOI{lag}'] = df['VOI0'].shift(lag)

        # Drop rows with NaN values introduced by shifting
        df.dropna(inplace=True)

        # Define feature columns
        feature_columns = ['VOI0', 'VOI4', 'VOI5', 'spread']

        # Prepare data for regression
        X = df[feature_columns]
        y = df['PriceChange']

        # Fit the regression model
        self.regression_model = LinearRegression()
        self.regression_model.fit(X, y)

        # Store the feature names used in the model
        self.model_feature_names = feature_columns

        # Log the model coefficients
        logger.info(f"Regression coefficients: {self.regression_model.coef_}")

##########
# Orders #
##########
    def prepare_order(self, side, level, optimal_bid, optimal_ask):
        """Create an order object."""
        if side == 'B':
            # Place bids in decreasing fashion from the optimal bid
            adjusted_price = optimal_bid - (level - 1) * self.price_increment
        else:
            # Place asks in increasing fashion from the optimal ask
            adjusted_price = optimal_ask + (level - 1) * self.price_increment
        
        # Ensure the price is valid
        adjusted_price = max(adjusted_price, 0)

        return {
            'price': round_to_tick(adjusted_price),
            'orderQty': order_size,
            'side': side
        }

    def place_orders(self):
        """Create order items for use in convergence."""
        logger.info("Attempting to place orders...")
        buy_orders = []
        sell_orders = []
        optimal_bid, optimal_ask = self.optimal_quotes()
        if optimal_bid is None or optimal_ask is None:
            logger.info("Optimal quotes are not available. Skipping order placement.")
            return
        logger.info(f"Spread = {round(optimal_ask - optimal_bid, 5)}")

        for i in range(1, self.order_pairs + 1):
            if not self.long_position_limit_exceeded():
                buy_order = self.prepare_order('B', i, optimal_bid, optimal_ask)
                if buy_order is not None:
                    buy_orders.append(buy_order)
            else:
                logger.info("Long position limit exceeded.")
            
            if not self.short_position_limit_exceeded():
                sell_order = self.prepare_order('A', i, optimal_bid, optimal_ask)
                if sell_order is not None:
                    sell_orders.append(sell_order)
                else:
                    logger.info("Short position limit exceeded")
        
        return self.converge_orders(buy_orders, sell_orders)

    def handle_order_fill(self, order_id, side):
        """Handle the event when an order is filled and place a immediate new order to maintain presence."""
        logger.info(f"Order {order_id} on side {side} filled.")
        optimal_bid, optimal_ask = self.optimal_quotes()
        if optimal_bid is None or optimal_ask is None:
            logger.warning("Optimal quotes not available. Skipping placing new order.")
            return

        if side == 'B':
            if not self.long_position_limit_exceeded():
                # Place a new bid order at the optimal level
                new_order = self.prepare_order('B', 1, optimal_bid, optimal_ask)
                self.exchange.place_HL_order(new_order['side'], new_order['orderQty'], new_order['price'])
        else:
            if not self.short_position_limit_exceeded():
                # Place a new ask order at the optimal level
                new_order = self.prepare_order('A', 1, optimal_bid, optimal_ask)
                self.exchange.place_HL_order(new_order['side'], new_order['orderQty'], new_order['price'])

    def converge_orders(self, buy_orders, sell_orders):
        """
        Converge the current orders with the desired orders.
        This function amends, creates, or cancels orders to match the desired state.

        Parameters:
        buy_orders (list): The list of desired buy orders.
        sell_orders (list): The list of desired sell orders.
        """
        logger.info("Converging orders...")
        # Initialize lists for order actions
        to_amend = []  # Orders to be amended
        to_create = []  # Orders to be created
        to_cancel = []  # Orders to be cancelled

        # Fetch the current open orders from HyperLiquid
        existing_orders = self.exchange.get_open_HL_orders()

        # Separate existing orders into buys and sells
        existing_buy_orders = [order for order in existing_orders if order['side'] == 'B']
        existing_sell_orders = [order for order in existing_orders if order['side'] == 'A']

        # Process buy orders
        buys_matched = 0
        for idx, desired_order in enumerate(buy_orders):
            if idx < len(existing_buy_orders):
                existing_order = existing_buy_orders[idx]
                # Compare and decide whether to amend
                price_difference = abs(desired_order['price'] - float(existing_order['limitPx']))
                if desired_order['orderQty'] != existing_order['sz'] or price_difference > (self.price_increment / 2):
                    # Only amend if there's a significant change
                    to_amend.append({
                        'oid': existing_order['oid'],
                        'is_buy': True,
                        'orderQty': desired_order['orderQty'],
                        'price': desired_order['price'],
                        'type': 'limit'
                    })
                buys_matched += 1
            else:
                # No existing order at this index, so create the desired order
                to_create.append(desired_order)

        # Any extra existing buy orders need to be cancelled
        for existing_order in existing_buy_orders[buys_matched:]:
            to_cancel.append(existing_order)

        # Process sell orders
        sells_matched = 0
        for idx, desired_order in enumerate(sell_orders):
            if idx < len(existing_sell_orders):
                existing_order = existing_sell_orders[idx]
                # Compare and decide whether to amend
                price_difference = abs(desired_order['price'] - float(existing_order['limitPx']))
                if desired_order['orderQty'] != existing_order['sz'] or price_difference > (self.price_increment / 2):
                    # Only amend if there's a significant change
                    to_amend.append({
                        'oid': existing_order['oid'],
                        'is_buy': False,
                        'orderQty': desired_order['orderQty'],
                        'price': desired_order['price'],
                        'type': 'limit'
                    })
                sells_matched += 1
            else:
                # No existing order at this index, so create the desired order
                to_create.append(desired_order)

        # Any extra existing sell orders need to be cancelled
        for existing_order in existing_sell_orders[sells_matched:]:
            to_cancel.append(existing_order)

        # Amend orders if there are any in the amendment list
        if to_amend:
            logger.info(f"Amending {len(to_amend)} orders...")
            for order in to_amend:
                self.exchange.amend_HL_order(
                    order['oid'], order['is_buy'], order['orderQty'], order['price'], order['type']
                )

        # Create new orders if there are any in the creation list
        if to_create:
            logger.info(f"Creating {len(to_create)} new orders...")
            for order in to_create:
                self.exchange.place_HL_order(order['side'], order['orderQty'], order['price'])

        # Cancel orders if there are any in the cancellation list
        if to_cancel:
            logger.info(f"Cancelling {len(to_cancel)} orders...")
            for order in to_cancel:
                self.exchange.cancel_HL_order(order['oid'])

    def update_glft_parameters(self):
        logger.info("Updating GLFT parameters...")

        # Ensure buffers are filled before updating
        if not self.buffers_filled():
            logger.info("Buffers not yet filled. Skipping parameter update.")
            return

        # Calibrate GLFT parameters
        self.glft.calibrate_parameters(self.arrival_depth, self.mid_price_chg, len(self.arrival_depth), self.ticks)

        logger.info(f"Volatility: {self.glft.sigma:.5f}")
        logger.info(f"Market Order Arrival Rate (A): {self.glft.A:.5f}")
        logger.info(f"Liquidity Intensity (kappa): {self.glft.kappa:.5f}")

        # Trim data to prevent lists from growing indefinitely
        MAX_DATA_POINTS = self.buffer_size
        if len(self.arrival_depth) > MAX_DATA_POINTS:
            self.arrival_depth = self.arrival_depth[-MAX_DATA_POINTS:]
        if len(self.mid_price_chg) > MAX_DATA_POINTS:
            self.mid_price_chg = self.mid_price_chg[-MAX_DATA_POINTS:]

    def update_inventory(self):
        """ Update the inventory and net position """
        self.q = self.get_positions()
        logger.info(f"Inventory: {self.q}")

    def long_position_limit_exceeded(self):
        """Check if the long position limit is exceeded"""
        return self.q >= Q

    def short_position_limit_exceeded(self):
        """Check if the short position limit is exceeded"""
        return self.q <= -Q
    
    def record_mid_price_change(self, mid_price):
        """Record the mid-price change for volatility calculation"""
        if not np.isnan(self.prev_mid_price_tick):
            self.mid_price_chg.append(mid_price - self.prev_mid_price_tick)
        self.prev_mid_price_tick = mid_price

    def record_arrival_depth(self, mid_price):
        """Record the market order arrival depth"""
        trades = self.exchange.get_HL_latest_trades()
        if trades:
            depths = []
            for trade in trades:
                side = trade['side']
                trade_price = float(trade['px'])
                if side == 'B':  # Buy trade
                    depths.append(abs(trade_price - mid_price))
                else:  # Sell trade
                    depths.append(abs(mid_price - trade_price))
            depth = max(depths)
            self.arrival_depth.append(depth)
        else:
            # Do not append if there are no trades
            pass

    def update_buffers(self):
        """Update the volatility and trading intensity buffers"""

        # Always call get_mid_price() to update data
        mid_price = self.get_mid_price()

        # Update volatility buffer
        if len(self.volatility_buffer) < self.buffer_size and len(self.mid_price_chg) > 1:
            vol = np.nanstd(self.mid_price_chg)
            self.volatility_buffer.append(vol)

        # Update trading intensity buffer
        if len(self.trading_intensity_buffer) < self.buffer_size:
            trading_intensity = measure_trading_intensity(self.arrival_depth)
            if len(trading_intensity) > 0:
                self.trading_intensity_buffer.append(trading_intensity.mean())
        
        # Calibrate parameters when buffers are filled
        if self.buffers_filled():
            self.glft.sigma = np.mean(self.volatility_buffer)
            self.glft.A = np.mean(self.trading_intensity_buffer)
            self.glft.calibrate_parameters(self.arrival_depth, self.mid_price_chg, len(self.arrival_depth), self.ticks)
            self.kappa = self.glft.kappa  # Ensure kappa is updated post-calibration


    def buffers_filled(self):
        """Check if both buffers (volatility and trading intensity) are filled."""
        return len(self.volatility_buffer) >= self.buffer_size and len(self.trading_intensity_buffer) >= self.buffer_size
    

    def run_loop(self):
        """Main trading loop."""
        last_param_update = time()
        last_model_update = time()
        model_update_interval = 60  # Update regression model every 60 seconds

        while True:
            try:
                sys.stdout.write("-----\n")
                sys.stdout.flush()

                # Check WebSocket connections
                if not self.check_connection():
                    logger.error("Realtime data connection unexpectedly closed, restarting.")
                    self.restart()
                    continue

                # Perform buffer update
                self.update_buffers()

                # Update order book indicators
                self.update_order_book_indicators()

                current_time = time()

                # Update GLFT parameters if interval has passed
                if current_time - last_param_update >= param_interval:
                    self.update_glft_parameters()
                    last_param_update = current_time

                # Update regression model if interval has passed
                if current_time - last_model_update >= model_update_interval:
                    self.update_regression_model()
                    last_model_update = current_time

                # Place orders if the buffers are filled
                if self.buffers_filled():
                    self.place_orders()
                    self.update_inventory()
                else:  # Just for logging
                    logger.info(f"Buffer: {len(self.trading_intensity_buffer)}")
                    # Log the current trading intensity mean
                    trading_intensity_mean = np.mean(self.trading_intensity_buffer) if len(self.trading_intensity_buffer) > 0 else None
                    logger.info(f"Trading intensity mean: {trading_intensity_mean}")
                    # Log the current volatility mean
                    volatility_mean = np.mean(self.volatility_buffer) * 100 if len(self.volatility_buffer) > 0 else None
                    logger.info(f"Volatility mean: {volatility_mean}%")

                # Sleep for the specified loop interval
                sleep(loop_interval)

            except Exception as e:
                error_message = f"An error occurred: {e}\n{traceback.format_exc()}"
                logger.error(error_message)
                break  # Exit the loop if an unexpected error occurs
       
##########
# Helpers #
##########

def round_to_tick(price):
    """Ensure price is multiple of tick size"""
    return round(price / tickSize) * tickSize

##########
# Runners #
##########
def run():
    logger.info('Starting the market maker...')
    order_manager = CustomOrderManager()

    try:
        order_manager.run_loop()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down the market maker...")
        if order_manager.parameter_update_timer:
            order_manager.parameter_update_timer.cancel()
        order_manager.exit()

if __name__ == "__main__":
    run()