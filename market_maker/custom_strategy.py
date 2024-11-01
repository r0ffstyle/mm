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

#
# Helpers
#
logger = log.setup_custom_logger('root')


Q = 50 # Maximum inventory level
order_size = 10
order_pairs = 1
tickSize = 0.0001 # WLD on HL
loop_interval = 1.0 # In seconds, 100ms is lowest allowed
param_interval = 20 # How often to update parameters, in seconds
buffer_size = 300 # In loop_interval ticks
parameter_history = {'A' : [], 'kappa' : [], 'sigma' : []} # Temporary logging to analyze issue with parameter initialization

class CustomOrderManager(OrderManager):
    def __init__(self):
        self.q = 0 # Inventory
        super().__init__()

        self.adj1 = 0.0015 # Arbitrary, lower => lower half spread
        self.adj2 = 0.5 # Arbitrary, affects skew which affects incentiveness to a neutral inventory
        self.delta = 0.5 # Higher delta => lower half-spread
        self.gamma = 1.0 #  1 <= gamma <= 10 is a typical range. Higher => wider spread
        self.xi = self.gamma
        
        # Parameters to be calibrated
        self.A = None
        self.kappa = 1e-8
        self.sigma = None

        self.glft = GLFT(self.A, self.gamma, self.xi, self.sigma, self.kappa, Q, self.adj1, self.adj2, self.delta)
        self.ticks = np.arange(500) + 0.5 # Tick range for linear regression

        # Buffers for trading intensity and volatility
        self.volatility_buffer = []
        self.trading_intensity_buffer = []
        self.buffer_size = buffer_size # In ticks

        self.arrival_depth = [] # For trading intensity
        self.mid_price_chg = [] # For volatiltiy
        self.prev_mid_price_tick = np.nan

        self.high_imbalance = 0.95
        self.low_imbalance = 0.05

        # OBI parameters
        self.beta = 0.001 # Scaling factor for alpha
        self.oir_buffer = []
        self.voi_buffer = []
        self.price_change_buffer = []
        self.spread_buffer = []
        self.buffer_size_oir = 1000  # Adjust as needed

        sleep(1)

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
    
    def compute_alpha(self):
        """Calculate alpha using regression based prediction of future price change"""
        # Ensure the regression model is available
        if not hasattr(self, 'regression_model'):
            logger.info("Regression model not available. Using default alpha of 0.")
            return 0

        # Prepare the current feature vector
        try:
            # Get the latest features
            spread = self.spread_buffer[-1] if self.spread_buffer else 0.0001  # Avoid division by zero
            features = {
                'VOI0': self.voi_buffer[-1],
                'OIR0': self.oir_buffer[-1],
                'Spread': spread
            }

            # Add lagged features
            for lag in range(1, 6):
                features[f'VOI{lag}'] = self.voi_buffer[-(lag+1)] if len(self.voi_buffer) > lag else 0
                features[f'OIR{lag}'] = self.oir_buffer[-(lag+1)] if len(self.oir_buffer) > lag else 0

            # Create DataFrame for prediction
            X_pred = pd.DataFrame([features])
            feature_columns = ['VOI0', 'VOI1', 'VOI2', 'VOI3', 'VOI4', 'VOI5',
                               'OIR0', 'OIR1', 'OIR2', 'OIR3', 'OIR4', 'OIR5']
            X_pred = X_pred[feature_columns]

            # Predict price change
            predicted_price_change = self.regression_model.predict(X_pred)[0]

            # Compute alpha
            alpha = self.beta * predicted_price_change
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
    
    def get_queue_position(self):
        """Position in order book queue"""
        pass
    
    def get_market_order_imbalance(self):
        """Market order imbalance"""
        market_buy_orders = self.exchange.get_market_buy_orders()
        market_sell_orders = self.exchange.get_market_sell_orders()
        order_flow_imbalance = (len(market_buy_orders) - len(market_sell_orders)) / ((len(market_buy_orders) + len(market_sell_orders)))
        return order_flow_imbalance

    def update_order_book_indicators(self):
        """Compute and store OIR, VOI, and related indicators."""
        book = self.get_book()
        if not book or not book[0] or not book[1]:
            logger.warning("Order book is empty or incomplete.")
            return

        try:
            # Best bid and ask prices and quantities
            Pb = float(book[0][0]['price'])
            Qb = float(book[0][0]['size'])
            Pa = float(book[1][0]['price'])
            Qa = float(book[1][0]['size'])
            spread = Pa - Pb

            # Mid-price
            mid_price = (Pb + Pa) / 2.0

            # Record spread
            self.spread_buffer.append(spread)

            # Compute OIR
            oir = (Qb - Qa) / (Qb + Qa) if (Qb + Qa) != 0 else 0
            self.oir_buffer.append(oir / spread if spread != 0 else 0)

            # Compute VOI
            delta_Qb = Qb - self.prev_Qb if hasattr(self, 'prev_Qb') else 0
            delta_Qa = Qa - self.prev_Qa if hasattr(self, 'prev_Qa') else 0

            # VOI calculation
            voi = delta_Qb - delta_Qa
            self.voi_buffer.append(voi / spread if spread != 0 else 0)

            # Record previous bid and ask quantities
            self.prev_Qb = Qb
            self.prev_Qa = Qa

            # Compute price change
            if hasattr(self, 'prev_mid_price'):
                price_change = mid_price - self.prev_mid_price
                self.price_change_buffer.append(price_change)
            self.prev_mid_price = mid_price

            # Trim buffers
            if len(self.oir_buffer) > self.buffer_size_oir:
                self.oir_buffer.pop(0)
            if len(self.voi_buffer) > self.buffer_size_oir:
                self.voi_buffer.pop(0)
            if len(self.price_change_buffer) > self.buffer_size_oir:
                self.price_change_buffer.pop(0)
            if len(self.spread_buffer) > self.buffer_size_oir:
                self.spread_buffer.pop(0)

        except (IndexError, KeyError, ValueError) as e:
            logger.error(f"Error updating order book indicators: {e}")
            
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

            # Calculate the minimum length accross all buffers
            min_length = min(len(self.voi_buffer), len(self.oir_buffer), len(self.spread_buffer), len(self.price_change_buffer))

            if len(self.price_change_buffer) < MIN_DATA_POINTS:
                logger.info("Not enough data for regression. Skipping model update.")
                return

            # Prepare the data
            df = pd.DataFrame({
                'VOI0': self.voi_buffer[-min_length:],
                'OIR0': self.oir_buffer[-min_length:],
                'Spread': self.spread_buffer[-min_length:],
                'PriceChange': self.price_change_buffer[-min_length:]
            })

            # Create lagged features
            for lag in range(1, 6):
                df[f'VOI{lag}'] = df['VOI0'].shift(lag)
                df[f'OIR{lag}'] = df['OIR0'].shift(lag)

            # Drop NaN values due to shifting
            df.dropna(inplace=True)

            # Features and target
            feature_columns = ['VOI0', 'VOI1', 'VOI2', 'VOI3', 'VOI4', 'VOI5', 'OIR0', 'OIR1', 'OIR2', 'OIR3', 'OIR4', 'OIR5']
            X = df[feature_columns]
            y = df['PriceChange']

            # Fit the regression model
            self.regression_model = LinearRegression()
            self.regression_model.fit(X, y)

            # Log the model coefficients
            logger.info(f"Regression coefficients: {self.regression_model.coef_}")

##########
# Orders #
##########
    def prepare_order(self, side, level, optimal_bid, optimal_ask):
        """Create an order object using GLFT model prices with adjustments for each level."""
        level_adjustment = abs(level - 1) * tickSize
        if side == 'B':
            adjusted_price = optimal_bid - level_adjustment
        else:
            adjusted_price = optimal_ask + level_adjustment
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
        logger.info(f"Spread = {round(optimal_ask - optimal_bid, 5)}")
        # Orders are created/amended from outside-in. This approach minimizes amendments, as innermost orders (closer to market price and more likely to be filled) are adjusted last. It efficiently manages order placement, reducing unnecessary adjustments and aligning with market activity probabilities.
        for i in reversed(range(1, order_pairs + 1)):
            if not self.long_position_limit_exceeded():
                buy_order = self.prepare_order('B', -i, optimal_bid, optimal_ask)
                if buy_order is not None:
                    buy_orders.append(buy_order)
            else:
                logger.info("Long position exceeded")
            if not self.short_position_limit_exceeded():
                sell_order = self.prepare_order('A', i, optimal_bid, optimal_ask)
                if sell_order is not None:
                    sell_orders.append(sell_order)
            else:
                logger.info("Short position exceeded")
        
        return self.converge_orders(buy_orders, sell_orders)

    def converge_orders(self, buy_orders, sell_orders):
        """
        Converge the current orders with the desired orders.
        This method amends, creates, or cancels orders to match the desired state.

        Parameters:
        buy_orders (list): The list of desired buy orders.
        sell_orders (list): The list of desired sell orders.
        """
        logger.info("Converging orders...")
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
                if desired_order['orderQty'] != order['sz'] or round_to_tick(desired_order['price']) != round_to_tick(float(order['limitPx'])):
                    # Add the order to the amendment list
                    to_amend.append({
                        'oid': order['oid'],
                        'is_buy': is_buy,
                        'orderQty': desired_order['orderQty'],
                        'price': desired_order['price'],
                        'type': 'limit'
                    })
            except IndexError:
                # Add orders to cancellation list if there's no matching desired order
                to_cancel.append(order)

        # Add any remaining unmatched desired orders to the creation list
        to_create += buy_orders[buys_matched:] + sell_orders[sells_matched:]

        # Amend orders if there are any in the amendment list
        if len(to_amend) > 0:
            logger.info(f"Amending {len(to_amend)} orders...")
            for order in to_amend:
                self.exchange.amend_HL_order(order['oid'], order['is_buy'], order['orderQty'], order['price'], order['type'])

        # Create new orders if there are any in the creation list
        if len(to_create) > 0:
            logger.info(f"Creating {len(to_create)} new orders...")
            for order in to_create:
                self.exchange.place_HL_order(order['side'], order['orderQty'], order['price'])

        # Cancel orders if there are any in the cancellation list
        if len(to_cancel) > 0:
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
        parameter_history['sigma'].append(self.glft.sigma)
        parameter_history['A'].append(self.glft.A)
        parameter_history['kappa'].append(self.glft.kappa)

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
import pickle
def save_parameter_history(filename='parameter_history.pkl'):
    """Save the parameter history to a file."""
    with open(filename, 'wb') as f:
        pickle.dump(parameter_history, f)
    logger.info(f"Parameter history saved to {filename}.")

def run():
    logger.info('Starting the market maker...')
    order_manager = CustomOrderManager()

    try:
        order_manager.run_loop()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down the market maker...")
        if order_manager.parameter_update_timer:
            order_manager.parameter_update_timer.cancel()
        save_parameter_history()
        order_manager.exit()

if __name__ == "__main__":
    run()