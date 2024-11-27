import sys
import uuid
import pickle
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
tick_size = 0.0001  # WLD on HyperLiquid
loop_interval = 0.1 # In seconds, 100ms is lowest allowed
param_interval = 20 # How often to update parameters, in seconds
buffer_size = 500  # In loop_interval ticks
MAKER_FEE = 0.0001 # Maker fee 0.010%

class CustomOrderManager(OrderManager):
    def __init__(self):
        self.q = 0.0  # Inventory
        super().__init__()

        self.price_increment = tick_size * 10  # Increment to the next placed order
        self.order_pairs = 5                   # Amount of pairs to place in the book (bid, ask)
        self.order_tracking = {}               # Track orders and their levels
        self.positions = {}                    # Tracking of positions and their state

        self.order_size = order_size

        self.adj1 = 0.0015  # Arbitrary, lower => lower half spread
        self.adj2 = 0.5     # Arbitrary, affects skew which affects incentive to a neutral inventory
        self.delta = 0.5    # Higher delta => lower half-spread
        self.gamma = 1.0    # 1 <= gamma <= 10 is a typical range. Higher => wider spread
        self.xi = self.gamma

        # Parameters to be calibrated
        self.A = None
        self.kappa = 1e-8
        self.sigma = None

        self.glft = GLFT(self.A, self.gamma, self.xi, self.sigma, self.kappa, Q, self.adj1, self.adj2, self.delta)
        self.ticks = np.arange(500) + 0.5  # Tick range for linear regression

        # Buffers
        self.volatility_buffer = []
        self.trading_intensity_buffer = []
        self.buffer_size = buffer_size  # In ticks

        self.arrival_depth = []  # For trading intensity
        self.mid_price_chg = []  # For volatility
        self.prev_mid_price_tick = np.nan

        self.high_imbalance = 0.95
        self.low_imbalance = 0.05

        # Order book imbalance parameters
        self.scale = 0.01  # Scaling factor for alpha signal
        self.oir_buffer = []
        self.voi_buffer = []
        self.price_change_buffer = []
        self.spread_buffer = []
        self.buffer_size_oir = 1000

        self.exchange.set_order_fill_callback(self.on_order_filled)

        # Locks for thread safety
        self.lock = threading.RLock()

        # Threads
        self.market_data_thread = threading.Thread(target=self.market_data_loop, daemon=True)
        self.order_management_thread = threading.Thread(target=self.order_management_loop, daemon=True)

        # Start the threads
        self.market_data_thread.start()
        self.order_management_thread.start()

        sleep(0.5)
        logger.info("CustomOrderManager initialized with multithreading.")

    """
    Binance Functions
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
            return None
        mid_price = (float(binance_order_book['bids'][0][0]) + float(binance_order_book['asks'][0][0])) / 2
        return mid_price
    
    """
    HyperLiquid Functions
    """
    def get_book(self):
        """Fetch the order book of the exchange"""
        return self.exchange.get_HL_order_book()

    def get_HL_volatility(self):
        """Get the volatility based on the latest trades"""
        return self.exchange.get_HL_volatility()

    def get_positions(self):
        """Fetch user positions"""
        return self.exchange.get_HL_positions()
    
    def on_order_filled(self, order_update):
        with self.lock:
            logger.info(f"Order update: {order_update}")
            order = order_update.get('order', {})
            order_id = order.get('oid')
            status = order_update.get('status', '').lower()
            is_buy = order.get('isBuy', order.get('side') == 'B')
            side = 'B' if is_buy else 'A'
            print(f"Order id type: {type(order_id)}")

            if status == 'fill':
                # 'sz' represents the filled quantity in this update
                filled_qty = float(order.get('sz', 0))
                remaining_size = 0  # Assuming 'fill' represents a partial fill
            elif status == 'filled':
                # 'sz' represents the remaining size, which should be 0 for fully filled orders
                filled_qty = float(order.get('origSz', 0)) - float(order.get('sz', 0))
                remaining_size = float(order.get('sz', 0))
            else:
                logger.warning(f"Unknown status '{status}' for order {order_id}. Skipping.")
                return

            if order_id in self.order_tracking:
                order_info = self.order_tracking[order_id]
                logger.debug(f"Order {order_id} found in tracking.")
            else:
                # Order not tracked, possibly filled immediately upon placement
                # Create order_info from order update
                orig_size = float(order.get('origSz', 0))
                price = float(order.get('limitPx', 0))
                order_info = {
                    'level': 1,  # Default level if unknown
                    'timestamp': order.get('timestamp', time()),
                    'price': price,
                    'orderQty': orig_size,
                    'filledQty': 0,
                    'remaining_size': orig_size
                }
                self.order_tracking[order_id] = order_info
                logger.warning(f"Order {order_id} was not in tracking. Added to tracking.")

            # Calculate filled quantity based on status
            if status == 'fill':
                # For 'fill', 'sz' is the filled quantity
                filled_qty = float(order.get('sz', 0))
                remaining_size = order_info['remaining_size'] - filled_qty
            elif status == 'filled':
                # For 'filled', 'sz' is the remaining size
                filled_qty = order_info['orderQty'] - float(order.get('sz', 0))
                remaining_size = float(order.get('sz', 0))

            # Update order tracking
            order_info['filledQty'] += filled_qty
            order_info['remaining_size'] = remaining_size
            self.order_tracking[order_id] = order_info

            logger.debug(f"Order {order_id}: filled_qty={filled_qty}, remaining_size={remaining_size}")

            if filled_qty <= 0:
                logger.warning(f"No new fill for order {order_id}.")
                return

            # Process the fill
            if status == 'fill':
                self.handle_order_fill(order_id, side, filled_qty, order)
            elif status == 'filled':
                if remaining_size <= 0:
                    self.handle_order_fill(order_id, side, filled_qty, order)
                else:
                    logger.info(f"Order {order_id} partially filled. Remaining quantity: {remaining_size}")

    def get_mid_price(self):
        """Calculate the mid-price of the order book"""
        book = self.get_book()
        if not book or not book[0] or not book[1]:
            logger.warning("Order book is empty or incomplete.")
            return None
        try:
            Pb = float(book[0][0]['price'])
            Pa = float(book[1][0]['price'])
            mid_price = (Pb + Pa) / 2.0
            with self.lock:
                self.record_mid_price_change(mid_price)
                self.record_arrival_depth(mid_price)
            return mid_price
        except (IndexError, KeyError, ValueError) as e:
            logger.error(f"Error calculating mid price: {e}")
            return None
        
    def compute_alpha(self):
        """Calculate alpha using regression-based prediction of future price change"""
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
        with self.lock:
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
        """Create an order object"""
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
            'orderQty': self.order_size,
            'side': side,
            'level': level,
            'timestamp': time()
        }

    def place_orders(self):
        """Create an order item for use in the convergence function"""
        with self.lock:
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

            self.converge_orders(buy_orders, sell_orders)

    def handle_order_fill(self, order_id, side, filled_qty, order):
        """Handle the event when an initial order is filled"""
        logger.info(f"Order {order_id} on side {side} filled {filled_qty} units.")
        order_info = self.order_tracking.get(order_id)
        if order_info:
            level = order_info.get('level', 1)  # Default to 1 if not available
            filled_price = order_info['price']

            # Update the filled quantity
            order_info['filledQty'] += filled_qty
            remaining_qty = order_info['orderQty'] - order_info['filledQty']

            logger.debug(f"Order {order_id}: filledQty={order_info['filledQty']}, remaining_qty={remaining_qty}")

            # Update inventory
            self.q += filled_qty if side == 'B' else -filled_qty
            logger.info(f"Updated inventory: {self.q}")

            # Check if the order is fully filled
            if remaining_qty <= 0:
                # Generate a unique position ID
                position_id = str(uuid.uuid4())

                # Add to positions tracking
                with self.lock:
                    self.positions[position_id] = {
                        'side': side,
                        'quantity': order_info['orderQty'],
                        'entry_price': filled_price,
                        'state': 'open',
                        'timestamp': time(),
                        'order_id': order_id  # Store the initial order_id for reference
                    }
                    logger.debug(f"Position {position_id} created: {self.positions[position_id]}")

                # Remove order from tracking
                self.order_tracking.pop(order_id, None)
                logger.info(f"Order {order_id} fully filled and removed from tracking.")

                # Decide whether to replace the order
                self.replace_order(side, level)
            else:
                # Update the order tracking with new filled quantity
                self.order_tracking[order_id] = order_info
                logger.info(f"Order {order_id} partially filled. Remaining quantity: {remaining_qty}")
        else:
            logger.warning(f"No tracking info for order {order_id}.")

    def handle_unwind_fill(self, position_id, side, filled_qty, order):
        """Handle the event when an unwind order is filled."""
        position = self.positions.get(position_id)
        if position:
            # Update inventory
            self.q += filled_qty if side == 'B' else -filled_qty
            logger.info(f"Updated inventory after unwinding: {self.q}")

            # Compute profit on the filled quantity
            exit_price = float(order.get('limitPx', 0))
            entry_price = position['entry_price']

            if side == 'B':
                # Closing short position
                gross_profit = (entry_price - exit_price) * filled_qty
            else:
                # Closing long position
                gross_profit = (exit_price - entry_price) * filled_qty

            # Calculate fees
            entry_fee = entry_price * filled_qty * MAKER_FEE
            exit_fee = exit_price * filled_qty * MAKER_FEE
            total_fees = entry_fee + exit_fee

            # Calculate net profit
            net_profit = gross_profit - total_fees

            logger.info(f"Position {position_id} unwound {filled_qty} units for a net profit of {net_profit:.6f}")

            # Update the position's remaining quantity
            position['quantity'] -= filled_qty
            logger.debug(f"Position {position_id}: new quantity={position['quantity']}")

            if position['quantity'] <= 0:
                # Position fully unwound
                self.positions.pop(position_id, None)
                logger.info(f"Position {position_id} fully unwound.")
            else:
                # Update the position
                self.positions[position_id] = position
                logger.info(f"Position {position_id} partially unwound. Remaining quantity: {position['quantity']}")
        else:
            logger.warning(f"Unwind fill received for unknown position {position_id}.")

    def check_unwind_profitability(self, position_id, position):
        """Check if unwinding the position now would be profitable."""
        side = position['side']
        entry_price = position['entry_price']
        quantity = position['quantity']

        # Get the latest order book
        book = self.get_book()
        if not book or not book[0] or not book[1]:
            logger.warning("Order book is empty or incomplete. Cannot check unwind profitability.")
            return False

        try:
            best_bid_price = float(book[0][0]['price'])
            best_ask_price = float(book[1][0]['price'])

            # Adjusted exit price is one tick beyond best bid/ask
            if side == 'B':
                # Long position, unwind with sell order at best_bid + tick_size
                exit_price = best_bid_price + tick_size
                gross_profit = (exit_price - entry_price) * quantity
            else:
                # Short position, unwind with buy order at best_ask - tick_size
                exit_price = best_ask_price - tick_size
                gross_profit = (entry_price - exit_price) * quantity

            # Calculate fees
            entry_fee = entry_price * quantity * 0.0001  # Maker fee: 0.010%
            exit_fee = exit_price * quantity * 0.0001    # Maker fee: 0.010%
            total_fees = entry_fee + exit_fee

            # Calculate net profit
            net_profit = gross_profit - total_fees

            logger.info(f"Position {position_id}: Gross Profit: {gross_profit}, Total Fees: {total_fees}, Net Profit: {net_profit}")

            minimum_profit_threshold = 0.0001  # Set a minimum profit threshold
            if net_profit > minimum_profit_threshold:
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Error checking unwind profitability: {e}")
            return False

    def unwind_position(self, position_id, position):
        """Execute an aggressive limit order to unwind the position."""
        try:
            # Check if an unwind order is already placed
            if position.get('state') == 'unwinding' and position.get('exit_order_id'):
                logger.info(f"Unwind order already placed for position {position_id}.")
                return

            side = position['side']
            remaining_quantity = position['quantity']

            if remaining_quantity <= 0:
                logger.info(f"Position {position_id} already fully unwound.")
                return

            book = self.get_book()
            if not book or not book[0] or not book[1]:
                logger.warning("Order book is empty or incomplete. Cannot unwind position.")
                return

            if side == 'B':
                # Long position; place sell limit order at best_bid + tick_size
                best_bid_price = float(book[0][0]['price'])
                exit_price = best_bid_price + tick_size
                unwind_side = 'A'  # Sell side
            else:
                # Short position; place buy limit order at best_ask - tick_size
                best_ask_price = float(book[1][0]['price'])
                exit_price = best_ask_price - tick_size
                unwind_side = 'B'  # Buy side

            logger.info(f"Unwinding position {position_id} with limit order on side {unwind_side} for quantity {remaining_quantity} at price {exit_price}")

            with self.lock:
                # Update position state before placing the order
                position['state'] = 'unwinding'
                self.positions[position_id] = position  # Ensure position is updated

            # Place the unwind order
            oid = self.exchange.place_HL_order(unwind_side, remaining_quantity, exit_price)
            logger.info(f"Unwind order placed, oid: {oid}")
            if oid is not None:
                with self.lock:
                    # Assign the new order ID to the position
                    position['exit_order_id'] = oid
                    self.positions[position_id] = position  # Update the position with new order ID
            else:
                logger.error("Failed to place unwind limit order")
        except Exception as e:
            logger.error(f"Error unwinding position: {e}")

    def replace_order(self, side, level):
        """Replace an order at the given level if necessary"""
        optimal_bid, optimal_ask = self.optimal_quotes()
        if optimal_bid is None or optimal_ask is None:
            logger.warning("Optimal quotes not available. Skipping placing new order.")
            return
        if side == 'B':
            if not self.long_position_limit_exceeded():
                new_order = self.prepare_order('B', level, optimal_bid, optimal_ask)
                oid = self.exchange.place_HL_order(new_order['side'], new_order['orderQty'], new_order['price'])
                if oid is not None:
                    with self.lock:
                        self.order_tracking[oid] = {
                            'level': new_order['level'],
                            'timestamp': new_order['timestamp'],
                            'price': new_order['price'],
                            'orderQty': new_order['orderQty'],
                            'filledQty': 0,
                            'remaining_size' : new_order['orderQty']
                        }
            else:
                logger.info("Long position limit exceeded. Not replacing buy order.")
        else:
            if not self.short_position_limit_exceeded():
                new_order = self.prepare_order('A', level, optimal_bid, optimal_ask)
                oid = self.exchange.place_HL_order(new_order['side'], new_order['orderQty'], new_order['price'])
                print(f"OID: {oid}, type: {type(oid)}")
                if oid is not None:
                    with self.lock:
                        self.order_tracking[oid] = {
                            'level': new_order['level'],
                            'timestamp': new_order['timestamp'],
                            'price': new_order['price'],
                            'orderQty': new_order['orderQty'],
                            'filledQty': 0,
                            'remaining_size' : new_order['orderQty']
                        }
            else:
                logger.info("Short position limit exceeded. Not replacing sell order.")

    def converge_orders(self, buy_orders, sell_orders):
        """
        Converge the current orders with the desired orders.
        This method amends, creates, or cancels orders to match the desired state.
        """
        logger.info("Converging orders...")
        # Initialize lists for order actions
        to_amend = []  # Orders to be amended
        to_create = []  # Orders to be created
        to_cancel = []  # Orders to be cancelled

        # Fetch the current open orders from HyperLiquid
        existing_orders = self.exchange.get_open_HL_orders()

        # Separate existing orders into buys and sells
        existing_buy_orders = []
        existing_sell_orders = []
        for order in existing_orders:
            if order['side'] == 'B':
                existing_buy_orders.append(order)
            elif order['side'] == 'A':
                existing_sell_orders.append(order)

        # Process buy orders
        buys_matched = 0
        for idx, desired_order in enumerate(buy_orders):
            if idx < len(existing_buy_orders):
                existing_order = existing_buy_orders[idx]
                # Get level information
                level = desired_order.get('level', 1)
                # Determine sensitivity
                sensitive = level == 1
                # Compare and decide whether to amend
                price_difference = abs(desired_order['price'] - float(existing_order['limitPx']))
                qty_difference = desired_order['orderQty'] != existing_order['sz']
                if sensitive or qty_difference or price_difference > (self.price_increment / 2):
                    # Only amend if sensitive or significant change
                    to_amend.append({
                        'oid': existing_order['oid'],
                        'is_buy': True,
                        'orderQty': desired_order['orderQty'],
                        'price': desired_order['price'],
                        'type': 'limit'
                    })
                    # Update timestamp in order tracking
                    if existing_order['oid'] in self.order_tracking:
                        self.order_tracking[existing_order['oid']]['timestamp'] = time()
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
                # Get level information
                level = desired_order.get('level', 1)
                # Determine sensitivity
                sensitive = level == 1
                # Compare and decide whether to amend
                price_difference = abs(desired_order['price'] - float(existing_order['limitPx']))
                qty_difference = desired_order['orderQty'] != existing_order['sz']
                if sensitive or qty_difference or price_difference > (self.price_increment / 2):
                    # Only amend if sensitive or significant change
                    to_amend.append({
                        'oid': existing_order['oid'],
                        'is_buy': False,
                        'orderQty': desired_order['orderQty'],
                        'price': desired_order['price'],
                        'type': 'limit'
                    })
                    # Update timestamp in order tracking
                    if existing_order['oid'] in self.order_tracking:
                        self.order_tracking[existing_order['oid']]['timestamp'] = time()
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
                # Update timestamp in order tracking
                if order['oid'] in self.order_tracking:
                    self.order_tracking[order['oid']]['timestamp'] = time()

        # Create new orders if there are any in the creation list
        if to_create:
            logger.info(f"Creating {len(to_create)} new orders...")
            for order in to_create:
                oid = self.exchange.place_HL_order(order['side'], order['orderQty'], order['price'])
                print(f"OID: {oid}, type: {type(oid)}")
                if oid is not None:
                    # Update order tracking
                    with self.lock:
                        self.order_tracking[oid] = {
                            'level': order['level'],
                            'timestamp': order['timestamp'],
                            'price': order['price'],
                            'orderQty': order['orderQty'],
                            'filledQty' : 0,
                            'remaining_size' : order['orderQty']
                        }
                    logger.info(f"Order {oid} placed and added to tracking.")
                else:
                    logger.error("Failed to place order")


        # Cancel orders if there are any in the cancellation list
        if to_cancel:
            logger.info(f"Cancelling {len(to_cancel)} orders...")
            for order in to_cancel:
                self.exchange.cancel_HL_order(order['oid'])
                # Remove from order tracking
                self.order_tracking.pop(order['oid'], None)

    def update_glft_parameters(self):
        """Update GLFT model parameters"""
        with self.lock:
            logger.info("Updating GLFT parameters...")

            if not self.buffers_filled():
                logger.info("Buffers not yet filled. Skipping parameter update.")
                return

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
        """Update the inventory"""
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
                depth = abs(trade_price - mid_price)
                depths.append(depth)
            depth = max(depths)
            self.arrival_depth.append(depth)
            logger.debug(f"Recorded arrival depth: {depth}")
        else:
            logger.warning("No trades received in record_arrival_depth.")


    def update_buffers(self):
        """Update the volatility and trading intensity buffers"""
        mid_price = self.get_mid_price()

        with self.lock:
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
                self.kappa = self.glft.kappa

    def buffers_filled(self):
        """Check if both buffers (volatility and trading intensity) are filled."""
        return len(self.volatility_buffer) >= self.buffer_size and len(self.trading_intensity_buffer) >= self.buffer_size
    
    ##########
    # Loops  #
    ##########
    def market_data_loop(self):
        """Loop to handle market data updates."""
        while True:
            try:
                # Update order book indicators
                self.update_order_book_indicators()

                # Update buffers
                self.update_buffers()

                # Sleep for a short duration to prevent tight loop
                sleep(loop_interval)
            except Exception as e:
                logger.error(f"Error in market_data_loop: {e}")
                break

    def order_management_loop(self):
        """Loop to handle order placement, management, and position monitoring."""
        last_param_update = time()
        last_model_update = time()
        model_update_interval = 60  # Update regression model every 60 seconds

        while True:
            try:
                current_time = time()

                # Update GLFT parameters if interval has passed
                if current_time - last_param_update >= param_interval:
                    self.update_glft_parameters()
                    last_param_update = current_time

                # Update regression model if interval has passed
                if current_time - last_model_update >= model_update_interval:
                    self.update_regression_model()
                    last_model_update = current_time

                # Place orders if buffers are filled
                if self.buffers_filled():
                    self.place_orders()
                    self.update_inventory()
                else:
                    logger.info(f"Buffer: {len(self.trading_intensity_buffer)}")
                    trading_intensity_mean = np.mean(self.trading_intensity_buffer) if len(self.trading_intensity_buffer) > 0 else None
                    logger.info(f"Trading intensity mean: {trading_intensity_mean}")
                    volatility_mean = np.mean(self.volatility_buffer) * 100 if len(self.volatility_buffer) > 0 else None
                    logger.info(f"Volatility mean: {volatility_mean}%")

                # Monitor positions for unwinding
                self.monitor_positions()

                # Sleep for a short duration to prevent tight loop
                sleep(loop_interval)
            except Exception as e:
                logger.error(f"Error in order_management_loop: {e}")
                break
       
    ##########
    # Helpers #
    ##########

    def monitor_positions(self):
        """Monitor open positions and unwind if profitable."""
        positions_to_unwind = []
        with self.lock:
            for position_id, position in self.positions.items():
                if position['state'] == 'open':
                    can_unwind = self.check_unwind_profitability(position_id, position)
                    if can_unwind:
                        positions_to_unwind.append((position_id, position))

        # Unwind positions outside the lock to avoid holding it during network operations
        for position_id, position in positions_to_unwind:
            self.unwind_position(position_id, position)

    ##########
    # Runners #
    ##########
    def run(self):
        logger.info('Starting the market maker...')

        try:
            while True:
                # Check WebSocket connections
                if not self.check_connection():
                    logger.error("Realtime data connection unexpectedly closed, restarting.")
                    self.restart()
                    continue

                # Sleep for a short duration to prevent tight loop
                sleep(1)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutting down the market maker...")
            self.exit()

    def exit(self):
        logger.info("Shutting down. All open orders will be cancelled.")
        try:
            self.exchange.cancel_all_HL_orders()
        except Exception as e:
            logger.info(f"Unable to cancel orders: {e}")
        sys.exit()

def round_to_tick(price):
    """Ensure price is multiple of tick size"""
    return round(price / tick_size) * tick_size

##########
# Runner #
##########
def run():
    order_manager = CustomOrderManager()
    order_manager.run()

if __name__ == "__main__":
    run()