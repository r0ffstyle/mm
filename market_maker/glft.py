"""
Guéant-Lehalle-Fernandez-Tapia
"""
import numpy as np
from utils import log

logger = log.setup_custom_logger('root')

# Measure trading intensity with binning
def measure_trading_intensity(arrival_depth, tickSize=0.0001, max_ticks=500, bin_size=10):
    """
    Measure trading intensity by aggregating counts across price depth intervals (bins).
    """
    out = np.zeros(max_ticks)  # Adjust size for bins
    max_tick = 0

    for depth in arrival_depth:
        if not np.isfinite(depth):
            continue

        # Calculate bin index instead of individual tick
        tick = int(depth / tickSize)

        if tick < 0 or tick >= len(out):
            continue

        out[tick] += 1
        max_tick = max(max_tick, tick)

    # Perform binning on the output array to reduce sparsity
    return bin_data(out[:max_tick + 1], bin_size)

def bin_data(data, bin_size):
    """
    Bin the data by summing values across bins of size bin_size
    """
    n_bins = len(data) // bin_size
    binned_data = np.array([np.sum(data[i * bin_size:(i + 1) * bin_size]) for i in range(n_bins)])
    return binned_data

# Linear regression function
def linear_regression(x, y):
    if len(x) == 0:
        return 0, 0
    sx = np.sum(x)
    sy = np.sum(y)
    sx2 = np.sum(x ** 2)
    sxy = np.sum(x * y)
    w = len(x)
    slope = (w * sxy - sx * sy) / (w * sx2 - sx**2)
    intercept = (sy - slope * sx) / w
    return slope, intercept

# Compute coefficients for optimal spread
def compute_coeff(xi, gamma, delta, A, kappa):
    inv_k = np.divide(1, kappa)
    c1 = 1 / (xi * delta) * np.log(1 + xi * delta * inv_k)
    c2 = np.sqrt(np.divide(gamma, 2 * A * delta * kappa) * ((1 + xi * delta * inv_k) ** (kappa / (xi * delta) + 1)))
    return c1, c2

# GLFT Class to manage model parameters and optimal quote calculations
class GLFT:
    def __init__(self, A, gamma, xi, sigma, kappa, Q, adj1, adj2, delta):
        self.A = A
        self.gamma = gamma
        self.xi = xi
        self.sigma = sigma
        self.kappa = kappa
        self.Q = Q
        self.delta = delta
        self.adj1 = adj1
        self.adj2 = adj2

    def optimal_bid_ask_quotes(self, S, q, alpha=0):
        self.xi = self.gamma
        c1, c2 = compute_coeff(self.xi, self.gamma, self.delta, self.A, self.kappa)

        half_spread = (c1 + self.delta / 2 * c2 * self.sigma) * self.adj1

        skew = c2 * self.sigma * self.adj2

        bid_depth = half_spread + skew * q
        ask_depth = half_spread - skew * q
        
        bid_price = S - bid_depth + alpha
        ask_price = S + ask_depth + alpha
        fair_price = (ask_price + bid_price) / 2

        logger.info(f"Fair Price = {fair_price:.4f}, Mid Price = {S:.4f}")
        logger.info(f"Bid distance: {S - bid_price:.4f}, Ask distance: {ask_price - S:.4f}")
        logger.info(f"C1 = {c1:.6f}, C2 = {c2:.6f}")
        logger.info(f"Half spread = {half_spread:.6f}")
        logger.info(f"Skew = {skew:.6f}")
        return min(bid_price, S), max(ask_price, S) # Do not exceed mid


    def calibrate_parameters(self, arrival_depth, mid_price_chg, t, ticks):
        min_depth = min(arrival_depth)
        max_depth = max(arrival_depth)

        num_bins = 10  # Desired number of bins
        bins = np.linspace(min_depth, max_depth, num_bins + 1)
        counts, _ = np.histogram(arrival_depth, bins=bins)

        lambda_ = measure_trading_intensity(arrival_depth)
        # lambda_ = lambda_ / 600  # Adjust this scaling factor if necessary
        lambda_ = counts / 600
        # x = ticks[:len(lambda_)]
        # Compute bin centers
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Filter out zero lambda_ values
        mask = lambda_ > 0
        x = bin_centers[mask]
        lambda_ = lambda_[mask]
        print(f"lambda_: {lambda_}")
        y = np.log(lambda_)
        k_, logA = linear_regression(x, y)
        self.A = np.exp(logA)
        self.kappa = -k_ # TODO kappa is extremely inflated
        print("A and kappa:")
        print(self.A, self.kappa)

        if self.A <= 0 or self.kappa <= 0:
            logger.warning("Invalid A or kappa. Using default values.")
            self.A = max(self.A, 1.1e-5)
            self.kappa = max(self.kappa, 0.2)

        if t > 1:  # Ensure at least one change exists
            self.sigma = np.nanstd(mid_price_chg) * np.sqrt(10) # *sqrt(10) to adjust for 100ms
        else:
            self.sigma = 0