import numpy as np

class AvellanedaStoikov:
    def __init__(self, A, theta, sigma, Q, T):
        """
        Initialize the Avellaneda-Stoikov model with given parameters.
        
        Parameters:
        - A: Market order arrival rate.
        - theta: Parameter related to the spread.
        - sigma: Stock's volatility.
        - Q: Maximum inventory level.
        - T: Terminal time.
        """
        self.A = A
        self.theta = theta
        self.sigma = sigma
        self.Q = Q
        self.T = T
        self.nu = theta**2 * sigma**2  # Derived parameter for convenience.
        self.mu = A / (1 + theta)      # Derived parameter for convenience.
        self.M = self.compute_M()      # Matrix M used in the model.

    def compute_M(self):
        """
        Compute the matrix M as defined in the Avellaneda-Stoikov model.
        
        Returns:
        - M: The matrix.
        """
        M = np.zeros((2*self.Q + 1, 2*self.Q + 1))
        for q in range(-self.Q, self.Q + 1):
            M[q][q] = self.nu * q**2
            if q < self.Q:
                M[q][q+1] = -self.mu
            if q > -self.Q:
                M[q][q-1] = -self.mu
        return M

    def compute_v(self):
        """
        Compute the vector v using the closed-form expression of Eq. (11.9).
        
        Returns:
        - v: The vector.
        """
        # Using the closed-form expression of Eq. (11.9)
        v = np.exp(-self.M * (self.T - 0))  # Assuming current time t=0
        return v

    def optimal_bid_ask_quotes(self, S, q):
        """
        Determine the optimal bid and ask quotes based on the Avellaneda-Stoikov model.
        
        Parameters:
        - S: Current stock price.
        - q: Current inventory level.
        
        Returns:
        - bid: Optimal bid price.
        - ask: Optimal ask price.
        """
        v = self.compute_v()
        bid_spread = (1/self.theta) * np.log(v[q]/v[q+1]) + (1/self.theta) * np.log(1 + self.theta)
        ask_spread = (1/self.theta) * np.log(v[q]/v[q-1]) + (1/self.theta) * np.log(1 + self.theta)
        
        bid = S - bid_spread
        ask = S + ask_spread
        
        return bid, ask




"""
The AvellanedaStoikov class implements the Avellaneda-Stoikov market-making model, which is a mathematical model used to determine optimal bid and ask quotes for a market maker. The model is parameterized by several values:

A: The rate at which market orders arrive.
theta: A parameter that relates to the spread between the bid and ask prices.
sigma: The volatility of the stock.
Q: The maximum inventory level the market maker is willing to hold.
T: The terminal time, representing the end of the trading period.
The matrix M is a central component of the model, capturing the dynamics of the market maker's inventory and its impact on the bid and ask prices. The vector v is derived from M and represents the value function of the market maker's problem at different inventory levels.

The method optimal_bid_ask_quotes uses the model to compute the optimal bid and ask prices given the current stock price S and the market maker's current inventory level q. The computed bid and ask prices are designed to maximize the market maker's expected profit while accounting for inventory risk.
"""