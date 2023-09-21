import numpy as np

class GLFT:
    def __init__(self, A, gamma, sigma, kappa):
        """
        Initialize the Guéant-Lehalle-Fernandez-Tapia model with given parameters.
        
        Parameters:
        - A: Market order arrival rate (liquidity of the asset).
        - gamma: The absolute risk aversion coefficient of the trader.
        - sigma: Stock's volatility.
        - kappa: Price sensitivity of market participants.
        """
        self.A = A
        self.gamma = gamma
        self.sigma = sigma
        self.kappa = kappa

    def optimal_bid_ask_quotes(self, S, q):
        """
        Determine the optimal bid and ask quotes based on the Guéant-Lehalle-Fernandez-Tapia model.
        
        Parameters:
        - S: Current stock price.
        - q: Current inventory level.
        
        Returns:
        - bid: Optimal bid price.
        - ask: Optimal ask price.
        """
        common_term = (1/self.gamma) * np.log(1 + self.gamma/self.kappa)
        variable_term = (2*q + 1)/2 * np.sqrt((self.sigma**2 * self.gamma) / (2 * self.kappa * self.A) * (1 + self.gamma/self.kappa)**(1 + self.kappa/self.gamma))
        
        bid = S - common_term - variable_term
        ask = S + common_term + variable_term
        
        return bid, ask
    
"""
    Parameters meaning:
    - A: Market order arrival rate (liquidity of the asset).
        This can be estimated from historical trade data. It represents how frequently trades are executed in the market. A higher value indicates a more liquid market. It can be computed as the average number of trades per unit of time over a historical period.
    - gamma: The absolute risk aversion coefficient of the trader.
        This is a measure of the trader's or market maker's risk aversion. It's a subjective parameter and might be determined based on the trader's preferences, past behavior, or through optimization techniques to maximize utility. In some cases, it might be estimated from historical trading data or set by the trading institution based on their risk management policies.
    - sigma: Stock's volatility.
    - kappa: Price sensitivity of market participants.
        This parameter represents how sensitive market participants are to price changes. It can be estimated from historical trade and quote data, observing how market participants react to changes in quoted prices.
"""