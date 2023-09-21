from sklearn.ensemble import GradientBoostingRegressor

class AdvancedPricePredictor:
    def __init__(self, window_size=50):
        """Initialize the price predictor with a given window size."""
        self.window_size = window_size
        self.model = GradientBoostingRegressor()

    def train(self, features, targets):
        """Train the price predictor using historical features and targets."""
        self.model.fit(features, targets)

    def predict(self, recent_features):
        """Predict the next price movement based on recent features."""
        return self.model.predict([recent_features])