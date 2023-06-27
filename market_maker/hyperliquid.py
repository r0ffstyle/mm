import json
import market_maker.auth.eth_keys as keys

import eth_account
from eth_account.signers.local import LocalAccount

from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants


class HyperLiquid():
    """HyperLiquid API Connector."""
    def __init__(self, testnet=True, symbol=None):
        self.url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
        self.symbol = symbol

        # Init account
        self.account: LocalAccount = eth_account.Account.from_key(keys.ETH_SECRET)
        print("Running with account address:", self.account.address)
        self.info = Info(constants.TESTNET_API_URL, skip_ws=True)
        # Init exchange
        self.ex = Exchange(self.account, self.url)

    """Account methods"""
    def position(self):
        user_state = self.info.user_state(self.account.address)
        positions = []
        for position in user_state["assetPositions"]:
            if float(position["position"]["szi"]) != 0:
                positions.append(position["position"])
        if len(positions) > 0:
            print("positions:")
            for position in positions:
                print(json.dumps(position, indent=2))
        else:
            print("no open positions")

    """Order methods"""
    def limit_order(self, is_buy, size, price):
        try:
            order_result = self.ex.order(self.symbol, is_buy, size, price, {"limit": {"tif": "Gtc"}}) # Gtc = Good till cancelled
            return order_result
        except Exception as err:
            print(f"Error: {err}")
    
    def market_order(self, is_buy, size, trigger_price):
        order_result = self.ex.order(self.symbol, is_buy, size, trigger_price, {"trigger": {"triggerPx": trigger_price, "isMarket": True, "tpsl": "tp"}}) # Trigger price should be best price at direction
        return order_result
    
    def cancel_order(self, order_id):
        if order_id:
            try:
                cancel_result = self.ex.cancel(self.symbol, order_id)
                return cancel_result
            except Exception as err:
                print(f"Invalid Order ID: {err}")
    
    """Info methods"""
    def get_exchange_metadata(self):
        return self.info.meta()

    def get_state(self):
        return self.info.user_state(self.account.address)

    def get_open_orders(self):
        return self.info.open_orders(self.account.address)

    def get_fills(self):
        return self.info.user_fills(self.account.address)
    
    def get_ticker(self):
        return self.info.all_mids()[self.symbol]
    
    def get_l2_snapshot(self):
        return self.info.l2_snapshot(self.symbol)