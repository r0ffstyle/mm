import market_maker.hyperliquid as hyperliquid

symbol = "BTC"
hyper = hyperliquid.HyperLiquid(symbol=symbol)
state = hyper.position()

print(state)