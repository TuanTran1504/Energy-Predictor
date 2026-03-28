"""
Test Binance connection — market data + testnet trading.
Run with: python scripts/test_binance.py
"""
import os
from dotenv import load_dotenv
from pathlib import Path
from binance.client import Client
from binance.exceptions import BinanceAPIException

load_dotenv(Path(__file__).parent.parent / ".env")

API_KEY = os.getenv("BINANCE_API_KEY", "")
SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")
IS_TESTNET = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

def get_client() -> Client:
    client = Client(API_KEY, SECRET_KEY, testnet=IS_TESTNET)
    return client


def test_market_data():
    """
    Fetch current BTC and ETH prices.
    No API key needed for this — public endpoint.
    """
    client = Client()  # no keys needed for market data

    btc = client.get_symbol_ticker(symbol="BTCUSDT")
    eth = client.get_symbol_ticker(symbol="ETHUSDT")

    print(f"BTC/USDT: ${float(btc['price']):,.2f}")
    print(f"ETH/USDT: ${float(eth['price']):,.2f}")

    # 24hr stats
    btc_stats = client.get_ticker(symbol="BTCUSDT")
    print(f"\nBTC 24hr change: {float(btc_stats['priceChangePercent']):+.2f}%")
    print(f"BTC 24hr volume: ${float(btc_stats['quoteVolume']):,.0f}")


def test_historical_data():
    """Fetch last 10 daily candles for BTC."""
    client = Client()

    candles = client.get_klines(
        symbol="BTCUSDT",
        interval=Client.KLINE_INTERVAL_1DAY,
        limit=5
    )

    print("\nBTC last 5 days:")
    for c in candles:
        import datetime
        date = datetime.datetime.fromtimestamp(c[0]/1000).strftime("%Y-%m-%d")
        open_p = float(c[1])
        close_p = float(c[4])
        change = ((close_p - open_p) / open_p) * 100
        print(f"  {date}: ${close_p:,.2f} ({change:+.2f}%)")


def test_testnet_account():
    """
    Check testnet account balance.
    Requires API key — uses testnet so no real money.
    """
    if not API_KEY:
        print("\nSkipping account test — no API key set")
        return

    try:
        client = get_client()
        account = client.get_account()
        balances = [
            b for b in account["balances"]
            if float(b["free"]) > 0
        ]
        print("\nTestnet account balances:")
        for b in balances[:5]:
            print(f"  {b['asset']}: {float(b['free']):.4f}")
    except BinanceAPIException as e:
        print(f"\nTestnet error: {e}")


if __name__ == "__main__":
    print("=== Binance Market Data (no key needed) ===")
    test_market_data()
    test_historical_data()

    print("\n=== Binance Testnet Account ===")
    test_testnet_account()