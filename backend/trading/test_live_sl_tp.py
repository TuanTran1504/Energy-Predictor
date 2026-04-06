"""
Diagnostic: test which SL/TP order types work on live Binance.
Does NOT place any entry — only tests conditional order placement
by checking what the API accepts. Runs dry by default.

Usage:
    python test_live_sl_tp.py          # dry run — just print what would be sent
    python test_live_sl_tp.py --real   # actually try to place (needs open position)
"""

import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from binance.um_futures import UMFutures

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

API_KEY    = os.getenv("BINANCE_LIVE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_LIVE_SECRET_KEY", "")

client = UMFutures(key=API_KEY, secret=API_SECRET)

SYMBOL     = "ETHUSDT"
CLOSE_SIDE = "SELL"
FAKE_SL    = 2000.00
FAKE_TP    = 2200.00
FAKE_QTY   = 0.01


def check_account():
    print("\n=== Account Info ===")
    info = client.account()
    print(f"  Can Trade      : {info.get('canTrade')}")
    print(f"  Can Deposit    : {info.get('canDeposit')}")
    pos_mode = client.get_position_mode()
    print(f"  Position mode  : {'Hedge' if pos_mode.get('dualSidePosition') else 'One-Way'}")
    bal = next((b for b in info.get("assets", []) if b["asset"] == "USDT"), {})
    print(f"  USDT Balance   : {bal.get('availableBalance', '?')}")


def try_order(label, **kwargs):
    print(f"\n--- {label} ---")
    print(f"  Params: {kwargs}")
    try:
        resp = client.new_order(**kwargs)
        print(f"  ✅ SUCCESS: orderId={resp.get('orderId')} status={resp.get('status')}")
        # Cancel it immediately
        try:
            client.cancel_order(symbol=SYMBOL, orderId=resp["orderId"])
            print(f"  (cancelled)")
        except Exception:
            pass
        return True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False


def run_tests(real: bool):
    check_account()

    if not real:
        print("\n⚠️  DRY RUN — not sending any orders. Pass --real to actually test.")
        print("\nWould test these order types:")
        print("  1. STOP_MARKET with closePosition=true")
        print("  2. STOP_MARKET with reduceOnly=true + quantity")
        print("  3. STOP with reduceOnly=true + quantity (limit stop)")
        print("  4. TAKE_PROFIT_MARKET with closePosition=true")
        print("  5. TAKE_PROFIT_MARKET with reduceOnly=true + quantity")
        return

    print("\n⚠️  REAL MODE — requires an open ETH position on live account")
    print("    Orders will be placed then immediately cancelled.\n")

    # Test 1: STOP_MARKET with closePosition (current approach)
    try_order(
        "STOP_MARKET + closePosition=true",
        symbol=SYMBOL, side=CLOSE_SIDE, type="STOP_MARKET",
        stopPrice=str(FAKE_SL), closePosition="true",
    )

    # Test 2: STOP_MARKET with reduceOnly + qty
    try_order(
        "STOP_MARKET + reduceOnly=true + qty",
        symbol=SYMBOL, side=CLOSE_SIDE, type="STOP_MARKET",
        stopPrice=str(FAKE_SL), reduceOnly="true", quantity=str(FAKE_QTY),
    )

    # Test 3: STOP_MARKET + workingType + priceProtect + timeInForce
    try_order(
        "STOP_MARKET + workingType=MARK_PRICE + priceProtect + GTC",
        symbol=SYMBOL, side=CLOSE_SIDE, type="STOP_MARKET",
        stopPrice=str(FAKE_SL), quantity=str(FAKE_QTY),
        timeInForce="GTC", workingType="MARK_PRICE", priceProtect="true",
    )

    # Test 4: STOP_MARKET + CONTRACT_PRICE
    try_order(
        "STOP_MARKET + workingType=CONTRACT_PRICE",
        symbol=SYMBOL, side=CLOSE_SIDE, type="STOP_MARKET",
        stopPrice=str(FAKE_SL), quantity=str(FAKE_QTY),
        timeInForce="GTC", workingType="CONTRACT_PRICE", priceProtect="true",
    )

    # Test 5: TAKE_PROFIT_MARKET + workingType + priceProtect
    try_order(
        "TAKE_PROFIT_MARKET + workingType=MARK_PRICE + priceProtect",
        symbol=SYMBOL, side=CLOSE_SIDE, type="TAKE_PROFIT_MARKET",
        stopPrice=str(FAKE_TP), quantity=str(FAKE_QTY),
        timeInForce="GTC", workingType="MARK_PRICE", priceProtect="true",
    )

    # Test 6: STOP (limit stop) with reduceOnly + qty
    try_order(
        "STOP (limit) + reduceOnly=true + qty",
        symbol=SYMBOL, side=CLOSE_SIDE, type="STOP",
        price=str(FAKE_SL - 1), stopPrice=str(FAKE_SL),
        reduceOnly="true", quantity=str(FAKE_QTY),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Actually send test orders")
    args = parser.parse_args()
    run_tests(real=args.real)
