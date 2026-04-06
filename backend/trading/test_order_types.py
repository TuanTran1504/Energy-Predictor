"""
Diagnostic script — checks what order types Binance accepts for this account.
Does NOT place any real orders. Run with:
    python test_order_types.py
"""
import os, sys
from pathlib import Path
from dotenv import load_dotenv
from binance.um_futures import UMFutures
from binance.error import ClientError

load_dotenv(Path(__file__).parent.parent.parent / ".env")

API_KEY    = os.getenv("BINANCE_FUTURES_API_KEY", "")
API_SECRET = os.getenv("BINANCE_FUTURES_SECRET_KEY", "").rstrip("\\")

if not API_KEY or not API_SECRET:
    sys.exit("ERROR: BINANCE_API_KEY / BINANCE_API_SECRET not set in .env")

TESTNET_BASE = "https://testnet.binancefuture.com"
client = UMFutures(key=API_KEY, secret=API_SECRET, base_url=TESTNET_BASE)
print(f"  Using base_url: {TESTNET_BASE}")

print("\n=== Account Info ===")
try:
    acc = client.account()
    print(f"  canTrade    : {acc.get('canTrade')}")
    print(f"  feeTier     : {acc.get('feeTier')}")
    print(f"  multiAssetsMargin: {acc.get('multiAssetsMargin')}")
except Exception as e:
    print(f"  account() failed: {e}")

print("\n=== Exchange Info — ETHUSDT order types ===")
try:
    info = client.exchange_info()
    for sym in info.get("symbols", []):
        if sym["symbol"] == "ETHUSDT":
            print(f"  orderTypes : {sym.get('orderTypes')}")
            print(f"  filters    : {[f['filterType'] for f in sym.get('filters', [])]}")
            break
except Exception as e:
    print(f"  exchange_info() failed: {e}")

print("\n=== Test: STOP order with quantity=0 (expect error, NOT a real order) ===")
# quantity=0 means Binance rejects it before matching — safe to test
# We're testing WHICH error we get, not actually placing an order.
for order_type in ("STOP", "STOP_MARKET", "TAKE_PROFIT", "TAKE_PROFIT_MARKET"):
    try:
        client.new_order(
            symbol="ETHUSDT", side="SELL", type=order_type,
            stopPrice="1000", price="999", quantity="0", reduceOnly="true",
        )
        print(f"  {order_type}: unexpectedly accepted (no error)")
    except ClientError as e:
        print(f"  {order_type}: error_code={e.error_code}  msg={e.error_message}")
    except Exception as e:
        print(f"  {order_type}: {e}")

print("\n=== Position Mode ===")
try:
    mode = client.get_position_mode()
    dual = mode.get("dualSidePosition", False)
    print(f"  dualSidePosition (Hedge Mode): {dual}")
    if dual:
        print("  *** HEDGE MODE DETECTED — reduceOnly is invalid, need positionSide= ***")
    else:
        print("  One-Way Mode — reduceOnly=true should work")
except Exception as e:
    print(f"  get_position_mode() failed: {e}")

print("\nDone.")
