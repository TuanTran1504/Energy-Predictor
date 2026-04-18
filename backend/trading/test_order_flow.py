"""
test_order_flow.py — Full end-to-end test of every order placement in engine_live.py.

Tests:
  1. Entry MARKET order (BUY)
  2. SL via algo order (STOP_MARKET, closePosition=true)
  3. TP1 via algo order (TAKE_PROFIT_MARKET, quantity=half_qty)
  4. TP2 via algo order (TAKE_PROFIT_MARKET, quantity=half_qty)
  5. List open algo orders (verify parse fix)
  6. Cancel SL algo order by algoId
  7. Place break-even SL (new STOP_MARKET at entry price)
  8. Cancel all remaining algo orders
  9. Close position (MARKET SELL, reduceOnly)

Run on VPS or locally:
  cd backend/trading && python test_order_flow.py
"""

import hashlib
import hmac
import json
import math
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

from binance.um_futures import UMFutures
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

API_KEY    = os.getenv("BINANCE_LIVE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_LIVE_SECRET_KEY", "")

SYMBOL     = "SOLUSDT"
BASE       = "SOL"
QTY        = 0.1
LEVERAGE   = 5
PRICE_PREC = 2
QTY_PREC   = 1

PASS = "✓ PASS"
FAIL = "✗ FAIL"


# ── Helpers (same as engine_live.py) ─────────────────────────────────────────

def _sign(params: dict) -> str:
    query = urllib.parse.urlencode(params)
    sig   = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    return f"{query}&signature={sig}"


def algo_order(symbol, side, order_type, trigger_price, quantity=None):
    params = {
        "algoType":     "CONDITIONAL",
        "symbol":       symbol,
        "side":         side,
        "type":         order_type,
        "triggerPrice": f"{trigger_price:.{PRICE_PREC}f}",
        "workingType":  "MARK_PRICE",
        "timestamp":    int(time.time() * 1000),
    }
    if quantity is not None:
        params["quantity"] = f"{quantity:.{QTY_PREC}f}"
    else:
        params["closePosition"] = "true"
    url = f"https://fapi.binance.com/fapi/v1/algoOrder?{_sign(params)}"
    req = urllib.request.Request(url, method="POST", headers={"X-MBX-APIKEY": API_KEY})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def list_open_algo_orders(symbol):
    params = {"symbol": symbol, "timestamp": int(time.time() * 1000)}
    url = f"https://fapi.binance.com/fapi/v1/openAlgoOrders?{_sign(params)}"
    req = urllib.request.Request(url, headers={"X-MBX-APIKEY": API_KEY})
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read())
    # Fix: API returns list directly
    if isinstance(data, list):
        return data
    return data.get("orders", [])


def cancel_algo_order(algo_id):
    params = {"algoId": algo_id, "timestamp": int(time.time() * 1000)}
    url = f"https://fapi.binance.com/fapi/v1/algoOrder?{_sign(params)}"
    req = urllib.request.Request(url, method="DELETE", headers={"X-MBX-APIKEY": API_KEY})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def cancel_all_algo_orders(symbol):
    orders = list_open_algo_orders(symbol)
    cancelled = 0
    for ao in orders:
        try:
            cancel_algo_order(ao["algoId"])
            cancelled += 1
        except Exception as e:
            print(f"     Warning: cancel algo {ao['algoId']} failed: {e}")
    return cancelled


# ── Test runner ───────────────────────────────────────────────────────────────

results = []

def check(label, ok, detail=""):
    status = PASS if ok else FAIL
    results.append((status, label))
    print(f"  {status}  {label}" + (f" — {detail}" if detail else ""))
    return ok


def main():
    client = UMFutures(key=API_KEY, secret=API_SECRET)

    print("\n" + "="*60)
    print("  TRADING ENGINE ORDER FLOW TEST")
    print("="*60)

    # Get current price
    ticker = client.ticker_price(symbol=SYMBOL)
    price  = float(ticker["price"])
    print(f"\n  SOL mark price: {price}")
    client.change_leverage(symbol=SYMBOL, leverage=LEVERAGE)

    # Price levels
    half  = math.floor((QTY / 2) * 10) / 10 or QTY
    sl    = round(price * 0.985, PRICE_PREC)   # -1.5%
    tp1   = round(price * 1.010, PRICE_PREC)   # +1.0%
    tp2   = round(price * 1.020, PRICE_PREC)   # +2.0%
    be    = price                               # break-even = entry

    print(f"  Levels: entry≈{price}  SL={sl}  TP1={tp1}  TP2={tp2}  half={half}")
    print()

    entry = price
    order_id = None
    sl_algo_id = None
    tp1_algo_id = None
    tp2_algo_id = None

    # ── 1. Entry order ────────────────────────────────────────────────────────
    print("[1] Entry MARKET BUY")
    try:
        order    = client.new_order(symbol=SYMBOL, side="BUY", type="MARKET", quantity=QTY)
        order_id = str(order.get("orderId", ""))
        entry    = float(order.get("avgPrice") or 0)
        if not entry:
            time.sleep(0.3)
            filled = client.query_order(symbol=SYMBOL, orderId=order_id)
            entry  = float(filled.get("avgPrice") or 0) or price
        be = round(entry, PRICE_PREC)
        check("Entry order placed", bool(order_id), f"orderId={order_id}")
        check("Entry avgPrice non-zero", entry > 0, f"avgPrice={entry}")
    except Exception as e:
        check("Entry order placed", False, str(e))
        print("\n  Cannot continue without entry. Aborting.")
        return

    time.sleep(0.5)

    # ── 2. SL via algo order ──────────────────────────────────────────────────
    print("\n[2] SL via algo order (STOP_MARKET, closePosition=true)")
    try:
        resp = algo_order(SYMBOL, "SELL", "STOP_MARKET", sl)
        sl_algo_id = resp.get("algoId")
        check("SL algo order placed", bool(sl_algo_id), f"algoId={sl_algo_id}")
    except Exception as e:
        check("SL algo order placed", False, str(e))

    # ── 3. TP1 via algo order ─────────────────────────────────────────────────
    print("\n[3] TP1 via algo order (TAKE_PROFIT_MARKET, quantity=half)")
    try:
        resp = algo_order(SYMBOL, "SELL", "TAKE_PROFIT_MARKET", tp1, quantity=half)
        tp1_algo_id = resp.get("algoId")
        check("TP1 algo order placed", bool(tp1_algo_id), f"algoId={tp1_algo_id}")
    except Exception as e:
        check("TP1 algo order placed", False, str(e))

    # ── 4. TP2 via algo order ─────────────────────────────────────────────────
    print("\n[4] TP2 via algo order (TAKE_PROFIT_MARKET, quantity=half)")
    try:
        resp = algo_order(SYMBOL, "SELL", "TAKE_PROFIT_MARKET", tp2, quantity=half)
        tp2_algo_id = resp.get("algoId")
        check("TP2 algo order placed", bool(tp2_algo_id), f"algoId={tp2_algo_id}")
    except Exception as e:
        check("TP2 algo order placed", False, str(e))

    time.sleep(0.5)

    # ── 5. List open algo orders ──────────────────────────────────────────────
    print("\n[5] List open algo orders (verify parse fix)")
    try:
        orders = list_open_algo_orders(SYMBOL)
        check("list_open_algo_orders returns list", isinstance(orders, list), f"type={type(orders).__name__}")
        algo_ids = [o.get("algoId") for o in orders]
        check("SL algo visible in list", sl_algo_id in algo_ids,   f"ids={algo_ids}")
        check("TP1 algo visible in list", tp1_algo_id in algo_ids, f"ids={algo_ids}")
        check("TP2 algo visible in list", tp2_algo_id in algo_ids, f"ids={algo_ids}")
        print(f"     Orders found: {[{'algoId': o.get('algoId'), 'type': o.get('type'), 'side': o.get('side'), 'triggerPrice': o.get('triggerPrice')} for o in orders]}")
    except Exception as e:
        check("list_open_algo_orders", False, str(e))
        orders = []

    # ── 6. Cancel SL algo order ───────────────────────────────────────────────
    print("\n[6] Cancel original SL algo order")
    if sl_algo_id:
        try:
            cancel_algo_order(sl_algo_id)
            time.sleep(0.3)
            orders_after = list_open_algo_orders(SYMBOL)
            ids_after = [o.get("algoId") for o in orders_after]
            check("SL algo cancelled", sl_algo_id not in ids_after, f"remaining ids={ids_after}")
        except Exception as e:
            check("SL algo cancelled", False, str(e))
    else:
        check("SL algo cancelled", False, "no sl_algo_id to cancel")

    # ── 7. Place break-even SL ────────────────────────────────────────────────
    # Use entry - 0.1% as trigger to ensure mark price is above it (valid SELL STOP).
    # In production the price is above entry when break-even fires, so this simulates that.
    be_trigger = round(entry * 0.999, PRICE_PREC)
    print(f"\n[7] Place break-even SL at {be_trigger} (entry={entry}, using entry-0.1% to avoid immediate trigger)")
    try:
        resp = algo_order(SYMBOL, "SELL", "STOP_MARKET", be_trigger)
        be_algo_id = resp.get("algoId")
        check("Break-even SL placed", bool(be_algo_id), f"algoId={be_algo_id}")
        time.sleep(0.3)
        orders_be = list_open_algo_orders(SYMBOL)
        ids_be = [o.get("algoId") for o in orders_be]
        check("Break-even SL visible in list", be_algo_id in ids_be, f"ids={ids_be}")
        # Verify it is correctly identified as SL (trigger < entry, side=SELL)
        be_order = next((o for o in orders_be if o.get("algoId") == be_algo_id), None)
        if be_order:
            ao_trigger = float(be_order.get("triggerPrice") or 0)
            is_sl = be_order.get("side") == "SELL" and ao_trigger < entry
            check("Break-even SL identified by price logic", is_sl,
                  f"side={be_order.get('side')} trigger={ao_trigger} entry={entry}")
    except Exception as e:
        check("Break-even SL placed", False, str(e))

    # ── 8. Cancel all algo orders ─────────────────────────────────────────────
    print("\n[8] Cancel all remaining algo orders")
    try:
        n = cancel_all_algo_orders(SYMBOL)
        time.sleep(0.3)
        remaining = list_open_algo_orders(SYMBOL)
        check("All algo orders cancelled", len(remaining) == 0, f"cancelled={n} remaining={len(remaining)}")
    except Exception as e:
        check("Cancel all algo orders", False, str(e))

    # ── 9. Close position ─────────────────────────────────────────────────────
    print("\n[9] Close position (MARKET SELL, reduceOnly)")
    try:
        resp = client.new_order(
            symbol=SYMBOL, side="SELL", type="MARKET",
            quantity=QTY, reduceOnly="true",
        )
        close_id    = str(resp.get("orderId", ""))
        close_price = float(resp.get("avgPrice") or 0)
        if not close_price and close_id:
            time.sleep(0.3)
            filled = client.query_order(symbol=SYMBOL, orderId=close_id)
            close_price = float(filled.get("avgPrice") or 0)
        check("Position closed", bool(close_id), f"orderId={close_id}")
        check("Close avgPrice non-zero", close_price > 0, f"avgPrice={close_price}")
        if close_price and entry:
            pnl = (close_price - entry) * QTY
            print(f"     pnl={pnl:+.4f} USDT  entry={entry}  exit={close_price}")
    except Exception as e:
        check("Position closed", False, str(e))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    passed = sum(1 for s, _ in results if s == PASS)
    total  = len(results)
    print(f"  RESULT: {passed}/{total} passed")
    print("="*60)
    for status, label in results:
        print(f"  {status}  {label}")
    print()


if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        print("ERROR: BINANCE_LIVE_API_KEY / BINANCE_LIVE_SECRET_KEY not set")
        exit(1)
    main()
