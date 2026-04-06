"""
Diagnostic: test which SL/TP algo order types work on live Binance USDⓈ-M Futures.
Does NOT place any entry order. Dry run by default.

Usage:
    python test_live_sl_tp_algo.py
    python test_live_sl_tp_algo.py --real
    python test_live_sl_tp_algo.py --real --symbol ETHUSDT --side SELL --sl 2000 --tp 2200
    python test_live_sl_tp_algo.py --real --symbol ETHUSDT --side SELL --position-side LONG
"""

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path

from dotenv import load_dotenv
from binance.um_futures import UMFutures

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("BINANCE_LIVE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_LIVE_SECRET_KEY", "")

if not API_KEY or not API_SECRET:
    print("Missing BINANCE_LIVE_API_KEY or BINANCE_LIVE_SECRET_KEY in environment.")
    sys.exit(1)

client = UMFutures(key=API_KEY, secret=API_SECRET)

BASE_URL = "https://fapi.binance.com"


def sign_params(params: dict, secret: str) -> str:
    """
    Build a signed query string for Binance REST signed endpoints.
    """
    query = urllib.parse.urlencode(params, doseq=True)
    signature = hmac.new(
        secret.encode("utf-8"),
        query.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{query}&signature={signature}"


def http_post(path: str, params: dict) -> dict:
    """
    Signed POST request.
    """
    params = dict(params)
    params["timestamp"] = int(time.time() * 1000)
    query = sign_params(params, API_SECRET)
    url = f"{BASE_URL}{path}?{query}"

    req = urllib.request.Request(
        url,
        method="POST",
        headers={"X-MBX-APIKEY": API_KEY},
    )

    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_delete(path: str, params: dict) -> dict:
    """
    Signed DELETE request.
    """
    params = dict(params)
    params["timestamp"] = int(time.time() * 1000)
    query = sign_params(params, API_SECRET)
    url = f"{BASE_URL}{path}?{query}"

    req = urllib.request.Request(
        url,
        method="DELETE",
        headers={"X-MBX-APIKEY": API_KEY},
    )

    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_account() -> dict:
    print("\n=== Account Info ===")
    info = client.account()
    print(f"  Can Trade      : {info.get('canTrade')}")
    print(f"  Can Deposit    : {info.get('canDeposit')}")

    pos_mode = client.get_position_mode()
    dual = bool(pos_mode.get("dualSidePosition"))
    print(f"  Position mode  : {'Hedge' if dual else 'One-Way'}")

    bal = next((b for b in info.get("assets", []) if b["asset"] == "USDT"), {})
    print(f"  USDT Balance   : {bal.get('availableBalance', '?')}")

    return {"dualSidePosition": dual, "account": info}


def validate_args(args, hedge_mode: bool) -> None:
    if hedge_mode and not args.position_side:
        raise ValueError(
            "Hedge Mode is enabled, so you must pass --position-side LONG or SHORT."
        )

    if not hedge_mode and args.position_side:
        print("Note: --position-side ignored in One-Way mode.")

    if args.side not in {"BUY", "SELL"}:
        raise ValueError("--side must be BUY or SELL.")

    if args.sl <= 0 or args.tp <= 0:
        raise ValueError("--sl and --tp must be positive.")

    if args.quantity is not None and args.quantity <= 0:
        raise ValueError("--quantity must be positive if provided.")


def infer_order_intent(side: str, position_side: str | None) -> str:
    """
    Explain what the close order likely means.
    side=SELL usually closes LONG, side=BUY usually closes SHORT.
    """
    if position_side:
        return f"close {position_side}"
    return "close position (depends on current exposure in One-Way mode)"


def build_algo_order(
    *,
    symbol: str,
    side: str,
    order_type: str,
    trigger_price: float,
    working_type: str,
    position_side: str | None,
    quantity: float | None,
    close_position: bool,
) -> dict:
    params = {
        "algoType": "CONDITIONAL",
        "symbol": symbol,
        "side": side,
        "type": order_type,
        "triggerPrice": f"{trigger_price:.8f}".rstrip("0").rstrip("."),
        "workingType": working_type,
    }

    if position_side:
        params["positionSide"] = position_side

    if close_position:
        params["closePosition"] = "true"
    else:
        if quantity is None:
            raise ValueError("quantity is required when closePosition is false.")
        params["quantity"] = f"{quantity:.8f}".rstrip("0").rstrip(".")

    return params


def place_algo_order(label: str, params: dict, real: bool) -> dict | None:
    print(f"\n--- {label} ---")
    print("  Endpoint: POST /fapi/v1/algoOrder")
    print(f"  Params  : {params}")

    if not real:
        print("  DRY RUN : not sent")
        return None

    try:
        resp = http_post("/fapi/v1/algoOrder", params)
        print("  ✅ SUCCESS")
        print(f"  Response: {json.dumps(resp, indent=2)}")
        return resp
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"  ❌ HTTP ERROR: {e.code} {body}")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

    return None


def try_cancel_algo_order(symbol: str, resp: dict | None) -> None:
    """
    Best-effort cancel after a successful placement.

    Binance responses can vary. We try common identifier fields.
    """
    if not resp:
        return

    possible_keys = [
        "algoId",
        "orderId",
        "clientAlgoId",
        "clientOrderId",
    ]

    found_key = None
    found_val = None
    for key in possible_keys:
        if key in resp and resp[key]:
            found_key = key
            found_val = resp[key]
            break

    if not found_key:
        print("  Cancel  : skipped, no known cancel identifier found in response")
        return

    cancel_params = {"symbol": symbol, found_key: found_val}

    try:
        cancel_resp = http_delete("/fapi/v1/algoOrder", cancel_params)
        print("  Cancel  : ✅ SUCCESS")
        print(f"  Response: {json.dumps(cancel_resp, indent=2)}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"  Cancel  : ❌ HTTP ERROR: {e.code} {body}")
    except Exception as e:
        print(f"  Cancel  : ❌ FAILED: {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Actually send test orders")
    parser.add_argument("--symbol", default="ETHUSDT", help="Futures symbol, e.g. ETHUSDT")
    parser.add_argument(
        "--side",
        default="SELL",
        choices=["BUY", "SELL"],
        help="Closing side. SELL usually closes longs, BUY usually closes shorts.",
    )
    parser.add_argument("--sl", type=float, default=2000.0, help="Stop-loss trigger price")
    parser.add_argument("--tp", type=float, default=2200.0, help="Take-profit trigger price")
    parser.add_argument(
        "--working-type",
        default="MARK_PRICE",
        choices=["MARK_PRICE", "CONTRACT_PRICE"],
        help="Trigger source price",
    )
    parser.add_argument(
        "--position-side",
        default=None,
        choices=["LONG", "SHORT"],
        help="Required in Hedge Mode",
    )
    parser.add_argument(
        "--quantity",
        type=float,
        default=None,
        help="Order quantity. Omit to use closePosition=true.",
    )
    parser.add_argument(
        "--no-close-position",
        action="store_true",
        help="Use quantity instead of closePosition=true",
    )
    args = parser.parse_args()

    account_state = check_account()
    hedge_mode = account_state["dualSidePosition"]

    try:
        validate_args(args, hedge_mode)
    except ValueError as e:
        print(f"\nArgument error: {e}")
        sys.exit(1)

    close_position = not args.no_close_position

    if not close_position and args.quantity is None:
        print("\nArgument error: --quantity is required when using --no-close-position")
        sys.exit(1)

    print("\n=== Test Plan ===")
    print(f"  Symbol         : {args.symbol}")
    print(f"  Intent         : {infer_order_intent(args.side, args.position_side)}")
    print(f"  Working Type   : {args.working_type}")
    print(f"  Mode           : {'REAL' if args.real else 'DRY RUN'}")
    print(f"  closePosition  : {close_position}")
    if args.quantity is not None:
        print(f"  Quantity       : {args.quantity}")
    if args.position_side:
        print(f"  Position Side  : {args.position_side}")

    sl_params = build_algo_order(
        symbol=args.symbol,
        side=args.side,
        order_type="STOP_MARKET",
        trigger_price=args.sl,
        working_type=args.working_type,
        position_side=args.position_side if hedge_mode else None,
        quantity=args.quantity,
        close_position=close_position,
    )

    tp_params = build_algo_order(
        symbol=args.symbol,
        side=args.side,
        order_type="TAKE_PROFIT_MARKET",
        trigger_price=args.tp,
        working_type=args.working_type,
        position_side=args.position_side if hedge_mode else None,
        quantity=args.quantity,
        close_position=close_position,
    )

    if not args.real:
        print("\n⚠️ DRY RUN ONLY")
        print("These requests will be prepared but not sent.")
    else:
        print("\n⚠️ REAL MODE")
        print("Orders will be sent to Binance live Futures and then cancellation will be attempted.")
        print("This still touches your real account, because apparently we enjoy stress as a species.")

    sl_resp = place_algo_order("STOP LOSS TEST", sl_params, args.real)
    tp_resp = place_algo_order("TAKE PROFIT TEST", tp_params, args.real)

    if args.real:
        print("\n=== Cancel Attempts ===")
        try_cancel_algo_order(args.symbol, sl_resp)
        try_cancel_algo_order(args.symbol, tp_resp)


if __name__ == "__main__":
    main()