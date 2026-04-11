"""
Compatibility entrypoint for the deployed trading engine.

The VPS trading container still starts `python engine.py --loop`.
We route that entrypoint to the scalping engine so a push to `main`
auto-deploys the scalping strategy without changing the container command.
"""

import argparse

from engine_scalp import run_loop, run_once


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scalp Engine entrypoint")
    parser.add_argument("--loop", action="store_true", help="Run continuous 1-min loop")
    parser.add_argument("--once", action="store_true", help="Run single cycle and exit")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run", help="No order execution")
    args = parser.parse_args()

    if args.loop:
        run_loop(dry_run=args.dry_run)
    else:
        run_once(dry_run=args.dry_run)
