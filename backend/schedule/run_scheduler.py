"""
VPS scheduler service — policy-review loop only.

All market-data ingestion schedules (fear&greed, funding, macro, ETF) are expected
to run on Modal. This process runs only the LLM strategy policy review cadence.
"""

import logging
import os
import time

from run_policy_review import run_policy_review_once

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

try:
    POLICY_REVIEW_INTERVAL_HOURS = max(
        1.0,
        float(os.environ.get("POLICY_REVIEW_INTERVAL_HOURS", "6")),
    )
except (TypeError, ValueError):
    POLICY_REVIEW_INTERVAL_HOURS = 6.0


def run_policy_review_loop():
    """Runs guarded policy review immediately, then every N hours (default 6h)."""
    while True:
        log.info("=== Policy review job starting ===")
        try:
            result = run_policy_review_once()
            log.info(
                "Policy review result: status=%s run_id=%s reason=%s",
                result.get("status"),
                result.get("run_id"),
                result.get("reason"),
            )
        except Exception as e:
            log.error(f"Policy review failed: {e}")
        log.info(
            "=== Policy review job done — next run in %.2fh ===",
            POLICY_REVIEW_INTERVAL_HOURS,
        )
        time.sleep(int(POLICY_REVIEW_INTERVAL_HOURS * 3600))


if __name__ == "__main__":
    log.info("VPS policy-review scheduler starting...")
    run_policy_review_loop()
