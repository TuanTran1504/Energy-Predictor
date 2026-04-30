#!/bin/sh
# Launch VPS policy-review loop only.
# Data ingestion schedules remain on Modal.
set -e

python run_scheduler.py
