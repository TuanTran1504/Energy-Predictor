#!/bin/sh
# Launch both scheduler processes.
# run_scheduler.py  → fear_greed_index, funding_rates, macro_events
# run_macro_etf_ingestor.py → macro_releases, etf_flows
set -e

python run_scheduler.py &
PID1=$!
python run_macro_etf_ingestor.py &
PID2=$!

# Wait for either process to exit; if one dies the container restarts.
wait $PID1 $PID2
