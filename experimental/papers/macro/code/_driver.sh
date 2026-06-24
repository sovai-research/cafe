#!/bin/bash
cd /Users/dereksnow/Sovai/Github/TIMARA/experimental/papers/macro/code
# wait until the fetch process is gone
while pgrep -f "python3 -u fetch_vintages.py" >/dev/null 2>&1; do sleep 10; done
echo "=== fetch finished at $(date +%H:%M:%S) ===" >> ../data/run_log.txt
python3 run_nowcast.py >> ../data/run_log.txt 2>&1
echo "=== run_nowcast exit $? ===" >> ../data/run_log.txt
python3 patch_paper.py >> ../data/run_log.txt 2>&1
echo "=== patch exit $? ===" >> ../data/run_log.txt
