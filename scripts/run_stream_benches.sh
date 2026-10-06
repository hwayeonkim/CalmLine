#!/bin/bash
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
for cfg in "tiny.en 0.5 window" "base.en 0.5 window" "tiny.en 0.25 window" "base.en 0.5 committed"; do
  set -- $cfg
  .venv/bin/python eval/stream_bench.py test $1 $2 $3 2>&1 | grep -v -E "Warning|Loading" > results/audio/log_$1_$2_$3.txt
done
echo ALLDONE > results/audio/stream_done.txt
