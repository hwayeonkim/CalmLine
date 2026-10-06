#!/bin/bash
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
while [ ! -f results/audio/stream_done.txt ]; do sleep 10; done
for cfg in "base.en 0.5 window" "small.en 0.5 window"; do
  set -- $cfg
  .venv/bin/python eval/stream_bench.py test $1 $2 $3 2>&1 | grep -v -E "Warning|Loading" > results/audio/log2_$1_$2_$3.txt
done
echo ALLDONE > results/audio/stream_done2.txt
