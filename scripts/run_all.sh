#!/bin/bash
# Full benchmark: 3 models x 3 seeds training + dumps, then CRF + analysis.
# Long stages are wrapped in caffeinate so the Mac never sleeps mid-run.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONHASHSEED=42
export HF_HUB_OFFLINE=1

SEEDS="${SEEDS:-13 42 123}"
MODELS="${MODELS:-kazbert kazroberta xlmr}"

for m in $MODELS; do
  for s in $SEEDS; do
    if [ -f "results/dumps/${m}_${s}_test.jsonl" ]; then
      echo "skip ${m}_${s} (dump exists)"; continue
    fi
    caffeinate -is $PY -m kzcalib.train --config configs/$m.yaml --seed $s
  done
done

# Optionally retrain the CRF baseline first (TRAIN_CRF=1); by default the
# bundled model under kzcalib/crf_baseline/ is used as released.
if [ "${TRAIN_CRF:-0}" = "1" ]; then
  caffeinate -is $PY scripts/train_crf.py
fi

caffeinate -is $PY -m kzcalib.analyze --crf
for m in $MODELS; do
  for s in $SEEDS; do
    $PY -m kzcalib.analyze --run ${m}_${s}
  done
done
echo "all done"
