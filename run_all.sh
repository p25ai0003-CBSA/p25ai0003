#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
mkdir -p logs
for s in step1_split step2_tree step3_bagging step4_compare step5_toy step6_boost step7_ensembles step8_bias_variance step9_ranges; do
  $PY $s.py 2>&1 | tee logs/$s.log
done
