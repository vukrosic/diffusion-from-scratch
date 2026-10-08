#!/bin/bash
# Runs the checks and the eight steps. The first run trains the models (25 to 30 minutes on a laptop CPU); later runs reuse runs/.
set -e
cd "$(dirname "$0")"
python3 tests.py
for step in step1_noise step2_jump step3_learn step4_sample step5_shortcut step6_digits step7_prompt step8_small_room; do
  echo
  echo "=== $step.py"
  python3 $step.py
done
