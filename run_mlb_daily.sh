#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CURRENT_STEP_NUMBER=""
CURRENT_STEP_NAME=""

trap 'exit_code=$?; echo "=== Pipeline failed at step ${CURRENT_STEP_NUMBER}: ${CURRENT_STEP_NAME} ==="; exit "$exit_code"' ERR

run_step() {
  CURRENT_STEP_NUMBER="$1"
  CURRENT_STEP_NAME="$2"
  shift 2

  echo "=== Step ${CURRENT_STEP_NUMBER}/5: ${CURRENT_STEP_NAME} ==="
  "$@"
}

run_step 1 "Fetching odds" python3 scripts/fetch_mlb_odds.py --force
run_step 2 "Fetching pitchers" python3 scripts/fetch_mlb_pitchers.py
run_step 3 "Fetching MLB stats" python3 scripts/fetch_mlb_stats.py --season-blend
run_step 4 "Fetching recent form" python3 scripts/fetch_mlb_recent_form.py
run_step 5 "Predicting MLB" python3 scripts/predict_mlb.py

echo "=== MLB daily pipeline complete ==="
