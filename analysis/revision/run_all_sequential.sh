#!/bin/bash
# Run all 5 revision recomputes sequentially so no two heavy jobs share memory.
# Ordered fastest first so early signals land quickly.
#
# Every script is resumable — if this wrapper (or the machine) dies mid-run,
# re-invoking it picks up where the last patient's CSV row was written.
#
# Launch pattern:
#   nohup caffeinate -i -m bash analysis/revision/run_all_sequential.sh > /tmp/revision_wrapper.log 2>&1 &
#   disown
#
# The nohup+disown detaches from the terminal; caffeinate blocks idle+disk sleep
# for the wrapper's lifetime. Clamshell-sleep suspends everything; on wake the
# scripts resume normally (macOS pauses, doesn't kill, on user-initiated sleep).

set -u
PROJECT_ROOT="/Users/anshshetty/Library/Mobile Documents/com~apple~CloudDocs/ExoHand"
cd "$PROJECT_ROOT"
LOG_DIR="/tmp"
DONE_MARK="/tmp/revision_all_done.marker"

echo "=== revision run started $(date) ===" > /tmp/revision_wrapper.log

run_one () {
  local script="$1"
  local log="$2"
  local label="$3"
  echo "" >> /tmp/revision_wrapper.log
  echo "[$(date '+%H:%M:%S')] START $label" >> /tmp/revision_wrapper.log
  python3 -u "analysis/revision/$script" > "$log" 2>&1
  local rc=$?
  echo "[$(date '+%H:%M:%S')] END $label (rc=$rc)" >> /tmp/revision_wrapper.log
  return 0
}

run_one recompute_deployed_feature_audit.py    "$LOG_DIR/deployed_audit.log"  "feature audit"
run_one recompute_B2_cross_session.py          "$LOG_DIR/expB2.log"           "B2 cross-session"
run_one recompute_B1_bad_cal_robustness.py     "$LOG_DIR/expB1.log"           "B1 bad-cal"
run_one recompute_B_heavy_hgb.py               "$LOG_DIR/expB.log"            "Exp B heavy HGB"
run_one recompute_A_lda_at_cal_sizes.py        "$LOG_DIR/expA.log"            "Exp A cal-size sweep"

echo "" >> /tmp/revision_wrapper.log
echo "[$(date '+%H:%M:%S')] ALL DONE" >> /tmp/revision_wrapper.log
touch "$DONE_MARK"
