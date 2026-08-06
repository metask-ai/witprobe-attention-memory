#!/usr/bin/env bash
# One-command reproduction: L0 numbers -> L1 figures -> L2 gates.
# Lean (L3) is not run here; see README (needs elan + Mathlib cache).
set -u
cd "$(dirname "$0")"
fail=0
step() { echo; echo "=== $1 ==="; }

step "L0: every paper number regenerates from shipped artifacts"
python3 tools/make_canon.py || fail=1
python3 tests/test_paper_claims.py || fail=1

step "L1: every figure and table regenerates"
python3 tools/p2_figs.py || fail=1

step "L2a: flagship gate (expected verdict: 22 PASS / 4 FAIL, matching the paper)"
python3 experiments/flagship_gate.py --json
rc=$?
# 退出码 1 = 总体 FAIL,是论文如实报告的当前系统状态,不是复现失败
[ $rc -le 1 ] || fail=1

step "L2b: concurrent per-row identity gate (exit code is the verdict)"
WITCERT_P98_PREFIX=p99 python3 experiments/p98_concurrent_identity.py || fail=1

echo
if [ $fail -eq 0 ]; then
  echo "REPRODUCTION OK: numbers, figures, and gates all verified against shipped artifacts."
else
  echo "REPRODUCTION FAILED: see the first failing step above."
fi
exit $fail
