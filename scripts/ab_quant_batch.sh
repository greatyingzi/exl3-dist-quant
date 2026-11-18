#!/bin/bash
# A/B one or more layers between two batched-quantization settings.
#
#   SRC          source model dir
#   EXLLAMAV3    patched exllamav3 checkout
#   MAX_MODULE   stop after this module index (so both runs convert the same layers)
#   BASE_COLS/BASE_STACK   first setting   (default 32768 / 16)
#   ALT_COLS/ALT_STACK     second setting  (default 65536 / 32)
#   ROOT         scratch dir (default /tmp/ab-quant-batch)
#
# Both runs are identical apart from the batch size, so any tensor difference
# comes from the batch layout alone.
set -uo pipefail
: "${SRC:?set SRC}"
: "${EXLLAMAV3:?set EXLLAMAV3}"
: "${MAX_MODULE:?set MAX_MODULE, e.g. 5 to cover layers 0-3}"
BASE_COLS="${BASE_COLS:-32768}"
BASE_STACK="${BASE_STACK:-16}"
ALT_COLS="${ALT_COLS:-65536}"
ALT_STACK="${ALT_STACK:-32}"
ROOT="${ROOT:-/tmp/ab-quant-batch}"
HERE="$(cd "$(dirname "$0")" && pwd)"

run() {
  local tag="$1" cols="$2" stack="$3"
  local wd="$ROOT/$tag-wd" out="$ROOT/$tag-out" log="$ROOT/$tag.log"
  rm -rf "$wd" "$out"
  mkdir -p "$wd" "$out"
  echo "== $tag: batch_cols=$cols stack=$stack =="
  ( cd "$EXLLAMAV3" && python3 convert.py -i "$SRC" -o "$out" -w "$wd" \
      -b 3.0 -cb mcg -hb 16 -eb 16 -mb 16 -vb 16 \
      --quant_batch_cols "$cols" --quant_stack "$stack" \
      --max_module "$MAX_MODULE" ) > "$log" 2>&1
  echo "   exit=$?"
  echo "   per-layer time:"
  tr -d "\r" < "$log" | grep -E "layers\.[0-9]+ +bpw" | sed -E "s/.*layers\.([0-9]+).*\[([0-9.]+) s\].*/     layer \1: \2 s/"
  echo "   quantize phase:"
  tr -d "\r" < "$log" | grep "\[phases\]" | sed -E "s/.*quantize=([0-9.]+)s.*/     \1 s/" | tail -3
}

run base "$BASE_COLS" "$BASE_STACK"
run alt "$ALT_COLS" "$ALT_STACK"

echo
echo "=== tensor comparison (base vs alt) ==="
python3 "$HERE/compare_qtensors.py" "$ROOT/base-wd/qtensors" "$ROOT/alt-wd/qtensors"
rc=$?

echo
echo "If RESULT is IDENTICAL and the alt run is faster, the batch increase is free."
echo "Logs: $ROOT/base.log , $ROOT/alt.log"
exit $rc
