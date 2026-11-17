#!/bin/bash
# Start the remote quantization worker.
#   SRC   source model dir (must match the orchestrator's model)
#   WD    working dir for the worker job
#   OUT   output dir (unused by the worker, required by exllamav3 argument checks)
#   PORT  TCP port to listen on
#   EXLLAMAV3  path to the patched exllamav3 checkout
set -uo pipefail
: "${SRC:?set SRC to the source model directory}"
: "${WD:?set WD}"
: "${OUT:?set OUT}"
PORT="${PORT:-9311}"
EXLLAMAV3="${EXLLAMAV3:?set EXLLAMAV3 to the patched exllamav3 checkout}"
mkdir -p "$WD" "$OUT"

# dist_worker imports exllamav3.conversion.convert_model, so run from the checkout
cd "$EXLLAMAV3"
exec python3 exllamav3/conversion/dist_worker.py \
  --src "$SRC" --wd "$WD" --out "$OUT" --port "$PORT"
