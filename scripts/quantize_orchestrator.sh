#!/bin/bash
# Run a conversion with half of each module's tensor groups offloaded to a worker.
#   SRC        source model dir
#   OUT        quantized output dir
#   WD         working dir (qtensors + checkpoints)
#   PEER       host:port of the worker
#   EXLLAMAV3  path to the patched exllamav3 checkout
#   BITS       default 3.0
#   CODEBOOK   default mcg
#   EXTRA_ARGS extra convert.py flags, e.g. "-r"
set -uo pipefail
: "${SRC:?set SRC}"
: "${OUT:?set OUT}"
: "${WD:?set WD}"
: "${PEER:?set PEER to host:port}"
: "${EXLLAMAV3:?set EXLLAMAV3 to the patched exllamav3 checkout}"
BITS="${BITS:-3.0}"
CODEBOOK="${CODEBOOK:-mcg}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$EXLLAMAV3"
exec python3 convert.py \
  -i "$SRC" -o "$OUT" -w "$WD" \
  -b "$BITS" -cb "$CODEBOOK" -hb 16 -eb 16 -mb 16 -vb 16 \
  --dist-peer "$PEER" $EXTRA_ARGS
