#!/bin/bash
# Apply the distributed-quantization patch to an exllamav3 checkout.
#   usage: ./scripts/install.sh /path/to/exllamav3
set -euo pipefail
TARGET="${1:?usage: install.sh /path/to/exllamav3}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
CM="$TARGET/exllamav3/conversion/convert_model.py"

[ -f "$CM" ] || { echo "not an exllamav3 checkout: $TARGET" >&2; exit 1; }

echo "applying patch to $CM"
git -C "$TARGET" apply --check "$HERE/patch/0001-distributed-quantization.patch" \
  || { echo "patch does not apply cleanly; is the checkout at v1.6.7 (8c593e4e)?" >&2; exit 1; }
git -C "$TARGET" apply "$HERE/patch/0001-distributed-quantization.patch"

echo "installing dist_worker.py"
cp "$HERE/src/dist_worker.py" "$TARGET/exllamav3/conversion/dist_worker.py"

echo "done. --dist-peer and the [phases] line are available; nothing else changed."
