#!/usr/bin/env bash
set -euo pipefail
# Execute from repo root or anywhere. Host Python needs no pip dependencies.
PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PACKAGE_DIR"
exec python3 "$PACKAGE_DIR/launch.py" "$@"
