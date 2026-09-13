#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec "$root/.venv/bin/bbox-bench-server" "$@"
