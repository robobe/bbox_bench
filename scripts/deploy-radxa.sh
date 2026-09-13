#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
target="${RADXA_USER:-radxa}@${RADXA_HOST:-radxa}"
remote_root=${BBOX_BENCH_ROOT:-/home/radxa/bbox_bench}

ssh "$target" "mkdir -p '$remote_root'"
rsync -a --delete --exclude .venv --exclude .uv-bin --exclude __pycache__ --exclude dist "$root/" "$target:$remote_root/"
ssh "$target" "chmod +x '$remote_root/scripts/install-server.sh'; '$remote_root/scripts/install-server.sh'"
