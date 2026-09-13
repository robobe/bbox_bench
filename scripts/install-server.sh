#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
uv_bin=${UV_BIN:-"$root/.uv-bin/uv"}
if ! test -x "$uv_bin"; then
  curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$root/.uv-bin" sh
fi
"$uv_bin" venv --clear --system-site-packages --no-managed-python --python /usr/bin/python3 "$root/.venv"
"$uv_bin" pip install --python "$root/.venv/bin/python" --strict "$root[server]"
"$root/.venv/bin/bbox-bench-server" --check-config
