#!/usr/bin/env bash
set -euo pipefail

runtime_root=${RUNTIME_ROOT:-/home/radxa/gst-rknn}
test -d "$runtime_root/plugins" -a -d "$runtime_root/models" -a -d "$runtime_root/datasets"
GST_PLUGIN_PATH="$runtime_root/plugins" gst-inspect-1.0 roi2csv >/dev/null
GST_PLUGIN_PATH="$runtime_root/plugins" gst-inspect-1.0 roi2udp >/dev/null
echo "bbox_bench runtime ready: $runtime_root"
