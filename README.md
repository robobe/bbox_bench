# bbox_bench

Metadata-paced client/server benchmarks for GStreamer ROI metadata. `bbox_bench`
uses the deployed `gst-rknn` runtime for plugins, models, diagnostics, and
datasets; it does not build or copy them.

## Install and run

The client and server are one wheel with optional dependencies:

```sh
pip install 'bbox-bench[client]'
pip install 'bbox-bench[server]'
```

For a source checkout, use uv and retain system GStreamer/PyGObject packages:

```sh
uv venv --system-site-packages
uv sync --extra client
uv run --active bbox-bench-client
```

On Radxa, run `scripts/deploy-radxa.sh`; it installs the `server` extra into
`/home/radxa/bbox_bench/.venv`. The server reads models, plugins, and datasets
from `runtime_root: /home/radxa/gst-rknn` and saves runs below
`/home/radxa/bbox_bench/runs`.

The Radxa server runs one selected tracker/detector pipeline at a time and sends
only metadata over UDP. The client decodes its matching local dataset and renders
the returned ROIs. Run artifacts include the request, pipeline, prediction CSV,
summary, and ground-truth IoU report when applicable.

The server browser roots at `$runtime_root/datasets`; by default this is
`/home/radxa/gst-rknn/datasets`. Configure another runtime location with
`--config`. Both programs log colourized diagnostics with timestamp, model,
source line, and exception traceback.

Run `scripts/preflight.sh` on the Radxa before starting the server. The client
keeps its existing preset file at
`~/.config/gst-rknn/nanotracker-benchmark-last-run.yaml` for compatibility.
