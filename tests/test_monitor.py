#!/usr/bin/env python3
"""Small check for Radxa monitor parsing and alert thresholds."""
import tempfile
from pathlib import Path

from bbox_bench.server import Monitor


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(value)


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); proc, thermal, devfreq = root / "stat", root / "thermal", root / "devfreq"
        write(proc, "cpu  100 0 100 700 100 0 0 0 0 0\n")
        write(thermal / "thermal_zone0/type", "soc-thermal\n"); write(thermal / "thermal_zone0/temp", "86000\n")
        write(thermal / "thermal_zone1/type", "gpu-thermal\n"); write(thermal / "thermal_zone1/temp", "70000\n")
        write(devfreq / "gpu/name", "fde60000.gpu\n"); write(devfreq / "gpu/load", "96@600000000Hz\n")
        write(devfreq / "npu/name", "fde40000.npu\n"); write(devfreq / "npu/load", "20@600000000Hz\n")
        monitor = Monitor({"thresholds": {"cpu_load_pct": 90, "gpu_load_pct": 95, "npu_load_pct": 95, "soc_temp_c": 85, "gpu_temp_c": 85}}, proc, thermal, devfreq)
        write(proc, "cpu  200 0 100 700 100 0 0 0 0 0\n")
        data = monitor.sample()
        assert data["cpu"] == {"load_pct": 100.0, "temperature_c": 86.0, "alert": True}
        assert data["gpu"]["alert"] and data["npu"]["alert"]
        assert data["gpu"]["temperature_c"] == 70.0


if __name__ == "__main__": main()
