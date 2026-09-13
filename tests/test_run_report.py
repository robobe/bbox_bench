#!/usr/bin/env python3
"""Check the client run-artifact layout without GUI dependencies."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from bbox_bench.run_report import write_run_report


def main():
    rows = [{"frame_id": 0, "pts_ns": 0, "metadata_lost": False, "kind": "nanotracker", "x": 1, "y": 2, "width": 3, "height": 4, "initialized": "1", "confidence": "", "class_id": ""}, {"frame_id": 1, "pts_ns": 10, "metadata_lost": True, "kind": "", "x": "", "y": "", "width": "", "height": "", "initialized": "", "confidence": "", "class_id": ""}]
    with TemporaryDirectory() as directory:
        report = write_run_report(Path(directory) / "run", {"run_id": "run", "ground_truth": {"path": "/data/truth.txt"}}, rows)
        assert json.loads((report / "metadata.json").read_text())["ground_truth"]["path"] == "/data/truth.txt"
        with (report / "tracker.csv").open(newline="") as source: saved = list(csv.DictReader(source))
    assert saved[0]["x"] == "1" and saved[1]["metadata_lost"] == "True"


if __name__ == "__main__": main()
