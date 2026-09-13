#!/usr/bin/env python3
"""Check IoU analysis of a client-side report."""
from pathlib import Path
from tempfile import TemporaryDirectory

from bbox_bench.report_analysis import analyze_report
from bbox_bench.run_report import write_run_report


def main():
    with TemporaryDirectory() as directory:
        root = Path(directory); truth = root / "truth.txt"; truth.write_text("300,90,300,90\n0,0,10,10\n")
        rows = [{"frame_id": 0, "pts_ns": 0, "metadata_lost": False, "kind": "tracker", "x": 100, "y": 30, "width": 100, "height": 30, "initialized": "1", "confidence": "", "class_id": ""}, {"frame_id": 1, "pts_ns": 1, "metadata_lost": True, "kind": "", "x": "", "y": "", "width": "", "height": "", "initialized": "", "confidence": "", "class_id": ""}]
        report = write_run_report(root / "report", {"source": {"original_width": 1920, "original_height": 1080}, "tracker_size": {"width": 640, "height": 360}, "ground_truth": {"path": str(truth), "format": {}}}, rows)
        result = analyze_report(report)
    assert result["scores"] == [(0, 1.0), (1, 0.0)] and result["truth_boxes"][0] == (100.0, 30.0, 100.0, 30.0) and result["tracker_boxes"][0] == (100, 30, 100, 30) and result["mean_iou"] == .5 and result["missing_tracker"] == 1


if __name__ == "__main__": main()
