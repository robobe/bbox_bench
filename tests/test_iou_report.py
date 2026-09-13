#!/usr/bin/env python3
"""Check local IoU report matching without client or GStreamer dependencies."""
from bbox_bench.compare_nanotracker_csv import compare, read_frame_rows, report
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    truth = [{"pts_ns": "100", "x": "0", "y": "0", "width": "10", "height": "10"}, {"pts_ns": "200", "x": "0", "y": "0", "width": "10", "height": "10"}]
    tracked = [{"pts_ns": "101", "x": "0", "y": "0", "width": "10", "height": "10"}, {"pts_ns": "200", "x": "20", "y": "20", "width": "10", "height": "10"}, {"pts_ns": "999", "x": "0", "y": "0", "width": "1", "height": "1"}]
    summary = compare(truth, tracked, tolerance_ns=5)
    assert summary["matched_rows"] == 2 and summary["unmatched_rows"] == 1
    assert summary["scores"] == [1.0, 0.0]
    assert "mean IoU:          0.5000" in report(summary, "stopped")
    with TemporaryDirectory() as directory:
        path = Path(directory) / "groundtruth.txt"; path.write_text("0,0,10,10\n20,20,10,10\n")
        mapped = read_frame_rows(path)
    assert mapped[0]["frame_id"] == "0" and mapped[1]["x"] == "20"
    assert compare(mapped, [{"frame_id": "1", "x": "20", "y": "20", "width": "10", "height": "10"}])["scores"] == [1.0]


if __name__ == "__main__": main()
