#!/usr/bin/env python3
"""Compare NanoTracker CSV output with frame ground truth using IoU."""
import argparse
import csv
from pathlib import Path

FIELDS = ("pts_ns", "x", "y", "width", "height")


def read_rows(path):
    with Path(path).open(newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or any(name not in reader.fieldnames for name in FIELDS):
            raise ValueError("CSV must contain pts_ns,x,y,width,height columns")
        rows = list(reader)
    try:
        for row in rows:
            pts_ns, x, y, width, height = (int(row[name]) for name in FIELDS)
            if pts_ns < 0 or x < 0 or y < 0 or width < 1 or height < 1: raise ValueError
    except (TypeError, ValueError) as error:
        raise ValueError("CSV rows need non-negative pts_ns/x/y and positive width/height") from error
    return rows


def read_frame_rows(path, delimiter=",", header=False, x=0, y=1, width=2, height=3):
    indexes = (x, y, width, height)
    if not isinstance(delimiter, str) or len(delimiter) != 1 or not isinstance(header, bool) or any(not isinstance(index, int) or index < 0 for index in indexes):
        raise ValueError("ground_truth_format needs a one-character delimiter, boolean header, and non-negative column indexes")
    with Path(path).open(newline="") as source:
        reader = csv.reader(source, delimiter=delimiter)
        if header: next(reader, None)
        rows = []
        try:
            for frame_id, row in enumerate(reader):
                values = [int(row[index]) for index in indexes]
                if values[0] < 0 or values[1] < 0 or values[2] < 1 or values[3] < 1: raise ValueError
                rows.append(dict(zip(("x", "y", "width", "height"), map(str, values)), frame_id=str(frame_id)))
        except (IndexError, ValueError) as error:
            raise ValueError("ground-truth rows do not match the configured x/y/width/height columns") from error
    return rows


def iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = a[2] * a[3] + b[2] * b[3] - intersection
    return intersection / union if union else 0.0


def box(row):
    return tuple(int(row[name]) for name in ("x", "y", "width", "height"))


def compare(expected, predicted, threshold=.5, tolerance_ns=20_000_000):
    if expected and "frame_id" in expected[0]:
        expected_by_frame = {int(row["frame_id"]): row for row in expected}
        scores, unmatched = [], 0
        for row in predicted:
            expected_row = expected_by_frame.get(int(row["frame_id"]))
            if expected_row is None: unmatched += 1
            else: scores.append(iou(box(expected_row), box(row)))
        return {"ground_truth_rows": len(expected), "tracker_rows": len(predicted), "matched_rows": len(scores),
                "unmatched_rows": unmatched, "scores": scores, "threshold": threshold}
    expected_by_pts = {int(row["pts_ns"]): row for row in expected}
    scores, unmatched = [], 0
    for row in predicted:
        pts = int(row["pts_ns"])
        if not expected_by_pts:
            unmatched += 1
            continue
        closest_pts = min(expected_by_pts, key=lambda value: abs(value - pts))
        if abs(closest_pts - pts) > tolerance_ns:
            unmatched += 1
            continue
        scores.append(iou(box(expected_by_pts[closest_pts]), box(row)))
    return {"ground_truth_rows": len(expected), "tracker_rows": len(predicted), "matched_rows": len(scores),
            "unmatched_rows": unmatched, "scores": scores, "threshold": threshold}


def report(summary, state=None):
    scores, threshold = summary["scores"], summary["threshold"]
    lines = ([] if state is None else [f"state:             {state}"])
    lines += [f"ground truth rows: {summary['ground_truth_rows']}", f"tracker rows:      {summary['tracker_rows']}",
              f"matched rows:      {summary['matched_rows']}", f"unmatched rows:    {summary['unmatched_rows']}"]
    if scores:
        lines += [f"mean IoU:          {sum(scores) / len(scores):.4f}", f"minimum IoU:       {min(scores):.4f}",
                  f"IoU >= {threshold:g}:       {sum(score >= threshold for score in scores)}/{len(scores)}",
                  f"initial IoU:       {scores[0]:.4f}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth")
    parser.add_argument("predictions")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tolerance-ns", type=int, default=20_000_000)
    args = parser.parse_args()
    print(report(compare(read_rows(args.ground_truth), read_rows(args.predictions), args.threshold, args.tolerance_ns)), end="")


if __name__ == "__main__":
    main()
