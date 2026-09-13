#!/usr/bin/env python3
"""Compare NanoTracker CSV output with frame ground truth using IoU."""
import argparse
import csv
from pathlib import Path


def read_rows(path):
    with Path(path).open(newline="") as source:
        return list(csv.DictReader(source))


def iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = a[2] * a[3] + b[2] * b[3] - intersection
    return intersection / union if union else 0.0


def box(row):
    return tuple(int(row[name]) for name in ("x", "y", "width", "height"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth")
    parser.add_argument("predictions")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tolerance-ns", type=int, default=20_000_000)
    args = parser.parse_args()
    expected, predicted = read_rows(args.ground_truth), read_rows(args.predictions)
    expected_by_pts = {int(row["pts_ns"]): row for row in expected}
    scores, unmatched = [], 0
    for row in predicted:
        pts = int(row["pts_ns"])
        closest_pts = min(expected_by_pts, key=lambda value: abs(value - pts))
        if abs(closest_pts - pts) > args.tolerance_ns:
            unmatched += 1
            continue
        scores.append(iou(box(expected_by_pts[closest_pts]), box(row)))
    passing = sum(score >= args.threshold for score in scores)
    print(f"ground truth rows: {len(expected)}")
    print(f"tracker rows:      {len(predicted)}")
    print(f"matched rows:      {len(scores)}")
    print(f"unmatched rows:    {unmatched}")
    if scores:
        print(f"mean IoU:          {sum(scores) / len(scores):.4f}")
        print(f"minimum IoU:       {min(scores):.4f}")
        print(f"IoU >= {args.threshold:g}:       {passing}/{len(scores)}")
        print(f"initial IoU:       {scores[0]:.4f}")


if __name__ == "__main__":
    main()
