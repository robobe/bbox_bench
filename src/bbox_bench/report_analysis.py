"""Calculate tracker IoU from a client run artifact."""
import csv
import json
from pathlib import Path

from .compare_nanotracker_csv import box, iou, read_frame_rows


def analyze_report(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    ground_truth = metadata.get("ground_truth") or {}
    if not ground_truth.get("path"):
        raise ValueError("report has no ground-truth path")
    source, tracker_size = metadata.get("source") or {}, metadata.get("tracker_size") or {}
    try:
        scale_x = int(tracker_size["width"]) / int(source["original_width"])
        scale_y = int(tracker_size["height"]) / int(source["original_height"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError("report lacks valid original and tracker dimensions") from error
    expected = {int(row["frame_id"]): row for row in read_frame_rows(ground_truth["path"], **(ground_truth.get("format") or {}))}
    with (directory / "tracker.csv").open(newline="") as source:
        tracked = list(csv.DictReader(source))
    scores, truth_boxes, tracker_boxes, missing_ground_truth, missing_tracker = [], {}, {}, 0, 0
    for row in tracked:
        truth = expected.get(int(row["frame_id"]))
        if truth is None:
            missing_ground_truth += 1
            continue
        x, y, width, height = box(truth); truth_boxes[int(row["frame_id"])] = (x * scale_x, y * scale_y, width * scale_x, height * scale_y)
        if row["metadata_lost"] == "True" or not row["x"]:
            score = 0.0; missing_tracker += 1
        else:
            tracker_boxes[int(row["frame_id"])] = box(row)
            score = iou(truth_boxes[int(row["frame_id"])], tracker_boxes[int(row["frame_id"])])
        scores.append((int(row["frame_id"]), score))
    values = [score for _frame, score in scores]
    return {"metadata": metadata, "scores": scores, "truth_boxes": truth_boxes, "tracker_boxes": tracker_boxes, "frames": len(tracked), "missing_ground_truth": missing_ground_truth,
            "missing_tracker": missing_tracker, "mean_iou": sum(values) / len(values) if values else None,
            "minimum_iou": min(values) if values else None}
