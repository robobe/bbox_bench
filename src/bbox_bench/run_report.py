"""Write client-side tracker run artifacts."""
import csv
import json
from pathlib import Path


FIELDS = ("frame_id", "pts_ns", "metadata_lost", "kind", "x", "y", "width", "height", "initialized", "confidence", "class_id")


def write_run_report(directory, metadata, rows):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (directory / "tracker.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    return directory
