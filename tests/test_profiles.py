#!/usr/bin/env python3
"""Small check for YAML-backed benchmark profile behavior."""
import tempfile
from pathlib import Path

import yaml

from bbox_bench import server
from bbox_bench.server import Service


def main():
    configured = Service.__new__(Service)
    configured.config = yaml.safe_load(Path(server.__file__).with_name("server.yaml").read_text())
    assert set(configured._profiles()) == {"v3-mixed", "v3-int8", "v3-int8-mmse", "v1-mixed", "v2-mixed", "v2-int8", "yolov8n-int8", "yolov8n-hybrid-int8"}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        datasets = root / "datasets"
        datasets.mkdir()
        source = datasets / "clip.mp4"
        source.touch()
        config = root / "server.yaml"
        config.write_text("""runtime_root: {0}\nresults_root: {0}/results\nprofiles:\n  - id: tracker\n    label: Tracker\n    element: rknnnanotrack\n    input_format: BGR\n    requires_roi: true\n    properties: {{enabled: true}}\n  - id: yolo\n    label: YOLO\n    element: rknnyolov8\n    input_format: RGB\n    requires_roi: false\n    properties: {{model: /tmp/model.rknn}}\n""".format(root))
        service = Service(config)
        roi = type("Roi", (), {"text": lambda self: "1,2,3,4"})()
        assert service.trackers["tracker"].gst_args(roi) == ["rknnnanotrack", "roi=1,2,3,4", 'enabled="true"']
        assert service.trackers["yolo"].gst_args(None) == ["rknnyolov8", 'model="/tmp/model.rknn"']
        assert service.inside_dataset_root(source) == source
        try:
            service.inside_dataset_root(root / "outside.mp4")
        except ValueError:
            return
        raise AssertionError("source outside dataset root accepted")


if __name__ == "__main__":
    main()
