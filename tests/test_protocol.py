#!/usr/bin/env python3
"""Small no-framework check for the benchmark command boundary."""
from bbox_bench.protocol import Roi, RunCommand, metadata_packet


def main():
    roi = Roi.from_json({"x": 1, "y": 2, "width": 3, "height": 4})
    assert roi.text() == "1,2,3,4"
    command = RunCommand.from_json({"source_path": "/home/radxa/a.mp4", "tracker_id": "v3-int8", "roi": roi.__dict__,
                                    "client_host": "10.0.0.2", "metadata_port": 5005, "playback_fps": "30"})
    assert command.json()["roi"]["width"] == 3
    detector = RunCommand.from_json({"source_path": "/home/radxa/a.mp4", "tracker_id": "yolov8n-int8", "roi": None,
                                     "client_host": "10.0.0.2", "metadata_port": 5005, "playback_fps": "30"})
    assert detector.roi is None
    assert RunCommand.from_json({"source_path": "/home/radxa/a.mp4", "tracker_id": "yolov8n-int8", "roi": None,
                                 "client_host": "10.0.0.2", "metadata_port": 5005, "playback_fps": "30", "single_frame": True}).single_frame
    assert metadata_packet(b"123,yolo8,1,2,3,4,0,0.9,6,7\n") == (7, 123, "yolo8", 1, 2, 3, 4, "0", "0.9", 6)
    assert metadata_packet(b"123,frame,,,,,,,,7\n") == (7, 123, "frame", None, None, None, None, "", "", None)
    try:
        Roi.from_json({"x": 0, "y": 0, "width": 0, "height": 1})
    except ValueError:
        return
    raise AssertionError("invalid ROI accepted")


if __name__ == "__main__":
    main()
