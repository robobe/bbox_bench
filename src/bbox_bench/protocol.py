"""Validated JSON commands shared by bbox_bench server and client."""
from dataclasses import asdict, dataclass


def _text(data, key):
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be non-empty text")
    return value


def _port(data, key):
    value = data.get(key)
    if not isinstance(value, int) or not 1 <= value <= 65535:
        raise ValueError(f"{key} must be a port number")
    return value


def metadata_packet(payload):
    fields = payload.decode().strip().split(",")
    if len(fields) != 10:
        raise ValueError("metadata packet is incomplete")
    frame_id, pts_ns, roi_type = int(fields[9]), int(fields[0]), fields[1]
    if roi_type == "frame":
        return frame_id, pts_ns, roi_type, None, None, None, None, "", "", None
    return (frame_id, pts_ns, roi_type, *(int(value) for value in fields[2:6]), fields[6], fields[7],
            int(fields[8]) if fields[8] else None)


@dataclass(frozen=True)
class Roi:
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_json(cls, data):
        if not isinstance(data, dict):
            raise ValueError("roi must be an object")
        values = [data.get(name) for name in ("x", "y", "width", "height")]
        if not all(isinstance(value, int) for value in values) or values[0] < 0 or values[1] < 0 or values[2] < 1 or values[3] < 1:
            raise ValueError("roi must have non-negative x/y and positive width/height")
        return cls(*values)

    def text(self):
        return f"{self.x},{self.y},{self.width},{self.height}"


@dataclass(frozen=True)
class RunCommand:
    source_path: str
    tracker_id: str
    roi: Roi | None
    client_host: str
    metadata_port: int
    playback_fps: str
    ground_truth: str | None = None
    single_frame: bool = False

    @classmethod
    def from_json(cls, data):
        ground_truth = data.get("ground_truth")
        if ground_truth is not None and (not isinstance(ground_truth, str) or not ground_truth):
            raise ValueError("ground_truth must be text when supplied")
        single_frame = data.get("single_frame", False)
        if not isinstance(single_frame, bool):
            raise ValueError("single_frame must be true or false")
        return cls(
            _text(data, "source_path"), _text(data, "tracker_id"),
            None if data.get("roi") is None else Roi.from_json(data.get("roi")),
            _text(data, "client_host"), _port(data, "metadata_port"),
            _text(data, "playback_fps"), ground_truth, single_frame,
        )

    def json(self):
        return asdict(self)
