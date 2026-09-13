#!/usr/bin/env python3
"""Radxa FastAPI service for one streamed RKNN benchmark at a time."""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import gi
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from loguru import logger

from .protocol import Roi, RunCommand

gi.require_version("Gst", "1.0")
from gi.repository import Gst


FPS_CHOICES = ("Auto", "1", "5", "10", "20", "30")
IMAGE_SEQUENCE = re.compile(r"^(.*?)(\d+)(\.(?:jpg|jpeg|png))$", re.IGNORECASE)
VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}
logger.remove()
logger.add(sys.stderr, colorize=True, format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> | <magenta>model={extra[model]}</magenta> | {message}\n{exception}")


def gst_value(value):
    if isinstance(value, bool):
        value = str(value).lower()
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


@dataclass(frozen=True)
class Profile:
    id: str
    label: str
    element: str
    input_format: str
    requires_roi: bool
    properties: dict

    def gst_args(self, roi):
        args = [self.element]
        if self.requires_roi:
            args.append(f"roi={roi.text()}")
        return args + [f"{name}={gst_value(value)}" for name, value in self.properties.items()]


@dataclass
class Run:
    id: str
    command: RunCommand
    directory: Path
    status: str = "running"
    started_at: float = field(default_factory=time.monotonic)
    ended_at: float | None = None
    process: subprocess.Popen | None = None
    error: str | None = None
    client_metrics: dict | None = None

    def public(self):
        pipeline_file = self.directory / "pipeline.txt"
        return {
            "id": self.id, "status": self.status, "error": self.error,
            "artifacts": sorted(path.name for path in self.directory.iterdir()) if self.directory.exists() else [],
            "client_metrics": self.client_metrics,
            "pipeline": pipeline_file.read_text().strip() if pipeline_file.is_file() else None,
        }


class Service:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        self.config = yaml.safe_load(self.config_path.read_text()) or {}
        self.runtime_root = Path(self.config["runtime_root"]).resolve()
        self.dataset_root = (self.runtime_root / "datasets").resolve()
        self.results_root = Path(self.config["results_root"]).resolve()
        self.results_root.mkdir(parents=True, exist_ok=True)
        self.trackers = self._profiles()
        self.runs, self.active, self.lock = {}, None, threading.Lock()

    def _profiles(self):
        items = self.config.get("profiles")
        if not isinstance(items, list) or not items:
            raise ValueError("profiles must be a non-empty list")
        profiles = {}
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("each profile must be a mapping")
            required = ("id", "label", "element", "input_format")
            if any(not isinstance(item.get(name), str) or not item[name] for name in required):
                raise ValueError("each profile needs non-empty id, label, element, and input_format")
            if not isinstance(item.get("requires_roi"), bool):
                raise ValueError(f"profile {item['id']} requires_roi must be true or false")
            properties = item.get("properties")
            if not isinstance(properties, dict) or not all(isinstance(name, str) and name and isinstance(value, (str, int, float, bool)) for name, value in properties.items()):
                raise ValueError(f"profile {item['id']} properties must map names to scalar values")
            if item["id"] in profiles:
                raise ValueError(f"duplicate profile id: {item['id']}")
            profiles[item["id"]] = Profile(item["id"], item["label"], item["element"], item["input_format"], item["requires_roi"], properties)
        return profiles

    def inside_dataset_root(self, value, label="path"):
        path = Path(value).resolve()
        try:
            path.relative_to(self.dataset_root)
        except ValueError as error:
            raise ValueError(f"{label} must be under {self.dataset_root}") from error
        return path

    def dataset_catalog(self):
        output = {}
        for name, item in (self.config.get("datasets") or {}).items():
            source = self.inside_dataset_root(item["source"], f"dataset {name} source")
            output[name] = {"source_path": str(source), "ground_truth": item.get("ground_truth"),
                            "initial_roi": item.get("initial_roi"), "playback_fps": str(item.get("fps", "Auto"))}
        return output

    def sequence(self, directory):
        groups = {}
        for path in directory.iterdir():
            match = IMAGE_SEQUENCE.match(path.name) if path.is_file() else None
            if match:
                prefix, number, suffix = match.groups()
                groups.setdefault((prefix, len(number), suffix.lower()), []).append(int(number))
        if len(groups) != 1:
            raise ValueError("folder must contain one numbered PNG/JPEG sequence")
        (prefix, width, suffix), indexes = next(iter(groups.items()))
        indexes.sort()
        if indexes != list(range(indexes[0], indexes[-1] + 1)):
            raise ValueError("image sequence indexes must be continuous")
        mime, decoder = ("image/png", "pngdec") if suffix == ".png" else ("image/jpeg", "jpegdec")
        return str(directory / f"{prefix}%0{width}d{suffix}"), indexes[0], indexes[-1], mime, decoder

    def source_args(self, source_path, playback_fps):
        if playback_fps not in FPS_CHOICES:
            raise ValueError(f"playback_fps must be one of {', '.join(FPS_CHOICES)}")
        source = self.inside_dataset_root(source_path, "source_path")
        if source.is_file():
            if source.suffix.lower() not in VIDEO_SUFFIXES:
                raise ValueError("source file must be a supported video")
            args = ["filesrc", f"location={gst_value(source)}", "!", "decodebin", "!", "videoconvert"]
            if playback_fps != "Auto":
                args += ["!", "videorate", "!", f"video/x-raw,framerate={playback_fps}/1"]
            return source, args
        if source.is_dir():
            pattern, start, stop, mime, decoder = self.sequence(source)
            fps = "20" if playback_fps == "Auto" else playback_fps
            return source, ["multifilesrc", f"location={gst_value(pattern)}", f"start-index={start}", f"stop-index={stop}",
                            f"caps={mime},framerate={fps}/1", "!", decoder, "!", "videoconvert"]
        raise ValueError("source_path does not exist")

    def start(self, command):
        _source, source_args = self.source_args(command.source_path, command.playback_fps)
        if command.tracker_id not in self.trackers:
            raise ValueError("unknown tracker_id")
        profile = self.trackers[command.tracker_id]
        log = logger.bind(model=str(profile.properties.get("model", profile.label)))
        if profile.requires_roi and command.roi is None:
            raise ValueError("roi is required for this profile")
        if command.ground_truth and not profile.requires_roi:
            raise ValueError("ground_truth is only supported for ROI tracker profiles")
        if command.ground_truth:
            ground_truth = self.inside_dataset_root(command.ground_truth, "ground_truth")
            if not ground_truth.is_file():
                raise ValueError("ground_truth does not exist")
        with self.lock:
            if self.active:
                raise RuntimeError("another benchmark run is active")
            run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
            directory = self.results_root / run_id
            directory.mkdir()
            run = Run(run_id, command, directory)
            self.runs[run_id] = run
            self.active = run_id
        prediction, log_file, pipeline_file = directory / "predictions.csv", directory / "pipeline.log", directory / "pipeline.txt"
        args = source_args + ["!", "videoscale", "!", "videoconvert", "!", f"video/x-raw,format={profile.input_format},width=640,height=360"]
        if command.single_frame:
            args += ["!", "identity", "eos-after=1"]
        args += ["!", *profile.gst_args(command.roi), "!", "roi2csv", f"location={prediction}", "!", "roi2udp",
                 f"host={command.client_host}", f"port={command.metadata_port}", "!", "fakesink", "sync=true"]
        process_args = ["gst-launch-1.0", "-q", *args]
        pipeline_file.write_text(" ".join(process_args) + "\n")
        (directory / "request.json").write_text(json.dumps(command.json(), indent=2) + "\n")
        environment = os.environ | {"GST_PLUGIN_PATH": str(self.runtime_root / "plugins"), "GST_REGISTRY_1_0": "/tmp/gst-rknn-registry.bin"}
        with log_file.open("wb") as output:
            run.process = subprocess.Popen(process_args, stdout=output, stderr=subprocess.STDOUT, env=environment)
        log.info("Started run {} single_frame={} source={}", run.id, command.single_frame, command.source_path)
        threading.Thread(target=self._finish, args=(run,), daemon=True).start()
        return run

    def _finish(self, run):
        code = run.process.wait()
        run.ended_at = time.monotonic()
        if run.status == "stopped":
            pass
        elif code:
            run.status, run.error = "failed", f"gst-launch exited with {code}"
        else:
            run.status = "finished"
        log = logger.bind(model=str(self.trackers[run.command.tracker_id].properties.get("model", self.trackers[run.command.tracker_id].label)))
        log.info("Run {} {} exit={} log={}", run.id, run.status, code, run.directory / "pipeline.log")
        rows = []
        prediction = run.directory / "predictions.csv"
        if prediction.exists():
            with prediction.open(newline="") as source:
                rows = list(csv.DictReader(source))
        confidence = [float(row["confidence"]) for row in rows if row.get("confidence")]
        elapsed = max((run.ended_at or time.monotonic()) - run.started_at, 1e-9)
        profile = self.trackers[run.command.tracker_id]
        detector = not profile.requires_roi
        summary = {"run_id": run.id, "mode": "detector" if detector else "tracker", "status": run.status, "error": run.error, "server_elapsed_s": round(elapsed, 6),
                   "server_fps": None if detector else round(len(rows) / elapsed, 2), "tracker_rows": len(rows),
                   "mean_confidence": round(sum(confidence) / len(confidence), 4) if confidence else None,
                   "low_confidence_threshold": 0.5,
                   "low_confidence_frames": sum(value < 0.5 for value in confidence),
                   "initialized_frames": None if detector else sum(row.get("initialized") == "1" for row in rows),
                   "client_metrics": run.client_metrics}
        if detector:
            summary.update({"detections": len(rows), "detections_per_second": round(len(rows) / elapsed, 2)})
        if not detector and run.command.ground_truth and prediction.exists():
            report = run.directory / "iou.txt"
            comparator = Path(__file__).with_name("compare_nanotracker_csv.py")
            result = subprocess.run([sys.executable, str(comparator), run.command.ground_truth, str(prediction)], text=True, capture_output=True)
            report.write_text(result.stdout + result.stderr)
            for key, label in (("mean_iou", "mean IoU:"), ("minimum_iou", "minimum IoU:")):
                for line in result.stdout.splitlines():
                    if label in line:
                        summary[key] = float(line.split()[-1])
        (run.directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        with self.lock:
            if self.active == run.id:
                self.active = None

    def stop(self, run_id):
        run = self.runs.get(run_id)
        if not run:
            raise KeyError(run_id)
        if run.status == "running":
            run.status = "stopped"
            run.process.terminate()
        return run


def build_app(config_path):
    Gst.init(None)
    service = Service(config_path)
    app = FastAPI(title="NanoTracker benchmark")

    def command(payload, kind):
        try:
            return kind.from_json(payload)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/v1/health")
    def health():
        return {"ok": True, "active_run": service.active}

    @app.get("/v1/catalog")
    def catalog():
        return {"datasets": service.dataset_catalog(), "mirror_root": str(service.dataset_root), "trackers": [asdict(item) for item in service.trackers.values()], "playback_fps": FPS_CHOICES}

    @app.get("/v1/browse")
    def browse(path: str | None = None):
        try:
            directory = service.inside_dataset_root(path or service.dataset_root)
            if not directory.is_dir():
                raise ValueError("path is not a directory")
            entries = [{"name": item.name, "path": str(item), "directory": item.is_dir()} for item in sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))]
            return {"path": str(directory), "entries": entries}
        except (OSError, ValueError) as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/v1/runs", status_code=201)
    def start(payload: dict):
        try:
            return service.start(command(payload, RunCommand)).public()
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/v1/runs/{run_id}")
    def status(run_id: str):
        run = service.runs.get(run_id)
        if not run:
            raise HTTPException(404, "run not found")
        return run.public()

    @app.post("/v1/runs/{run_id}/stop")
    def stop(run_id: str):
        try:
            return service.stop(run_id).public()
        except KeyError as error:
            raise HTTPException(404, "run not found") from error

    @app.post("/v1/runs/{run_id}/client-metrics")
    def client_metrics(run_id: str, payload: dict):
        run = service.runs.get(run_id)
        if not run:
            raise HTTPException(404, "run not found")
        frames, elapsed = payload.get("frames"), payload.get("elapsed_s")
        if not isinstance(frames, int) or frames < 0 or not isinstance(elapsed, (int, float)) or elapsed <= 0:
            raise HTTPException(422, "frames and positive elapsed_s are required")
        losses = payload.get("metadata_losses", 0)
        if not isinstance(losses, int) or losses < 0:
            raise HTTPException(422, "metadata_losses must be a non-negative integer")
        run.client_metrics = {"frames": frames, "elapsed_s": round(elapsed, 6), "fps": round(frames / elapsed, 2), "metadata_losses": losses}
        summary = run.directory / "summary.json"
        if summary.exists():
            data = json.loads(summary.read_text())
            data["client_metrics"] = run.client_metrics
            summary.write_text(json.dumps(data, indent=2) + "\n")
        return run.public()

    @app.get("/v1/runs/{run_id}/artifacts/{name}")
    def artifact(run_id: str, name: str):
        run = service.runs.get(run_id)
        if not run or Path(name).name != name or not (run.directory / name).is_file():
            raise HTTPException(404, "artifact not found")
        return FileResponse(run.directory / name)

    return app, service


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("server.yaml"))
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    app, service = build_app(args.config)
    if args.check_config:
        service.dataset_catalog()
        print(f"Valid config: {len(service.trackers)} trackers, {len(service.dataset_catalog())} datasets")
        return
    import uvicorn
    uvicorn.run(app, host=service.config.get("listen_host", "0.0.0.0"), port=int(service.config.get("listen_port", 8000)))


if __name__ == "__main__":
    main()
