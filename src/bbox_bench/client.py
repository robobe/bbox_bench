#!/usr/bin/env python3
"""Desktop client for metadata-paced RKNN benchmark playback."""
import argparse, json, re, socket, sys, threading, time, tkinter as tk, urllib.error, urllib.parse, urllib.request
from datetime import datetime
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk

import cv2, gi, numpy as np, yaml
from loguru import logger
from .protocol import metadata_packet
from .report_analysis import analyze_report
from .run_report import write_run_report

gi.require_version("Gst", "1.0"); gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstVideo

FPS_CHOICES = ("Auto", "1", "5", "10", "20", "30")
IMAGE_SEQUENCE = re.compile(r"^(.*?)(\d+)(\.(?:jpg|jpeg|png))$", re.I)
VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}
logger.remove()
logger.add(sys.stderr, colorize=True, format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> | <magenta>model={extra[model]}</magenta> | {message}\n{exception}")


def gst_value(value): return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def relative_source(source, mirror_root):
    return str(Path(source).resolve().relative_to(Path(mirror_root).resolve()))


class Api:
    def __init__(self, base): self.base = base.rstrip("/")
    def request(self, method, path, payload=None, timeout=10):
        data = json.dumps(payload).encode() if payload is not None else None; request = urllib.request.Request(self.base + path, data=data, method=method)
        if data: request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response: return json.loads(response.read())
        except urllib.error.HTTPError as error: raise RuntimeError(error.read().decode() or error.reason) from error
        except urllib.error.URLError as error: raise RuntimeError(str(error.reason)) from error


class LocalViewer:
    def __init__(self, source, playback_fps, metadata_port=None, single_frame=False, model_name="-"):
        self.source, self.playback_fps, self.metadata_port = Path(source), playback_fps, metadata_port
        self.single_frame = single_frame
        self.pipeline = self.sink = self.pending = None; self.frames = self.expected_frame = self.losses = 0
        self.first_frame = self.last_frame = self.pending_at = None; self.eos = False
        self.metadata, self.lock, self.stop_event, self.thread = {}, threading.Lock(), threading.Event(), None
        self.frame_rows = []
        self.receiver_ready, self.receiver_error = threading.Event(), None
        self.log = logger.bind(model=model_name)

    @staticmethod
    def sequence(directory):
        groups = {}
        for path in directory.iterdir():
            match = IMAGE_SEQUENCE.match(path.name) if path.is_file() else None
            if match:
                prefix, number, suffix = match.groups(); groups.setdefault((prefix, len(number), suffix.lower()), []).append(int(number))
        if len(groups) != 1: raise ValueError("folder must contain one numbered PNG/JPEG sequence")
        (prefix, width, suffix), indexes = next(iter(groups.items())); indexes.sort()
        if indexes != list(range(indexes[0], indexes[-1] + 1)): raise ValueError("image sequence indexes must be continuous")
        mime, decoder = ("image/png", "pngdec") if suffix == ".png" else ("image/jpeg", "jpegdec")
        return str(directory / f"{prefix}%0{width}d{suffix}"), indexes[0], indexes[-1], mime, decoder

    @classmethod
    def source_args(cls, source, playback_fps):
        if playback_fps not in FPS_CHOICES: raise ValueError("invalid playback FPS")
        source = Path(source).resolve()
        if source.is_file():
            if source.suffix.lower() not in VIDEO_SUFFIXES: raise ValueError("source file must be a supported video")
            args = ["filesrc", f"location={gst_value(source)}", "!", "decodebin", "!", "videoconvert"]
            return args if playback_fps == "Auto" else args + ["!", "videorate", "!", f"video/x-raw,framerate={playback_fps}/1"]
        if source.is_dir():
            pattern, start, stop, mime, decoder = cls.sequence(source); fps = "20" if playback_fps == "Auto" else playback_fps
            return ["multifilesrc", f"location={gst_value(pattern)}", f"start-index={start}", f"stop-index={stop}", f"caps={mime},framerate={fps}/1", "!", decoder, "!", "videoconvert"]
        raise ValueError("local mirrored source does not exist")

    @staticmethod
    def image(sample):
        caps = sample.get_caps().get_structure(0); ok_w, width = caps.get_int("width"); ok_h, height = caps.get_int("height")
        if not ok_w or not ok_h: raise ValueError("decoded frame has no dimensions")
        buffer = sample.get_buffer(); meta = GstVideo.buffer_get_video_meta(buffer); ok, mapped = buffer.map(Gst.MapFlags.READ)
        if not ok: raise ValueError("could not read decoded frame")
        try:
            stride = meta.stride[0] if meta else width * 3
            return np.ndarray((height, width, 3), dtype=np.uint8, buffer=mapped.data, strides=(stride, 3, 1)).copy()
        finally: buffer.unmap(mapped)

    @classmethod
    def preview(cls, source, playback_fps):
        pipeline = Gst.parse_launch(" ".join(cls.source_args(source, playback_fps) + ["!", "videoscale", "!", "videoconvert", "!", "video/x-raw,format=BGR,width=640,height=360", "!", "appsink", "name=sink", "sync=false"]))
        sink = pipeline.get_by_name("sink"); pipeline.set_state(Gst.State.PAUSED); state, _, _ = pipeline.get_state(10 * Gst.SECOND)
        if state == Gst.StateChangeReturn.FAILURE: pipeline.set_state(Gst.State.NULL); raise ValueError("could not decode local preview")
        sample = sink.emit("pull-preroll"); pipeline.set_state(Gst.State.NULL)
        if sample is None: raise ValueError("local source has no decodable frame")
        return cls.image(sample)

    @classmethod
    def dimensions(cls, source, playback_fps):
        pipeline = Gst.parse_launch(" ".join(cls.source_args(source, playback_fps) + ["!", "appsink", "name=sink", "sync=false"]))
        sink = pipeline.get_by_name("sink")
        try:
            pipeline.set_state(Gst.State.PAUSED); state, _, _ = pipeline.get_state(10 * Gst.SECOND)
            if state == Gst.StateChangeReturn.FAILURE: raise ValueError("could not decode local source dimensions")
            sample = sink.emit("pull-preroll")
            if sample is None: raise ValueError("local source has no decodable frame")
            caps = sample.get_caps().get_structure(0); ok_w, width = caps.get_int("width"); ok_h, height = caps.get_int("height")
            if not ok_w or not ok_h: raise ValueError("decoded source has no dimensions")
            return width, height
        finally: pipeline.set_state(Gst.State.NULL)

    def start(self):
        self.source_width, self.source_height = self.dimensions(self.source, self.playback_fps)
        self.start_metadata_receiver()
        args = self.source_args(self.source, self.playback_fps) + ["!", "videoscale", "!", "videoconvert", "!", "video/x-raw,format=BGR,width=640,height=360"]
        if self.single_frame: args += ["!", "identity", "eos-after=1"]
        args += ["!", "appsink", "name=sink", "sync=false", "max-buffers=1", "drop=false"]
        self.pipeline = Gst.parse_launch(" ".join(args)); self.sink = self.pipeline.get_by_name("sink"); self.pipeline.set_state(Gst.State.PLAYING)

    def start_metadata_receiver(self):
        if self.metadata_port is None: return
        self.thread = threading.Thread(target=self.receive_metadata, daemon=True); self.thread.start()
        if not self.receiver_ready.wait(1): raise ValueError("metadata receiver did not start")
        if self.receiver_error: raise ValueError(f"metadata receiver failed: {self.receiver_error}")
        self.log.debug("UDP metadata receiver ready on port {}", self.metadata_port)

    def receive_metadata(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind(("0.0.0.0", self.metadata_port)); self.metadata_port = sock.getsockname()[1]; sock.settimeout(.25); self.receiver_ready.set()
            while not self.stop_event.is_set():
                try:
                    frame_id, pts_ns, kind, x, y, width, height, initialized, confidence, class_id = metadata_packet(sock.recvfrom(2048)[0])
                    with self.lock:
                        item = self.metadata.setdefault(frame_id, {"pts_ns": pts_ns, "boxes": [], "complete_at": None})
                        if kind == "frame": item["complete_at"] = time.monotonic()
                        else: item["boxes"].append((kind, x, y, width, height, initialized, confidence, class_id))
                except (socket.timeout, UnicodeDecodeError, ValueError): pass
        except OSError as error:
            self.receiver_error = str(error); self.receiver_ready.set(); self.log.opt(exception=error).error("UDP metadata receiver failed")
        finally: sock.close()

    def finished(self):
        if self.pipeline and self.pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS): self.eos = True
        return self.eos and self.pending is None

    def poll(self):
        if not self.sink: return False
        if self.pending is None:
            self.pending = self.sink.emit("try-pull-sample", 0)
            if self.pending is not None: self.pending_at = time.monotonic()
        if self.pending is None: return False
        boxes, loss = [], False
        if self.metadata_port:
            with self.lock:
                item = self.metadata.get(self.expected_frame); ready = item and item["complete_at"] and time.monotonic() - item["complete_at"] >= .01
                if ready: item = self.metadata.pop(self.expected_frame); boxes = item["boxes"]
            if not ready and time.monotonic() - self.pending_at < .5: return False
            if not ready or item["pts_ns"] != self.pending.get_buffer().pts: loss = True; boxes = []; self.losses += 1
            row = {"frame_id": self.expected_frame, "pts_ns": self.pending.get_buffer().pts, "metadata_lost": loss,
                   "kind": "", "x": "", "y": "", "width": "", "height": "", "initialized": "", "confidence": "", "class_id": ""}
            if boxes:
                kind, x, y, width, height, initialized, confidence, class_id = boxes[0]
                row.update(kind=kind, x=x, y=y, width=width, height=height, initialized=initialized, confidence=confidence, class_id=class_id if class_id is not None else "")
            self.frame_rows.append(row)
            self.expected_frame += 1
        self.render(self.image(self.pending), boxes, loss); self.pending = None; return True

    def render(self, frame, boxes, loss):
        now = time.monotonic(); self.first_frame = self.first_frame or now; self.last_frame, self.frames = now, self.frames + 1
        for kind, x, y, width, height, initialized, confidence, class_id in boxes:
            name = f"class {class_id}" if kind == "yolo8" and class_id is not None else kind
            color = (0, 255, 255) if initialized == "1" else (0, 0, 255) if confidence and float(confidence) < .5 else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2); cv2.putText(frame, f"{name} {confidence or 'init'}", (x, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
        elapsed = max(now - self.first_frame, 1e-9); text = f"client FPS: {self.frames / elapsed:.1f}" + ("  metadata lost" if loss else "")
        cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255) if loss else (0, 255, 0), 2); cv2.imshow("RKNN benchmark", frame); cv2.waitKey(1)

    def metrics(self): return {"frames": self.frames, "elapsed_s": max((self.last_frame or 0) - (self.first_frame or 0), 1e-9), "metadata_losses": self.losses}
    def close(self):
        self.stop_event.set()
        if self.pipeline: self.pipeline.set_state(Gst.State.NULL)
        self.pipeline = self.sink = self.pending = None; cv2.destroyAllWindows()


class ReportPlayer:
    def __init__(self, source, playback_fps): self.source, self.playback_fps, self.pipeline, self.sink, self.frame_id, self.eos = Path(source), playback_fps, None, None, 0, False
    def start(self):
        args = LocalViewer.source_args(self.source, self.playback_fps) + ["!", "videoscale", "!", "videoconvert", "!", "video/x-raw,format=BGR,width=640,height=360", "!", "appsink", "name=sink", "sync=true", "max-buffers=1", "drop=false"]
        self.pipeline = Gst.parse_launch(" ".join(args)); self.sink = self.pipeline.get_by_name("sink"); self.pipeline.set_state(Gst.State.PLAYING)
    def poll(self):
        sample = self.sink.emit("try-pull-sample", 0) if self.sink else None
        if sample is None: return None
        frame_id, self.frame_id = self.frame_id, self.frame_id + 1
        return frame_id, LocalViewer.image(sample)
    def finished(self):
        if self.pipeline and self.pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS): self.eos = True
        return self.eos
    def close(self):
        if self.pipeline: self.pipeline.set_state(Gst.State.NULL)
        self.pipeline = self.sink = None
        try: cv2.destroyWindow("Tracker vs ground truth")
        except cv2.error: pass


class Client:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve(); config = yaml.safe_load(self.config_path.read_text()) or {}; self.api, self.client_host = Api(config["server_url"]), config["client_host"]; self.metadata_port = int(config["metadata_port"]); self.ground_truth_format = config.get("ground_truth_format") or {}; self.report_location = config.get("report_location", "reports")
        self.state_path = Path.home() / ".config" / "gst-rknn" / "nanotracker-benchmark-last-run.yaml"; Gst.init(None)
        self.catalog = {}; self.viewer = self.run_id = self.roi = self.run_context = None; self.source_only = self.preview_open = False; self.next_server_check = self.next_run_check = 0.0
        self.root = tk.Tk(); self.root.title("RKNN benchmark"); self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.source, self.source_display, self.local_root, self.local_source, self.ground_truth, self.ground_truth_display = tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(); self.dataset, self.tracker, self.rate = tk.StringVar(value="Ad-hoc source"), tk.StringVar(), tk.StringVar(value="Auto")
        saved_root = self.state_document().get("client_dataset_root")
        if isinstance(saved_root, str): self.local_root.set(saved_root)
        self.saved_run, self.preset_name = tk.StringVar(), tk.StringVar(); self.saved_items = {}; self.server_status, self.status = tk.StringVar(value="Server: connecting"), tk.StringVar(value="Connecting to benchmark server")
        self.build(); self.load_catalog(); self.root.after(30, self.tick)

    def build(self):
        panel = ttk.Frame(self.root, padding=12); panel.grid(sticky="nsew"); ttk.Label(panel, textvariable=self.server_status).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        ttk.Label(panel, text="Recent / preset").grid(row=1, column=0, sticky="nw", pady=3); saved_frame = ttk.Frame(panel); saved_frame.grid(row=1, column=1, sticky="nsew", pady=3); self.saved_run_list = tk.Listbox(saved_frame, height=4, exportselection=False); saved_scroll = ttk.Scrollbar(saved_frame, orient="vertical", command=self.saved_run_list.yview); self.saved_run_list.configure(yscrollcommand=saved_scroll.set); self.saved_run_list.pack(side="left", fill="both", expand=True); saved_scroll.pack(side="right", fill="y"); self.saved_run_list.bind("<<ListboxSelect>>", lambda _: self.load_saved_run())
        actions = ttk.Frame(panel); actions.grid(row=1, column=2, padx=(6, 0)); ttk.Button(actions, text="Save", command=self.save_preset).pack(fill="x"); ttk.Button(actions, text="Rename", command=self.rename_preset).pack(fill="x", pady=(3, 0)); ttk.Button(actions, text="Clear all", command=self.clear_presets).pack(fill="x", pady=(3, 0))
        labels = ("Preset name", "Client dataset root", "Dataset", "Server source", "Relative folder", "Ground truth", "Profile", "Playback FPS")
        for row, label in enumerate(labels, start=2): ttk.Label(panel, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(panel, textvariable=self.preset_name, width=52).grid(row=2, column=1, sticky="ew", pady=3); ttk.Entry(panel, textvariable=self.local_root, width=52).grid(row=3, column=1, sticky="ew", pady=3); ttk.Button(panel, text="Choose", command=self.choose_local_root).grid(row=3, column=2, padx=(6, 0))
        self.dataset_box = ttk.Combobox(panel, textvariable=self.dataset, state="readonly", width=52); self.dataset_box.grid(row=4, column=1, sticky="ew", pady=3); self.dataset_box.bind("<<ComboboxSelected>>", lambda _: self.choose_dataset())
        ttk.Entry(panel, textvariable=self.source_display, width=52, state="readonly").grid(row=5, column=1, sticky="ew", pady=3); ttk.Button(panel, text="Browse", command=self.browse_server).grid(row=5, column=2, padx=(6, 0)); ttk.Entry(panel, textvariable=self.local_source, width=52, state="readonly").grid(row=6, column=1, sticky="ew", pady=3)
        self.ground_truth_box = ttk.Entry(panel, textvariable=self.ground_truth_display, width=52, state="readonly"); self.ground_truth_box.grid(row=7, column=1, sticky="ew", pady=3); self.ground_truth_browse = ttk.Button(panel, text="Browse", command=self.choose_ground_truth); self.ground_truth_browse.grid(row=7, column=2, padx=(6, 0)); self.tracker_box = ttk.Combobox(panel, textvariable=self.tracker, state="readonly", width=52); self.tracker_box.grid(row=8, column=1, sticky="ew", pady=3); self.tracker_box.bind("<<ComboboxSelected>>", lambda _: self.profile_changed())
        ttk.Combobox(panel, textvariable=self.rate, values=FPS_CHOICES, state="readonly", width=52).grid(row=9, column=1, sticky="ew", pady=3); self.preview_button = ttk.Button(panel, text="Preview / select ROI", command=self.preview); self.preview_button.grid(row=10, column=0, sticky="ew", pady=(8, 3)); self.inference_preview_button = ttk.Button(panel, text="Preview inference", command=self.preview_inference); self.inference_preview_button.grid(row=10, column=1, sticky="ew", padx=(6, 0), pady=(8, 3)); ttk.Button(panel, text="Start benchmark", command=self.start).grid(row=10, column=2, sticky="ew", padx=(6, 0), pady=(8, 3)); ttk.Button(panel, text="Play source", command=lambda: self.start(True)).grid(row=11, column=0, sticky="ew", pady=3); ttk.Button(panel, text="Show pipeline", command=self.show_pipeline).grid(row=11, column=1, sticky="ew", pady=3); ttk.Button(panel, text="Stop", command=self.stop).grid(row=11, column=2, sticky="ew", padx=(6, 0), pady=3); ttk.Button(panel, text="Analyze report", command=self.open_report_analysis).grid(row=12, column=1, sticky="ew", pady=3); ttk.Label(panel, textvariable=self.status, wraplength=620, justify="left").grid(row=13, column=0, columnspan=3, sticky="ew", pady=(8, 0)); panel.columnconfigure(1, weight=1)

    def load_catalog(self):
        try:
            self.catalog = self.api.request("GET", "/v1/catalog"); self.dataset_box.configure(values=["Ad-hoc source", *self.catalog["datasets"]]); self.tracker_labels = {item["label"]: item["id"] for item in self.catalog["trackers"]}; self.tracker_profiles = {item["id"]: item for item in self.catalog["trackers"]}; self.tracker_box.configure(values=list(self.tracker_labels)); self.tracker.set(next(iter(self.tracker_labels))); self.load_last_selection(); self.profile_changed(); self.refresh_saved_runs(); self.server_status.set("Server: connected"); self.status.set("Choose a mirrored dataset root, then select a server source")
        except RuntimeError as error: self.server_status.set(f"Server unavailable: {error}")

    def selected_profile(self): return self.tracker_profiles.get(self.tracker_labels.get(self.tracker.get(), ""), {})
    def model_name(self): return str(self.selected_profile().get("properties", {}).get("model", self.selected_profile().get("label", "-")))
    def profile_changed(self):
        required = self.selected_profile().get("requires_roi", True); self.preview_button.configure(text="Preview / select ROI" if required else "Preview"); self.inference_preview_button.configure(state="disabled" if required else "normal"); self.ground_truth_box.configure(state="normal" if required else "disabled"); self.ground_truth_browse.configure(state="normal" if required else "disabled")
        if not required: self.roi = None; self.set_ground_truth("")
    def local_path(self):
        root, source, mirror = Path(self.local_root.get()).expanduser().resolve(), Path(self.source.get()).resolve(), Path(self.catalog["mirror_root"]).resolve()
        try: relative = Path(relative_source(source, mirror)); local = root / relative
        except ValueError as error: raise ValueError("server source is outside the mirrored datasets root") from error
        if not root.is_dir() or not local.exists(): raise ValueError("mirrored local source does not exist")
        self.source_display.set(relative.name); self.local_source.set("" if relative.parent == Path(".") else str(relative.parent)); return local
    def set_source(self, source):
        self.source.set(str(Path(source).resolve()))
        try:
            relative = Path(relative_source(source, self.catalog["mirror_root"])); self.source_display.set(relative.name); self.local_source.set("" if relative.parent == Path(".") else str(relative.parent))
        except ValueError: self.source_display.set("")
        self.update_local_source()
    def choose_local_root(self):
        root = filedialog.askdirectory(title="Select mirrored client dataset root", initialdir=self.local_root.get() or str(Path.home()))
        if root:
            self.local_root.set(root); state = self.state_document(); state["client_dataset_root"] = root; self.write_state(state); self.update_local_source()
    def choose_ground_truth(self):
        path = filedialog.askopenfilename(title="Select local ground truth", initialdir=self.ground_truth.get() or self.local_root.get() or str(Path.home()), filetypes=[("Ground truth", "*.csv *.txt"), ("All files", "*")])
        if path: self.set_ground_truth(path)
    def set_ground_truth(self, path):
        if not path: self.ground_truth.set(""); self.ground_truth_display.set(""); return
        path = Path(path).expanduser().resolve(); self.ground_truth.set(str(path))
        try: self.ground_truth_display.set(str(path.relative_to(Path(self.local_root.get()).expanduser().resolve())))
        except ValueError: self.ground_truth_display.set(str(path))
    def update_local_source(self):
        try: self.local_path()
        except ValueError as error: self.local_source.set(""); self.status.set(str(error))
    def state_document(self):
        try: state = yaml.safe_load(self.state_path.read_text()) or {}
        except (OSError, yaml.YAMLError): state = {}
        if not isinstance(state, dict): state = {}
        state.setdefault("recent_runs", []); state.setdefault("presets", {}); return state
    def write_state(self, state): self.state_path.parent.mkdir(parents=True, exist_ok=True); self.state_path.write_text(yaml.safe_dump(state, sort_keys=False))
    def selection(self): return {"dataset": self.dataset.get(), "source_path": self.source.get(), "ground_truth": self.ground_truth_display.get(), "tracker_id": self.tracker_labels.get(self.tracker.get(), ""), "playback_fps": self.rate.get(), "roi": self.roi, "roi_size": [640, 360], "client_dataset_root": self.local_root.get(), "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    def apply_selection(self, state):
        for variable, key in ((self.source, "source_path"), (self.rate, "playback_fps"), (self.local_root, "client_dataset_root")):
            if isinstance(state.get(key), str): variable.set(state[key])
        ground_truth = state.get("ground_truth")
        if isinstance(ground_truth, str): self.set_ground_truth(Path(self.local_root.get()).expanduser() / ground_truth if ground_truth and not Path(ground_truth).is_absolute() else ground_truth)
        for label, profile_id in self.tracker_labels.items():
            if profile_id == state.get("tracker_id"): self.tracker.set(label); break
        if state.get("dataset") in self.dataset_box["values"]: self.dataset.set(state["dataset"])
        roi = state.get("roi"); self.roi = roi if state.get("roi_size") == [640, 360] and isinstance(roi, dict) else None; self.profile_changed()
        if self.source.get(): self.set_source(self.source.get())
    def load_last_selection(self):
        state = self.state_document()
        if isinstance(state.get("last_run"), dict): self.apply_selection(state["last_run"])
    def refresh_saved_runs(self, state=None):
        state = state or self.state_document(); self.saved_items = {}
        for index, run in enumerate(state["recent_runs"]):
            if isinstance(run, dict): self.saved_items[f"Recent: {run.get('saved_at', '?')} | {Path(run.get('source_path', '?')).name} #{index + 1}"] = run
        for name, run in state["presets"].items():
            if isinstance(name, str) and isinstance(run, dict): self.saved_items[f"Preset: {name}"] = run
        self.saved_run_list.delete(0, tk.END)
        for label in self.saved_items: self.saved_run_list.insert(tk.END, label)
    def load_saved_run(self):
        choice = self.saved_run_list.curselection()
        if choice:
            label = self.saved_run_list.get(choice[0]); self.saved_run.set(label); self.apply_selection(self.saved_items[label])
            if label.startswith("Preset: "): self.preset_name.set(label[8:])
    def save_preset(self):
        name = self.preset_name.get().strip()
        if not name: self.status.set("Enter a preset name first"); return
        state = self.state_document(); state["presets"][name] = self.selection(); self.write_state(state); self.refresh_saved_runs(state)
    def rename_preset(self):
        selected, name = self.saved_run.get(), self.preset_name.get().strip()
        if not selected.startswith("Preset: ") or not name: self.status.set("Select a preset and enter its new name"); return
        state = self.state_document(); state["presets"][name] = state["presets"].pop(selected[8:]); self.write_state(state); self.refresh_saved_runs(state)
    def clear_presets(self):
        if messagebox.askyesno("Clear all", "Delete all saved presets and recent runs?", parent=self.root):
            state = self.state_document(); state["presets"] = {}; state["recent_runs"] = []; state.pop("last_run", None); self.write_state(state); self.saved_run.set(""); self.preset_name.set(""); self.refresh_saved_runs(state); self.status.set("Saved presets and recent runs cleared")
    def choose_dataset(self):
        item = self.catalog["datasets"].get(self.dataset.get())
        if item:
            self.set_source(item["source_path"])
            ground_truth = item.get("ground_truth")
            try: self.set_ground_truth(Path(self.local_root.get()).expanduser().resolve() / relative_source(ground_truth, self.catalog["mirror_root"])) if ground_truth and self.local_root.get() else self.set_ground_truth("")
            except ValueError: self.set_ground_truth("")
            self.rate.set(item.get("playback_fps", "Auto")); roi = item.get("initial_roi"); self.roi = dict(zip(("x", "y", "width", "height"), roi)) if roi else None; self.profile_changed()
    def browse_server(self, ground_truth=False):
        chooser = tk.Toplevel(self.root); chooser.title("Select ground truth" if ground_truth else "Browse Radxa mirrored datasets"); path, entries = tk.StringVar(value=self.catalog["mirror_root"]), tk.Listbox(chooser, width=80, height=20); ttk.Entry(chooser, textvariable=path, width=72).pack(padx=8, pady=5); entries.pack(fill="both", expand=True, padx=8, pady=5)
        def refresh(directory=None):
            try:
                data = self.api.request("GET", "/v1/browse?path=" + urllib.parse.quote(directory or path.get())); path.set(data["path"]); entries.delete(0, tk.END); entries.insert(tk.END, ".."); chooser.items = data["entries"]
                for item in chooser.items: entries.insert(tk.END, ("[D] " if item["directory"] else "[F] ") + item["name"])
            except RuntimeError as error: messagebox.showerror("Browse failed", str(error), parent=chooser)
        def select(_event=None):
            choice = entries.curselection()
            if not choice: return
            if choice[0] == 0: refresh(str(Path(path.get()).parent)); return
            item = chooser.items[choice[0] - 1]
            if item["directory"]: refresh(item["path"])
            elif ground_truth: self.ground_truth.set(item["path"]); chooser.destroy()
            else: self.set_source(item["path"]); self.dataset.set("Ad-hoc source"); self.set_ground_truth(""); self.roi = None; chooser.destroy()
        def use_folder():
            self.set_source(path.get()); self.dataset.set("Ad-hoc source"); self.set_ground_truth(""); self.roi = None; chooser.destroy()
        if not ground_truth: ttk.Button(chooser, text="Use this folder", command=use_folder).pack(side="left", padx=8, pady=5)
        entries.bind("<Double-Button-1>", select); refresh()
    def preview(self):
        try:
            frame = LocalViewer.preview(self.local_path(), self.rate.get())
            if not self.selected_profile().get("requires_roi", True):
                cv2.imshow("Preview", frame); cv2.waitKey(1); self.preview_open = True; self.status.set("Preview decoded locally; YOLO inference has not run"); return
            selected = cv2.selectROI("Select tracker ROI", frame, showCrosshair=True, fromCenter=False); cv2.destroyWindow("Select tracker ROI")
            if selected[2] > 0 and selected[3] > 0: self.roi = dict(zip(("x", "y", "width", "height"), map(int, selected))); self.status.set(f"ROI selected: {self.roi}")
        except ValueError as error: self.status.set(f"Preview failed: {error}")
    def preview_inference(self):
        if self.selected_profile().get("requires_roi", True): return
        try:
            if self.preview_open: cv2.destroyWindow("Preview"); self.preview_open = False
            local = self.local_path(); self.viewer = LocalViewer(local, self.rate.get(), self.metadata_port, single_frame=True, model_name=self.model_name()); self.viewer.start()
            payload = {"source_path": self.source.get(), "ground_truth": None, "tracker_id": self.tracker_labels.get(self.tracker.get(), ""), "roi": None, "client_host": self.client_host, "metadata_port": self.metadata_port, "playback_fps": self.rate.get(), "single_frame": True}
            logger.bind(model=self.model_name()).info("Requesting one-frame preview source={}", self.source.get())
            self.run_id = self.api.request("POST", "/v1/runs", payload)["id"]; self.status.set(f"Running one-frame inference {self.run_id}")
        except (RuntimeError, ValueError) as error:
            if self.viewer: self.viewer.close(); self.viewer = None
            logger.bind(model=self.model_name()).opt(exception=error).error("Preview inference failed source={}", self.source.get())
            self.status.set(f"Preview inference failed: {error}")
    def start(self, source_only=False):
        if not source_only and self.selected_profile().get("requires_roi", True) and not self.roi: self.status.set("Preview and select an ROI first"); return
        try:
            self.run_context = None
            if self.preview_open: cv2.destroyWindow("Preview"); self.preview_open = False
            local = self.local_path(); self.source_only = source_only; self.viewer = LocalViewer(local, self.rate.get(), None if source_only else self.metadata_port, model_name=self.model_name()); self.viewer.start()
            if source_only: self.status.set("Playing local mirrored source"); return
            payload = {"source_path": self.source.get(), "ground_truth": None, "tracker_id": self.tracker_labels.get(self.tracker.get(), ""), "roi": self.roi, "client_host": self.client_host, "metadata_port": self.metadata_port, "playback_fps": self.rate.get()}; self.run_id = self.api.request("POST", "/v1/runs", payload)["id"]
            if self.selected_profile().get("requires_roi", True): self.run_context = {"started_at": datetime.now().astimezone().isoformat(timespec="seconds"), "name": self.preset_name.get(), "selection": self.selection(), "profile": self.selected_profile(), "local_source": str(local), "ground_truth_path": self.ground_truth.get()}
            state = self.state_document(); run = self.selection(); state["last_run"] = run; state["recent_runs"] = [run, *state["recent_runs"]][:10]; self.write_state(state); self.refresh_saved_runs(state); self.status.set(f"Running {self.run_id}")
        except (OSError, RuntimeError, ValueError) as error:
            if self.viewer: self.viewer.close(); self.viewer = None
            logger.bind(model=self.model_name()).opt(exception=error).error("Start failed source={}", self.source.get())
            self.status.set(f"Start failed: {error}")
    def write_client_report(self, status):
        if not self.viewer or not self.run_context: return None
        profile, selection = self.run_context["profile"], self.run_context["selection"]
        metadata = {"schema_version": 1, "run_id": self.run_id, "status": status, "name": self.run_context["name"] or self.run_id,
                    "started_at": self.run_context["started_at"], "ended_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "dataset": selection["dataset"], "source": {"server_path": selection["source_path"], "local_path": self.run_context["local_source"], "original_width": self.viewer.source_width, "original_height": self.viewer.source_height}, "tracker_size": {"width": 640, "height": 360},
                    "tracker": {"id": selection["tracker_id"], "label": profile.get("label"), "model": str(profile.get("properties", {}).get("model", profile.get("label", "-"))), "properties": profile.get("properties", {})},
                    "roi": selection["roi"], "playback_fps": selection["playback_fps"], "ground_truth": {"path": self.run_context["ground_truth_path"] or None, "display_path": selection["ground_truth"] or None, "format": self.ground_truth_format}, "client_metrics": self.viewer.metrics()}
        directory = Path(self.report_location).expanduser(); directory = directory if directory.is_absolute() else self.config_path.parent / directory
        return write_run_report(directory / self.run_id, metadata, self.viewer.frame_rows)
    def stop(self):
        if self.run_id:
            try: self.api.request("POST", f"/v1/runs/{self.run_id}/stop")
            except RuntimeError as error: self.status.set(f"Stop failed: {error}")
        path = self.write_client_report("stopped")
        if self.viewer: self.viewer.close(); self.viewer = None
        if path: self.status.set(f"stopped: client report {path}")
        self.run_id = self.run_context = None
        self.source_only = False
    def show_pipeline(self):
        if not self.run_id: self.status.set("Start a benchmark first"); return
        try: pipeline = self.api.request("GET", f"/v1/runs/{self.run_id}").get("pipeline")
        except RuntimeError as error: self.status.set(f"Pipeline unavailable: {error}"); return
        window = tk.Toplevel(self.root); text = tk.Text(window, width=108, height=8, wrap="word"); text.pack(fill="both", expand=True, padx=8, pady=8); text.insert("1.0", pipeline or "Pipeline is not ready"); text.configure(state="disabled")
    def open_report_analysis(self):
        window = tk.Toplevel(self.root); window.title("Analyze tracker report")
        path, summary, tracker_color, truth_color = tk.StringVar(), tk.StringVar(value="Select a report folder"), tk.StringVar(value="#00cc44"), tk.StringVar(value="#e53935")
        loaded, player = {"result": None}, {"item": None}
        top = ttk.Frame(window, padding=10); top.pack(fill="x")
        ttk.Entry(top, textvariable=path, width=72).pack(side="left", fill="x", expand=True)
        def choose():
            selected = filedialog.askdirectory(title="Select client report folder", initialdir=path.get() or str(Path(self.report_location).expanduser()))
            if selected: path.set(selected); load()
        ttk.Button(top, text="Browse", command=choose).pack(side="left", padx=(6, 0))
        canvas = tk.Canvas(window, width=760, height=340, background="white"); canvas.pack(fill="both", expand=True, padx=10, pady=(0, 6))
        ttk.Label(window, textvariable=summary, padding=(10, 0, 10, 10)).pack(anchor="w")
        def draw(scores):
            canvas.delete("all"); width, height, left, right, top, bottom = int(canvas["width"]), int(canvas["height"]), 48, 18, 16, 34
            canvas.create_line(left, top, left, height - bottom); canvas.create_line(left, height - bottom, width - right, height - bottom)
            for score in (0, .5, 1):
                y = height - bottom - score * (height - top - bottom); canvas.create_line(left, y, width - right, y, fill="#dddddd"); canvas.create_text(left - 7, y, text=f"{score:g}", anchor="e")
            if not scores: return
            span = max(len(scores) - 1, 1); points = [(left + index * (width - left - right) / span, height - bottom - score * (height - top - bottom)) for index, (_frame, score) in enumerate(scores)]
            if len(points) == 1: canvas.create_oval(points[0][0] - 3, points[0][1] - 3, points[0][0] + 3, points[0][1] + 3, fill="#1473e6", outline="")
            else: canvas.create_line(*[value for point in points for value in point], fill="#1473e6", width=2)
            canvas.create_text((left + width - right) / 2, height - 10, text="frame")
            canvas.create_text(14, (top + height - bottom) / 2, text="IoU", angle=90)
        def load():
            try:
                result = analyze_report(path.get()); loaded["result"] = result; draw(result["scores"])
                mean = "n/a" if result["mean_iou"] is None else f"{result['mean_iou']:.4f}"
                minimum = "n/a" if result["minimum_iou"] is None else f"{result['minimum_iou']:.4f}"
                summary.set(f"frames: {result['frames']}  mean IoU: {mean}  minimum IoU: {minimum}  missing tracker: {result['missing_tracker']}  missing ground truth: {result['missing_ground_truth']}")
            except (OSError, ValueError, json.JSONDecodeError) as error:
                messagebox.showerror("Report analysis failed", str(error), parent=window)
        ttk.Button(top, text="Analyze", command=load).pack(side="left", padx=(6, 0))
        colors = ttk.Frame(window, padding=(10, 0, 10, 10)); colors.pack(fill="x")
        def choose_color(variable, button):
            chosen = colorchooser.askcolor(variable.get(), parent=window)[1]
            if chosen: variable.set(chosen); button.configure(background=chosen)
        tracker_button = tk.Button(colors, text="Tracker color", background=tracker_color.get()); tracker_button.configure(command=lambda: choose_color(tracker_color, tracker_button)); tracker_button.pack(side="left")
        truth_button = tk.Button(colors, text="Ground-truth color", background=truth_color.get()); truth_button.configure(command=lambda: choose_color(truth_color, truth_button)); truth_button.pack(side="left", padx=(6, 0))
        def stop_player():
            if player["item"]: player["item"].close(); player["item"] = None
        def play():
            result = loaded["result"]
            if result is None: load(); result = loaded["result"]
            if result is None: return
            stop_player()
            try:
                player["item"] = ReportPlayer(result["metadata"]["source"]["local_path"], result["metadata"].get("playback_fps", "Auto")); player["item"].start()
            except ValueError as error: messagebox.showerror("Playback failed", str(error), parent=window); return
            def bgr(color):
                red, green, blue = (int(color[index:index + 2], 16) for index in (1, 3, 5)); return blue, green, red
            def tick():
                active = player["item"]
                if not active: return
                item = active.poll()
                if item:
                    frame_id, frame = item
                    for box, color, label in ((result["truth_boxes"].get(frame_id), truth_color.get(), "ground truth"), (result["tracker_boxes"].get(frame_id), tracker_color.get(), "tracker")):
                        if box:
                            x, y, width, height = map(round, box); cv2.rectangle(frame, (x, y), (x + width, y + height), bgr(color), 2); cv2.putText(frame, label, (x, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, .55, bgr(color), 2)
                    cv2.imshow("Tracker vs ground truth", frame)
                    if cv2.waitKey(1) == 27: stop_player(); return
                if active.finished(): stop_player()
                else: window.after(10, tick)
            tick()
        ttk.Button(colors, text="Play video", command=play).pack(side="left", padx=(12, 0))
        ttk.Button(colors, text="Stop video", command=stop_player).pack(side="left", padx=(6, 0))
        window.protocol("WM_DELETE_WINDOW", lambda: (stop_player(), window.destroy()))
    def tick(self):
        now = time.monotonic()
        if self.preview_open: cv2.waitKey(1)
        if self.viewer: self.viewer.poll()
        if self.source_only and self.viewer and self.viewer.finished(): self.status.set("Local source finished"); self.viewer.close(); self.viewer = None; self.source_only = False
        if not self.run_id and now >= self.next_server_check:
            self.next_server_check = now + 2
            try: self.api.request("GET", "/v1/health", timeout=1.5); self.server_status.set("Server: connected")
            except RuntimeError: self.server_status.set("Server: unavailable")
        if self.run_id and now >= self.next_run_check:
            self.next_run_check = now + .5
            try:
                result = self.api.request("GET", f"/v1/runs/{self.run_id}"); self.server_status.set("Server: connected")
                if result["status"] in ("failed", "stopped"):
                    path = self.write_client_report(result["status"])
                    if self.viewer: self.viewer.close(); self.viewer = None
                    logger.bind(model=self.model_name()).error("Run {} {}: {}", self.run_id, result["status"], result.get("error") or "no error detail")
                    self.status.set(f"{result['status']}: client report {path}" if path else f"{result['status']}: artifacts ready under {self.run_id}"); self.run_id = self.run_context = None
                elif result["status"] == "finished" and self.viewer and self.viewer.finished():
                    self.api.request("POST", f"/v1/runs/{self.run_id}/client-metrics", self.viewer.metrics()); path = self.write_client_report("finished"); self.viewer.close(); self.viewer = None; self.status.set(f"finished: client report {path}" if path else f"finished: artifacts ready under {self.run_id}"); self.run_id = self.run_context = None
            except RuntimeError as error: self.server_status.set("Server: unavailable"); self.status.set(f"Status failed: {error}")
        self.root.after(30, self.tick)
    def close(self): self.stop(); cv2.destroyAllWindows(); self.root.destroy()
    def run(self): self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=Path(__file__).with_name("client.yaml")); Client(parser.parse_args().config).run()
if __name__ == "__main__": main()
