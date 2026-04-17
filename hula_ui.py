"""
Hula Drone - Flight Control UI with YOLO OBB + Pose + Live Detection Tuning
Install: pip install pyhula opencv-python ultralytics pillow

Controls via buttons AND keyboard:
  T         - Takeoff        L - Land
  W/S       - Forward/Back   A/D - Left/Right
  Up/Down   - Altitude       Q/E - Rotate
  Space     - Hover          ESC - Emergency land & quit
  M         - Cycle model (locks to MANUAL mode)
  R         - Toggle recording
"""

import threading
import time
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import pyhula


# ── Configuration ────────────────────────────────────────────────────────────
MODEL_PATHS = {
    "obb":  "C:\\Users\\rizro\\HulaDroneControlApp-desktop\\runs\\obb\\train35\\weights\\best.pt",
    "pose": "C:\\Users\\rizro\\HulaDroneControlApp-desktop\\runs\\pose\\train16\\weights\\best.pt",
}
ACTIVE_MODEL      = "obb"             # starting model key
DRONE_IP          = "192.168.100.85"
MOVE_DIST         = 30
MOVE_SPEED        = 60
ROTATE_DEG        = 30
VIDEO_W           = 960
VIDEO_H           = 540
SWITCH_EVAL_EVERY = 30    # kept for reference (no longer used in auto mode)
SWITCH_HYSTERESIS = 0.05  # kept for reference

# ── Environmental model selection (condition_analysis-derived thresholds) ──
# Thresholds were read from the raw_frames CSVs:
#   brightness: dim ~50 → "dim", bright ~107-121 → "normal".  Cutoff ≈ 70.
#   blur_var:   heavy_blur 63-95, slight_blur 124+.  Cutoff ≈ 100 (Laplacian var).
#   contrast:   low <30, mid 30-57, high 60+.  High-contrast OBB edge starts at ≈ 58.
# Confidence margins that drove these rules (from condition_summary CSVs):
#   dim any condition       → POSE +0.07-0.10 advantage  (strong signal)
#   bright + heavy blur     → POSE slight advantage
#   bright + high contrast  → OBB  +0.02 advantage       (marginal)
#   bright + normal         → POSE slight advantage       (default)
ENV_BRIGHTNESS_DIM  = 70    # mean grayscale < this → dim lighting → POSE
ENV_BLUR_HEAVY      = 100   # Laplacian variance < this → heavy blur → POSE
ENV_CONTRAST_HIGH   = 58    # grayscale std >= this AND bright AND sharp → OBB
ENV_EVAL_EVERY      = 5     # measure env every N inference frames (cheap: pure OpenCV)
ENV_STABLE_FRAMES   = 5     # env recommendation must be stable N evals before switching
ENV_EMA_ALPHA       = 0.25  # smoothing factor applied to raw env readings
POSE_MIN_FPS        = 1.5   # if Pose inference FPS drops below this, force OBB
                            # baseline from condition_analysis: Pose normal = 2.47-2.71 fps


# ── Shared video state ────────────────────────────────────────────────
latest_frame     = None
frame_lock       = threading.Lock()
_original_imshow = cv2.imshow

# Capture-side FPS tracking and recording writer
_capture_times          = []
_capture_fps            = 10
_intercept_writer       = None
_intercept_writer_lock  = threading.Lock()

def _intercept_imshow(winname, img):
    global latest_frame, _capture_fps, _intercept_writer
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return

    now = time.time()

    with frame_lock:
        latest_frame = img.copy()

    _capture_times.append(now)
    while _capture_times and now - _capture_times[0] > 2.0:
        _capture_times.pop(0)
    if len(_capture_times) > 1:
        _capture_fps = len(_capture_times) / (now - _capture_times[0])

    with _intercept_writer_lock:
        if _intercept_writer is not None:
            _intercept_writer.write(img)

cv2.imshow = _intercept_imshow


# ── Connect helper ────────────────────────────────────────────────────
def connect_drone(api, ip=None, retries=3):
    for attempt in range(1, retries + 1):
        print(f"[INFO] Connect attempt {attempt}/{retries}"
              + (f" to {ip}" if ip else " (auto-detect)") + "...")
        try:
            ok = api.connect(ip) if ip else api.connect()
            if ok:
                return True
        except Exception as e:
            print(f"[WARN] {e}")
        time.sleep(2)
    if ip:
        for _ in range(retries):
            try:
                if api.connect():
                    return True
            except Exception:
                pass
            time.sleep(2)
    return False


# ── Image enhancement ────────────────────────────────────────
def enhance_frame(img, sharpen=0.0, use_clahe=False, use_denoise=False):
    out = img.copy()
    if use_denoise:
        out = cv2.fastNlMeansDenoisingColored(out, None, 5, 5, 7, 21)
    if use_clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l  = cl.apply(l)
        out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    if sharpen > 0.01:
        kernel    = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
        sharpened = cv2.filter2D(out, -1, kernel)
        out = cv2.addWeighted(out, 1.0 - sharpen, sharpened, sharpen, 0)
    return out


# ── Styles ────────────────────────────────────────────────────────────
BG     = "#0a0a0f"
BG2    = "#0d0d14"
BG3    = "#12121c"
ACCENT = "#00ffe0"
WARN   = "#ffcc00"
DANGER = "#ff4455"
DIM    = "#333355"
TEXT   = "#aaaacc"
FONT   = "Courier New"


# ── Main App ──────────────────────────────────────────────────────────
class HulaDroneApp:
    def __init__(self, root):
        self.root           = root
        self.api            = None
        self.models         = {}          # key → YOLO instance
        self.model          = None        # active YOLO instance
        self.active_key     = ACTIVE_MODEL
        self.conf_scores     = {k: 0.0 for k in MODEL_PATHS}  # active-model running conf
        self.auto_mode       = True
        # Environmental switching state
        self._env_readings   = {"brightness": 0.0, "blur": 0.0,
                                 "contrast": 0.0, "saturation": 0.0}
        self._env_preferred  = ACTIVE_MODEL   # last env-recommended model
        self._env_stable_cnt = 0              # consecutive evals agreeing on _env_preferred
        self._env_reason     = "starting"     # human-readable label for current condition
        self._infer_times    = []             # timestamps of recent inference completions
        self._infer_fps      = 0.0            # rolling inference FPS (pose FPS fallback check)
        self.running        = False
        self.airborne       = False
        self.last_annotated = None
        self._fps_times     = []
        self._annotated_lock    = threading.Lock()
        self._infer_input       = None
        self._infer_input_lock  = threading.Lock()
        threading.Thread(target=self._infer_loop, daemon=True).start()

        self.conf_var    = tk.DoubleVar(value=0.65)
        self.imgsz_var   = tk.IntVar(value=640)
        self.every_n_var = tk.IntVar(value=2)
        self.sharpen_var = tk.DoubleVar(value=0.0)    # display only — off by default
        self.clahe_var   = tk.BooleanVar(value=False) # display only — off by default
        self.denoise_var = tk.BooleanVar(value=False)

        # Recording state
        self._writer      = None
        self._recording   = False
        self._record_lock = threading.Lock()

        self._build_ui()
        self._load_models()
        self._connect()

    # ── UI Build ──────────────────────────────────────────────────────
    def _build_ui(self):
        self.root.title("Hula Drone Control")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        self._build_topbar()

        content = tk.Frame(self.root, bg=BG)
        content.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self._build_left_col(content)
        self._build_right_col(content)

        self.root.bind("<KeyPress>", self._on_key)
        self.root.focus_set()
        self._update_video()
        self._update_telemetry()

    def _build_topbar(self):
        bar = tk.Frame(self.root, bg=BG2, height=48)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        tk.Label(bar, text="◈  HULA DRONE CONTROL",
                 font=(FONT, 13, "bold"), fg=ACCENT, bg=BG2
                 ).pack(side="left", padx=16, pady=12)
        self.status_label = tk.Label(bar, text="⬤  DISCONNECTED",
                                     font=(FONT, 10), fg=DANGER, bg=BG2)
        self.status_label.pack(side="right", padx=16)
        self.battery_label = tk.Label(bar, text="BAT: --%",
                                      font=(FONT, 10), fg="#666", bg=BG2)
        self.battery_label.pack(side="right", padx=8)

    def _build_left_col(self, parent):
        col = tk.Frame(parent, bg=BG)
        col.pack(side="left", padx=(0, 12))

        self.canvas = tk.Canvas(col, width=VIDEO_W, height=VIDEO_H,
                                bg="#111118", highlightthickness=1,
                                highlightbackground="#1e1e2e")
        self.canvas.pack()
        self._draw_placeholder()
        self._canvas_img_id = self.canvas.create_image(0, 0, anchor="nw")

        info = tk.Frame(col, bg=BG2, height=28)
        info.pack(fill="x")
        info.pack_propagate(False)
        self.det_label = tk.Label(info, text="DETECTIONS: --",
                                  font=(FONT, 9), fg="#555", bg=BG2)
        self.det_label.pack(side="left", padx=10)
        self.fps_label = tk.Label(info, text="FPS: --",
                                  font=(FONT, 9), fg="#555", bg=BG2)
        self.fps_label.pack(side="right", padx=10)

        self.rec_btn = tk.Button(info, text="⏺ REC",
                                 command=self.toggle_recording,
                                 font=(FONT, 8, "bold"),
                                 fg="#555", bg=BG2,
                                 activeforeground=DANGER,
                                 activebackground=BG2,
                                 relief="flat", cursor="hand2", bd=0)
        self.rec_btn.pack(side="right", padx=(0, 4))

        self._build_tuning_panel(col)

    def _build_tuning_panel(self, parent):
        panel = tk.Frame(parent, bg=BG2)
        panel.pack(fill="x", pady=(6, 0))
        self._section_lbl(panel, "DETECTION TUNING", padx=10)

        sliders = tk.Frame(panel, bg=BG2)
        sliders.pack(fill="x", padx=10, pady=(0, 6))
        ls = tk.Frame(sliders, bg=BG2)
        rs = tk.Frame(sliders, bg=BG2)
        ls.pack(side="left", expand=True, fill="x", padx=(0, 16))
        rs.pack(side="left", expand=True, fill="x")

        self._slider(ls, "CONFIDENCE THRESHOLD",  self.conf_var,    0.05, 0.95, 0.05, lambda v: f"{float(v):.2f}")
        self._slider(ls, "SHARPENING",             self.sharpen_var, 0.0,  1.0,  0.05, lambda v: f"{float(v):.0%}")
        self._slider(rs, "INFER EVERY N FRAMES",   self.every_n_var, 1,    6,    1,    lambda v: str(int(float(v))))
        self._slider(rs, "INFERENCE SIZE (px)",    self.imgsz_var,   320,  1280, 320,  lambda v: str(int(float(v))))

        tog = tk.Frame(panel, bg=BG2)
        tog.pack(fill="x", padx=10, pady=(2, 8))
        self._toggle(tog, "CLAHE CONTRAST BOOST", self.clahe_var)
        self._toggle(tog, "DENOISE (slower)",      self.denoise_var)

        tk.Label(panel,
                 text="TIP: Lower confidence + CLAHE helps most in-flight.  "
                      "Raise inference size for distant/small objects (slower).",
                 font=(FONT, 7), fg=DIM, bg=BG2,
                 wraplength=VIDEO_W - 24, justify="left"
                 ).pack(anchor="w", padx=10, pady=(0, 8))

    def _build_right_col(self, parent):
        col = tk.Frame(parent, bg=BG, width=230)
        col.pack(side="left", fill="y")
        col.pack_propagate(False)

        self._section_lbl(col, "FLIGHT CONTROLS")
        r1 = tk.Frame(col, bg=BG); r1.pack(fill="x", pady=(0,3))
        self._btn(r1,"▲ TAKEOFF",   self.cmd_takeoff,   ACCENT, "#003830").pack(side="left",expand=True,fill="x",padx=(0,2))
        self._btn(r1,"▼ LAND",      self.cmd_land,      WARN,   "#332a00").pack(side="left",expand=True,fill="x",padx=(2,0))
        r2 = tk.Frame(col, bg=BG); r2.pack(fill="x", pady=(0,10))
        self._btn(r2,"◉ HOVER",     self.cmd_hover,     "#aaaaff","#1a1a33").pack(side="left",expand=True,fill="x",padx=(0,2))
        self._btn(r2,"✕ EMERGENCY", self.cmd_emergency, DANGER, "#330010").pack(side="left",expand=True,fill="x",padx=(2,0))

        self._section_lbl(col, "DIRECTION")
        dpad = tk.Frame(col, bg=BG); dpad.pack(pady=(0,10))
        for txt, cmd, r, c in [("↑\nFWD",self.cmd_forward,0,1),("←\nLEFT",self.cmd_left,1,0),("→\nRGHT",self.cmd_right,1,2),("↓\nBACK",self.cmd_backward,2,1)]:
            tk.Button(dpad,text=txt,command=cmd,width=6,height=2,font=(FONT,8,"bold"),fg=TEXT,bg=BG3,activeforeground=TEXT,activebackground=BG3,relief="flat",cursor="hand2").grid(row=r,column=c,padx=2,pady=2)
        tk.Label(dpad,text="■",width=6,height=2,font=(FONT,9),fg=DIM,bg=BG).grid(row=1,column=1,padx=2,pady=2)

        self._section_lbl(col, "ALTITUDE & ROTATION")
        r3 = tk.Frame(col, bg=BG); r3.pack(fill="x", pady=(0,3))
        self._btn(r3,"⬆ UP",    self.cmd_up,       "#88ffcc","#0d2e1e").pack(side="left",expand=True,fill="x",padx=(0,2))
        self._btn(r3,"⬇ DOWN",  self.cmd_down,     "#ff8866","#2e1200").pack(side="left",expand=True,fill="x",padx=(2,0))
        r4 = tk.Frame(col, bg=BG); r4.pack(fill="x", pady=(0,10))
        self._btn(r4,"↺ ROT L", self.cmd_rot_left, "#ccaaff","#1e1033").pack(side="left",expand=True,fill="x",padx=(0,2))
        self._btn(r4,"↻ ROT R", self.cmd_rot_right,"#ccaaff","#1e1033").pack(side="left",expand=True,fill="x",padx=(2,0))

        # ── Model switching panel ──────────────────────────────────────
        self._section_lbl(col, "MODEL SWITCHING")

        self.model_name_label = tk.Label(col, text=f"MODEL: {ACTIVE_MODEL.upper()}",
                                         font=(FONT, 8, "bold"), fg=ACCENT, bg=BG)
        self.model_name_label.pack(anchor="w", pady=(0, 2))

        self.model_mode_label = tk.Label(col, text="MODE: AUTO",
                                         font=(FONT, 8), fg=WARN, bg=BG)
        self.model_mode_label.pack(anchor="w", pady=(0, 2))

        self.model_conf_label = tk.Label(col, text="",
                                         font=(FONT, 7), fg=DIM, bg=BG,
                                         justify="left")
        self.model_conf_label.pack(anchor="w", pady=(0, 2))

        self.env_label = tk.Label(col, text="ENV: --",
                                  font=(FONT, 7), fg="#335544", bg=BG,
                                  justify="left")
        self.env_label.pack(anchor="w", pady=(0, 6))

        mbtn_row = tk.Frame(col, bg=BG)
        mbtn_row.pack(fill="x", pady=(0, 4))
        self._btn(mbtn_row, "◀▶ CYCLE",   self._manual_cycle_model,
                  TEXT, BG3).pack(side="left", expand=True, fill="x", padx=(0, 2))
        self._btn(mbtn_row, "⟳ AUTO",     self._enable_auto_mode,
                  "#aaffcc", "#0d2e1e").pack(side="left", expand=True, fill="x", padx=(2, 0))

        tk.Label(col, text="M=cycle (manual)  Auto=environment-driven\n(brightness/blur/contrast from condition_analysis)",
                 font=(FONT, 6), fg=DIM, bg=BG, justify="left",
                 wraplength=210).pack(anchor="w", pady=(0, 8))

        # ── Telemetry ──────────────────────────────────────────────────
        self._section_lbl(col, "TELEMETRY")
        self.telem_text = tk.Text(col, height=6, width=26, bg=BG2, fg=ACCENT,
                                  font=(FONT, 8), relief="flat", state="disabled",
                                  insertbackground=ACCENT)
        self.telem_text.pack(fill="x", pady=(0,10))

        tk.Label(col, text="KEYS: T=Takeoff  L=Land\nW/S=Fwd/Back  A/D=Left/Right\n↑↓=Altitude   Q/E=Rotate\nSpace=Hover   ESC=Emergency\nM=Cycle model  R=Record",
                 font=(FONT,7), fg=DIM, bg=BG, justify="left").pack(anchor="w")

    # ── Widget helpers ────────────────────────────────────────────────
    def _section_lbl(self, parent, title, padx=0):
        tk.Label(parent, text=title, font=(FONT,8,"bold"), fg="#444466", bg=parent["bg"]).pack(anchor="w", pady=(8,2), padx=padx)
        tk.Frame(parent, bg="#1e1e2e", height=1).pack(fill="x", pady=(0,4), padx=padx)

    def _btn(self, parent, text, cmd, fg=TEXT, bg=BG3):
        return tk.Button(parent, text=text, command=cmd, font=(FONT,8,"bold"),
                         fg=fg, bg=bg, activeforeground=fg, activebackground=bg,
                         relief="flat", cursor="hand2", pady=5)

    def _slider(self, parent, label, var, from_, to, res, fmt):
        row = tk.Frame(parent, bg=parent["bg"]); row.pack(fill="x", pady=(0,6))
        hdr = tk.Frame(row, bg=parent["bg"]); hdr.pack(fill="x")
        tk.Label(hdr, text=label, font=(FONT,7,"bold"), fg="#555577", bg=parent["bg"]).pack(side="left")
        val_lbl = tk.Label(hdr, text=fmt(var.get()), font=(FONT,7), fg=ACCENT, bg=parent["bg"])
        val_lbl.pack(side="right")
        tk.Scale(row, variable=var, from_=from_, to=to, resolution=res,
                 orient="horizontal", bg=parent["bg"], fg=TEXT, troughcolor=BG3,
                 highlightthickness=0, sliderrelief="flat", activebackground=ACCENT,
                 showvalue=False, command=lambda v: val_lbl.config(text=fmt(v))
                 ).pack(fill="x")

    def _toggle(self, parent, label, var):
        def toggle():
            var.set(not var.get())
            btn.config(fg=ACCENT if var.get() else "#333355",
                       bg="#003830" if var.get() else BG3)
        btn = tk.Button(parent, text=f"◉ {label}", command=toggle,
                        font=(FONT,7,"bold"), relief="flat", cursor="hand2",
                        pady=3, padx=6,
                        fg=ACCENT if var.get() else "#333355",
                        bg="#003830" if var.get() else BG3)
        btn.pack(side="left", padx=(0,12))

    def _draw_placeholder(self):
        self.canvas.create_rectangle(0, 0, VIDEO_W, VIDEO_H, fill="#111118", outline="")
        self.canvas.create_text(VIDEO_W//2, VIDEO_H//2, text="AWAITING VIDEO STREAM",
                                font=(FONT,14), fill="#222233")

    # ── Pose box-only renderer ────────────────────────────────────────
    def _draw_pose_boxes(self, img, result):
        """Draw bounding boxes and labels for pose detections without keypoints."""
        out  = img.copy()
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss  = result.boxes.cls.cpu().numpy().astype(int)
        names = result.names  # {class_id: class_name}

        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = f"{names.get(cls_id, cls_id)}: {conf:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(out, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        return out

    # ── Model switching helpers ───────────────────────────────────────

    def _mean_conf(self, key, m, img):
        """Return mean raw confidence for model m on img, or 0.0.
        Uses result.obb for OBB models and result.boxes for all others —
        reading from the wrong attribute was silently zeroing OBB scores,
        causing auto-switch to never return to OBB."""
        all_confs = []
        is_obb    = getattr(m, "task", "") == "obb"
        try:
            for result in m.predict(source=img, verbose=False,
                                    conf=self.conf_var.get()):
                preds = result.obb if is_obb else result.boxes
                if preds is not None and len(preds):
                    all_confs.extend(preds.conf.cpu().tolist())
        except Exception:
            pass
        return float(sum(all_confs) / len(all_confs)) if all_confs else 0.0

    def _switch_to(self, key, reason=""):
        """Switch active model to key and reset annotations.
        Safe to call from any thread — Tk widget updates go through root.after."""
        self.active_key = key
        self.model      = self.models[key]
        with self._annotated_lock:
            self.last_annotated = None
        self.root.after(0, self._refresh_model_labels)
        print(f"[{reason}] Switched to {key} ({MODEL_PATHS[key]})")

    def _refresh_model_labels(self):
        """Update the model panel labels from current state."""
        self.model_name_label.config(text=f"MODEL: {self.active_key.upper()}")
        mode_str = "ENV-AUTO" if self.auto_mode else "MANUAL"
        self.model_mode_label.config(
            text=f"MODE: {mode_str}",
            fg=WARN if self.auto_mode else "#ccaaff"
        )
        conf_lines = "\n".join(
            f"{'►' if k == self.active_key else ' '} {k}: {self.conf_scores[k]:.2f}"
            for k in self.models
        )
        self.model_conf_label.config(text=conf_lines)

        env = self._env_readings
        env_str = (f"br:{env['brightness']:.0f}  bl:{env['blur']:.0f}\n"
                   f"ct:{env['contrast']:.0f}  sat:{env['saturation']:.0f}\n"
                   f"→ {self._env_reason}")
        self.env_label.config(text=env_str)

    def _manual_cycle_model(self):
        if not self.models:
            return
        self.auto_mode = False
        keys = list(self.models.keys())
        next_key = keys[(keys.index(self.active_key) + 1) % len(keys)]
        self._switch_to(next_key, "MANUAL")

    def _enable_auto_mode(self):
        self.auto_mode       = True
        self._env_stable_cnt = 0    # force re-evaluation on next env check
        self._refresh_model_labels()
        print("[ENV-AUTO] Environment-driven switching enabled")

    # ── Environment measurement & decision ───────────────────────────
    @staticmethod
    def _measure_env(img):
        """Compute per-frame environment metrics that match the condition_analysis columns.

        Returns (brightness, blur_var, contrast, saturation):
          brightness — mean grayscale intensity
          blur_var   — Laplacian variance (lower = blurrier; heavy_blur < 100)
          contrast   — grayscale standard deviation
          saturation — mean S channel of HSV
        """
        gray       = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        blur_var   = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast   = float(np.std(gray))
        hsv        = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = float(np.mean(hsv[:, :, 1]))
        return brightness, blur_var, contrast, saturation

    def _env_preferred_model(self, brightness, blur_var, contrast):
        """Rule-based model preference derived from condition_analysis data.

        Priority order (strongest signal first):
          1. Dim lighting (brightness < ENV_BRIGHTNESS_DIM):
               POSE wins by +0.07-0.10 in ALL dim conditions (dim_motion, dim_static).
          2. Bright + heavy blur (blur_var < ENV_BLUR_HEAVY):
               POSE wins in bright_motion heavy-blur frames.
          3. Bright + slight blur + high contrast (contrast >= ENV_CONTRAST_HIGH):
               OBB wins by +0.02 in bright_static high-contrast frames.
          4. Default (bright, normal):
               POSE wins slightly — chosen as the safe default.

        Returns the string key ("obb" or "pose") and sets self._env_reason.
        Saturation is measured but not used as a decision signal — its margins
        in the condition_analysis data were < 0.01 and not reliable enough.
        """
        if brightness < ENV_BRIGHTNESS_DIM:
            self._env_reason = f"dim(br={brightness:.0f})→POSE"
            return "pose"
        if blur_var < ENV_BLUR_HEAVY:
            self._env_reason = f"heavyblur(bl={blur_var:.0f})→POSE"
            return "pose"
        if contrast >= ENV_CONTRAST_HIGH:
            self._env_reason = f"highcontrast(ct={contrast:.0f})→OBB"
            return "obb"
        self._env_reason = "bright+normal→POSE"
        return "pose"


    # ── Video display loop (main thread — display only, no inference) ──
    def _update_video(self):
        with frame_lock:
            img = latest_frame.copy() if latest_frame is not None else None

        if img is not None:
            # Queue latest frame for the inference thread
            with self._infer_input_lock:
                self._infer_input = img

            with self._annotated_lock:
                display = self.last_annotated.copy() if self.last_annotated is not None else img

            # ── HUD overlay ───────────────────────────────────────────
            mode_tag = "ENV-AUTO" if self.auto_mode else "MANUAL"
            env      = self._env_readings
            hud_line1 = (f"[{mode_tag}] {self.active_key.upper()}"
                         f"  br:{env['brightness']:.0f}"
                         f"  bl:{env['blur']:.0f}"
                         f"  ct:{env['contrast']:.0f}")
            hud_line2 = self._env_reason
            cv2.putText(display, hud_line1,
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display, hud_line2,
                        (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 120), 1, cv2.LINE_AA)

            rgb = cv2.cvtColor(cv2.resize(display, (VIDEO_W, VIDEO_H)), cv2.COLOR_BGR2RGB)
            self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.canvas.itemconfig(self._canvas_img_id, image=self._photo)

            now = time.time()
            self._fps_times.append(now)
            self._fps_times = [t for t in self._fps_times if now - t < 1.0]
            self.fps_label.config(
                text=f"UI:{len(self._fps_times)}fps  CAM:{_capture_fps:.0f}fps",
                fg=ACCENT)

        self.root.after(33, self._update_video)

    # ── Inference loop (background thread — GPU work lives here) ──────
    def _infer_loop(self):
        frame_count = 0
        while True:
            with self._infer_input_lock:
                img = self._infer_input
                self._infer_input = None

            if img is None or self.model is None:
                time.sleep(0.005)
                continue

            conf    = self.conf_var.get()   # raw slider value — no scaling applied
            imgsz   = int(self.imgsz_var.get())
            every_n = int(self.every_n_var.get())
            # Inference always uses the raw frame — enhancements shift the
            # pixel distribution away from training data and hurt confidence.
            # enhanced is kept for display/drawing only.
            infer_frame = img
            enhanced    = enhance_frame(img,
                                        sharpen=self.sharpen_var.get(),
                                        use_clahe=self.clahe_var.get(),
                                        use_denoise=self.denoise_var.get())
            frame_count += 1

            # ── Environment-based auto-switch ────────────────────────
            # Measures brightness/blur/contrast directly from the frame —
            # no second-model inference needed, so zero extra GPU cost.
            if self.auto_mode and frame_count % ENV_EVAL_EVERY == 0:

                # FPS fallback — highest priority: if Pose is too slow, force OBB
                # immediately (no stability counter needed — degradation is obvious).
                # Baseline from condition_analysis: Pose normal = 2.47-2.71 fps.
                if (self.active_key == "pose"
                        and self._infer_fps > 0          # wait until we have a reading
                        and self._infer_fps < POSE_MIN_FPS):
                    self._env_reason = f"POSE FPS too low ({self._infer_fps:.1f}<{POSE_MIN_FPS})→OBB"
                    print(f"[FPS-FALLBACK] Pose inference FPS={self._infer_fps:.2f}"
                          f" < {POSE_MIN_FPS} — forcing OBB")
                    self._switch_to("obb", "FPS-FALLBACK")
                    continue

                br, bl, ct, sat = self._measure_env(img)
                # EMA-smooth each metric to suppress frame-to-frame noise
                for key, val in zip(("brightness", "blur", "contrast", "saturation"),
                                    (br, bl, ct, sat)):
                    prev = self._env_readings[key]
                    self._env_readings[key] = (prev + ENV_EMA_ALPHA * (val - prev)
                                               if prev != 0.0 else val)

                preferred = self._env_preferred_model(
                    self._env_readings["brightness"],
                    self._env_readings["blur"],
                    self._env_readings["contrast"],
                )

                # Require ENV_STABLE_FRAMES consecutive agreements before switching
                if preferred == self._env_preferred:
                    self._env_stable_cnt += 1
                else:
                    self._env_preferred  = preferred
                    self._env_stable_cnt = 1

                if (self._env_stable_cnt >= ENV_STABLE_FRAMES
                        and preferred != self.active_key):
                    print(f"[ENV-AUTO] {self.active_key} → {preferred}"
                          f"  ({self._env_reason})")
                    self._switch_to(preferred, "ENV-AUTO")
                else:
                    self.root.after(0, self._refresh_model_labels)

            # ── Regular inference ─────────────────────────────────────
            if frame_count % every_n == 0:
                try:
                    dets = 0
                    annotated  = None
                    is_obb_run = self.active_key == "obb"
                    for result in self.model.predict(source=infer_frame, stream=True,
                                                     verbose=False, conf=conf, imgsz=imgsz):
                        # Use the correct attribute depending on model type
                        preds = result.obb if is_obb_run else result.boxes

                        # Best-per-class filter
                        if preds is not None and len(preds) > 1:
                            cls    = preds.cls
                            conf_t = preds.conf
                            keep   = []
                            for c in cls.unique():
                                mask   = (cls == c).nonzero(as_tuple=True)[0]
                                best_i = mask[conf_t[mask].argmax()].item()
                                keep.append(best_i)
                            keep = sorted(keep)
                            preds.data = preds.data[keep]
                            if not is_obb_run and result.keypoints is not None:
                                result.keypoints.data = result.keypoints.data[keep]

                        if self.active_key == "pose" and result.boxes is not None:
                            annotated = self._draw_pose_boxes(enhanced, result)
                        else:
                            annotated = result.plot()
                        dets = len(preds) if preds is not None else 0
                        # Track active-model confidence for panel display
                        if preds is not None and len(preds):
                            self.conf_scores[self.active_key] = float(
                                preds.conf.cpu().mean())

                    if annotated is not None:
                        with self._annotated_lock:
                            self.last_annotated = annotated

                    # Rolling inference FPS (2-second window)
                    _now = time.time()
                    self._infer_times.append(_now)
                    self._infer_times = [t for t in self._infer_times
                                         if _now - t < 2.0]
                    if len(self._infer_times) > 1:
                        self._infer_fps = (len(self._infer_times) /
                                           (_now - self._infer_times[0]))

                    d = dets
                    self.root.after(0, lambda d=d: self.det_label.config(
                        text=f"DETECTIONS: {d}", fg=ACCENT if d > 0 else "#555"))
                except Exception:
                    pass

    # ── Telemetry loop ────────────────────────────────────────────────
    def _update_telemetry(self):
        if self.api and self.running:
            try:
                bat = self.api.get_battery()
                self.battery_label.config(text=f"BAT: {bat}%",
                                          fg=ACCENT if bat > 30 else DANGER)
                lines = (f"BAT  : {bat}%\n"
                         f"POS  : {self.api.get_coordinate()}\n"
                         f"YAW  : {self.api.get_yaw()}\n"
                         f"SPD  : {self.api.get_plane_speed()}\n"
                         f"TOF  : {self.api.get_plane_distance()} cm")
                self.telem_text.config(state="normal")
                self.telem_text.delete("1.0", "end")
                self.telem_text.insert("end", lines)
                self.telem_text.config(state="disabled")
            except Exception:
                pass
        self.root.after(1000, self._update_telemetry)

    # ── Connect & model ───────────────────────────────────────────────
    def _load_models(self):
        def _load():
            print("[INFO] Loading models...")
            for key, path in MODEL_PATHS.items():
                print(f"  [{key}] {path}")
                self.models[key] = YOLO(path)
            self.model       = self.models[self.active_key]
            self.conf_scores = {k: 0.0 for k in self.models}
            print(f"[INFO] Models loaded. Active: {self.active_key}")
            self.root.after(0, self._refresh_model_labels)
        threading.Thread(target=_load, daemon=True).start()

    def _connect(self):
        def _do():
            self.api = pyhula.UserApi()
            if not connect_drone(self.api, ip=DRONE_IP):
                self.status_label.config(text="⬤  CONNECT FAILED", fg=DANGER)
                return
            self.running = True
            self.status_label.config(text="⬤  CONNECTED", fg=ACCENT)
            self.api.Plane_cmd_swith_rtp(0)
            time.sleep(1.0)
            threading.Thread(target=self.api.single_fly_flip_rtp, daemon=True).start()
            print("[INFO] Stream started")
        threading.Thread(target=_do, daemon=True).start()

    # ── Flight commands ───────────────────────────────────────────────
    def _guard(self): return self.api is not None and self.running

    def cmd_takeoff(self):
        if not self._guard(): return
        self.airborne = True
        self.status_label.config(text="⬤  AIRBORNE", fg=WARN)
        threading.Thread(target=self.api.single_fly_takeoff, daemon=True).start()

    def cmd_land(self):
        if not self._guard(): return
        self.airborne = False
        self.status_label.config(text="⬤  LANDING", fg="#aaaaff")
        threading.Thread(target=self.api.single_fly_touchdown, daemon=True).start()

    def cmd_hover(self):
        if not self._guard(): return
        threading.Thread(target=lambda: self.api.single_fly_hover_flight(3), daemon=True).start()

    def cmd_emergency(self):
        if not self._guard(): return
        self.airborne = False
        self.status_label.config(text="⬤  EMERGENCY", fg=DANGER)
        threading.Thread(target=self.api.single_fly_touchdown, daemon=True).start()

    def cmd_forward(self):
        if not self._guard(): return
        threading.Thread(target=lambda: self.api.single_fly_forward(MOVE_DIST, MOVE_SPEED), daemon=True).start()

    def cmd_backward(self):
        if not self._guard(): return
        threading.Thread(target=lambda: self.api.single_fly_back(MOVE_DIST, MOVE_SPEED), daemon=True).start()

    def cmd_left(self):
        if not self._guard(): return
        threading.Thread(target=lambda: self.api.single_fly_left(MOVE_DIST, MOVE_SPEED), daemon=True).start()

    def cmd_right(self):
        if not self._guard(): return
        threading.Thread(target=lambda: self.api.single_fly_right(MOVE_DIST, MOVE_SPEED), daemon=True).start()

    def cmd_up(self):
        if not self._guard(): return
        threading.Thread(target=lambda: self.api.single_fly_up(MOVE_DIST, MOVE_SPEED), daemon=True).start()

    def cmd_down(self):
        if not self._guard(): return
        threading.Thread(target=lambda: self.api.single_fly_down(MOVE_DIST, MOVE_SPEED), daemon=True).start()

    def cmd_rot_left(self):
        if not self._guard(): return
        threading.Thread(target=lambda: self.api.single_fly_turnleft(ROTATE_DEG), daemon=True).start()

    def cmd_rot_right(self):
        if not self._guard(): return
        threading.Thread(target=lambda: self.api.single_fly_turnright(ROTATE_DEG), daemon=True).start()

    # ── Recording ─────────────────────────────────────────────────────
    def toggle_recording(self):
        with self._record_lock:
            if self._recording:
                self._stop_recording_locked()
            else:
                self._start_recording_locked()

    def _start_recording_locked(self):
        global _intercept_writer
        with frame_lock:
            ref = latest_frame
        if ref is None:
            print("[WARN] No frame available yet — start stream first")
            return

        h, w      = ref.shape[:2]
        fps       = max(1.0, round(_capture_fps))
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename  = f"hula_raw_{timestamp}.avi"
        fourcc    = cv2.VideoWriter_fourcc(*"XVID")
        writer    = cv2.VideoWriter(filename, fourcc, fps, (w, h))

        with _intercept_writer_lock:
            _intercept_writer = writer

        self._recording = True
        print(f"[REC] Started → {filename}  (capture FPS: {fps})")
        self.rec_btn.config(text="⏹ STOP", fg=DANGER)
        self.status_label.config(text="⬤  RECORDING", fg=DANGER)

    def _stop_recording_locked(self):
        global _intercept_writer
        self._recording = False
        with _intercept_writer_lock:
            if _intercept_writer is not None:
                _intercept_writer.release()
                _intercept_writer = None
        print("[REC] Stopped — file saved")
        self.rec_btn.config(text="⏺ REC", fg="#555")
        self.status_label.config(
            text="⬤  AIRBORNE" if self.airborne else "⬤  CONNECTED",
            fg=WARN if self.airborne else ACCENT
        )

    # ── Keyboard ──────────────────────────────────────────────────────
    def _on_key(self, event):
        k = event.keysym.lower()
        if k == "m":
            self._manual_cycle_model()
            return
        mapping = {
            "t": self.cmd_takeoff, "l": self.cmd_land,
            "w": self.cmd_forward, "s": self.cmd_backward,
            "a": self.cmd_left,    "d": self.cmd_right,
            "up": self.cmd_up,     "down": self.cmd_down,
            "q": self.cmd_rot_left,"e": self.cmd_rot_right,
            "space": self.cmd_hover, "escape": self.cmd_emergency,
            "r": self.toggle_recording,
        }
        fn = mapping.get(k)
        if fn: fn()

    # ── Shutdown ──────────────────────────────────────────────────────
    def on_close(self):
        self.running = False
        with self._record_lock:
            if self._recording:
                self._stop_recording_locked()
        cv2.imshow = _original_imshow
        if self.api:
            try: self.api.Plane_cmd_swith_rtp(1)
            except Exception: pass
        self.root.destroy()


def main():
    root = tk.Tk()
    app  = HulaDroneApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
    cv2.imshow = _original_imshow

if __name__ == "__main__":
    main()