"""
Model Condition Performance Analyser

Runs OBB and Pose models side-by-side on a video source, measures
per-frame environmental conditions, and produces charts and a CSV showing
which model performs better under each condition.

To run, follow examples:
    python condition_analysis.py --source hula_raw_20250330.avi
    python condition_analysis.py --source 0 --frames 300
    python condition_analysis.py --source drone --frames 400
"""

import argparse
import csv
import sys
import time
import threading
from pathlib import Path
 
import cv2
import numpy as np
from ultralytics import YOLO
 
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not found — install with: pip install matplotlib")
 
# ── Drone stream support (optional) ──────────────────────────────────
try:
    import av
    import pyhula
    HAS_PYHULA = True
except ImportError:
    HAS_PYHULA = False
 
 
# ── Styling ───────────────────────────────────────────────────────────
BG      = "#0a0a0f"
BG2     = "#0d0d14"
ACCENT  = "#00ffe0"
WARN    = "#ffcc00"
DANGER  = "#ff4455"
C_OBB   = "#00ffe0"
C_POSE  = "#ffcc00"
 
 
# ─────────────────────────────────────────────────────────────────────
# Frame-level condition measurements
# ─────────────────────────────────────────────────────────────────────
def measure_conditions(frame: np.ndarray) -> dict:
    """
    Returns a dict of scalar condition metrics for a single BGR frame.
 
    brightness  : mean pixel intensity (0–255). Low = dim, high = bright.
    blur        : Laplacian variance. Low = blurry, high = sharp.
    contrast    : std-dev of pixel intensities. Low = flat, high = contrasty.
    saturation  : mean HSV saturation. Low = grey/indoor, high = vivid/outdoor.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return {
        "brightness":  float(np.mean(gray)),
        "blur":        float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "contrast":    float(np.std(gray)),
        "saturation":  float(np.mean(hsv[:, :, 1])),
    }
 
 
# ─────────────────────────────────────────────────────────────────────
# Per-model inference — returns timing + confidence stats
# ─────────────────────────────────────────────────────────────────────
def run_model(model: YOLO, frame: np.ndarray, conf: float, imgsz: int) -> dict:
    t0        = time.perf_counter()
    confs     = []
    det_count = 0
    is_obb    = getattr(model, "task", "") == "obb"
    try:
        for result in model.predict(source=frame, verbose=False,
                                    conf=conf, imgsz=imgsz, stream=True):
            # OBB models store detections in result.obb, not result.boxes
            preds = result.obb if is_obb else result.boxes
            if preds is not None and len(preds):
                confs.extend(preds.conf.cpu().tolist())
                det_count += len(preds)
    except Exception as e:
        print(f"  [WARN] Inference error: {e}")
    elapsed_ms = (time.perf_counter() - t0) * 1000
 
    return {
        "infer_ms":   round(elapsed_ms, 1),
        "fps":        round(1000 / elapsed_ms, 1) if elapsed_ms > 0 else 0.0,
        "mean_conf":  round(float(np.mean(confs)), 4) if confs else 0.0,
        "max_conf":   round(float(np.max(confs)), 4)  if confs else 0.0,
        "detections": det_count,
    }
 
 
# ─────────────────────────────────────────────────────────────────────
# Video source helpers
# ─────────────────────────────────────────────────────────────────────
def open_source(source: str):
    """Returns an iterable that yields BGR numpy frames."""
    if source == "drone":
        return DroneFrameSource()
    try:
        idx = int(source)
        return CvSource(idx)
    except ValueError:
        return CvSource(source)
 
 
class CvSource:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open source: {src}")
            sys.exit(1)
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[INFO] Opened: {src}  frames={total}  fps={fps:.1f}")
 
    def __iter__(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame
 
    def release(self):
        self.cap.release()
 
 
class DroneFrameSource:
    """Grabs frames from Hula drone via pyhula + PyAV."""
    def __init__(self):
        if not HAS_PYHULA:
            print("[ERROR] pyhula/av not installed. "
                  "Install with: pip install pyhula av")
            sys.exit(1)
        self._frame  = None
        self._lock   = threading.Lock()
        self._stop   = False
        self._api    = pyhula.UserApi()
 
        print("[INFO] Connecting to drone...")
        if not self._api.connect():
            print("[ERROR] Could not connect to drone")
            sys.exit(1)
        print("[INFO] Connected")
 
        _orig = cv2.imshow
        def _intercept(w, img):
            if img is not None and isinstance(img, np.ndarray):
                with self._lock:
                    self._frame = img.copy()
        cv2.imshow = _intercept
 
        self._api.Plane_cmd_swith_rtp(0)
        time.sleep(1.0)
        threading.Thread(target=self._api.single_fly_flip_rtp,
                         daemon=True).start()
        cv2.imshow = _orig
 
        # Wait for first frame
        print("[INFO] Waiting for drone stream...")
        deadline = time.time() + 15
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None:
                    break
            time.sleep(0.1)
        else:
            print("[ERROR] No frames from drone after 15s")
            sys.exit(1)
        print("[INFO] Stream live")
 
    def __iter__(self):
        while not self._stop:
            with self._lock:
                f = self._frame.copy() if self._frame is not None else None
            if f is not None:
                yield f
            time.sleep(0.033)
 
    def release(self):
        self._stop = True
        try:
            self._api.Plane_cmd_swith_rtp(1)
        except Exception:
            pass
 
 
# ─────────────────────────────────────────────────────────────────────
# Binning helpers
# ─────────────────────────────────────────────────────────────────────
BINS = {
    "brightness": [
        (0,    80,  "dim"),
        (80,   150, "normal"),
        (150,  255, "bright"),
    ],
    "blur": [
        (0,    100,  "heavy_blur"),
        (100,  500,  "slight_blur"),
        (500,  1e9,  "sharp"),
    ],
    "contrast": [
        (0,    30,  "low_contrast"),
        (30,   60,  "mid_contrast"),
        (60,   1e9, "high_contrast"),
    ],
    "saturation": [
        (0,    40,  "low_sat"),
        (40,   100, "mid_sat"),
        (100,  255, "high_sat"),
    ],
}
 
def bin_value(metric: str, value: float) -> str:
    for lo, hi, label in BINS[metric]:
        if lo <= value < hi:
            return label
    return "unknown"
 
 
# ─────────────────────────────────────────────────────────────────────
# Analysis + reporting
# ─────────────────────────────────────────────────────────────────────
def analyse(rows: list[dict]) -> dict:
    """
    Group rows by each condition bin and compute mean metrics per model.
    Returns summary dict keyed by (metric, bin_label).
    """
    from collections import defaultdict
 
    groups = defaultdict(list)
    for r in rows:
        for metric in BINS:
            label = bin_value(metric, r[metric])
            groups[(metric, label)].append(r)
 
    summary = {}
    for (metric, label), group in groups.items():
        def avg(key):
            vals = [g[key] for g in group if g[key] is not None]
            return round(float(np.mean(vals)), 4) if vals else 0.0
 
        summary[(metric, label)] = {
            "n":              len(group),
            "avg_brightness": avg("brightness"),
            "avg_blur":       avg("blur"),
            "obb_mean_conf":  avg("obb_mean_conf"),
            "obb_fps":        avg("obb_fps"),
            "obb_dets":       avg("obb_detections"),
            "pose_mean_conf": avg("pose_mean_conf"),
            "pose_fps":       avg("pose_fps"),
            "pose_dets":      avg("pose_detections"),
            "winner_conf":    "obb"  if avg("obb_mean_conf")  >= avg("pose_mean_conf") else "pose",
            "winner_fps":     "obb"  if avg("obb_fps")        >= avg("pose_fps")       else "pose",
        }
    return summary
 
 
def save_csv(rows: list[dict], summary: dict, out_dir: Path):
    # Timestamped filenames so open files never block a re-run
    ts       = time.strftime("%Y%m%d_%H%M%S")
    raw_path = out_dir / f"raw_frames_{ts}.csv"
    if rows:
        with open(raw_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"[CSV]  Raw data   → {raw_path}  ({len(rows)} frames)")
 
    sum_path = out_dir / f"condition_summary_{ts}.csv"
    with open(sum_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition_metric", "bin", "n_frames",
                          "obb_conf", "obb_fps", "obb_dets",
                          "pose_conf", "pose_fps", "pose_dets",
                          "winner_conf", "winner_fps"])
        for (metric, label), s in sorted(summary.items()):
            writer.writerow([metric, label, s["n"],
                              s["obb_mean_conf"], s["obb_fps"], s["obb_dets"],
                              s["pose_mean_conf"], s["pose_fps"], s["pose_dets"],
                              s["winner_conf"], s["winner_fps"]])
    print(f"[CSV]  Summary     → {sum_path}")
 
 
def print_summary(summary: dict):
    metrics = list(BINS.keys())
    for metric in metrics:
        bins = [label for _, _, label in BINS[metric]]
        print(f"\n{'─'*72}")
        print(f"  {metric.upper()}")
        print(f"{'─'*72}")
        print(f"  {'BIN':<16} {'N':>5}  "
              f"{'OBB conf':>9}  {'OBB fps':>8}  "
              f"{'POSE conf':>9}  {'POSE fps':>8}  "
              f"{'WINNER(conf)':>13}  {'WINNER(fps)':>11}")
        print(f"  {'─'*16}  {'─'*5}  {'─'*9}  {'─'*8}  {'─'*9}  {'─'*8}  {'─'*13}  {'─'*11}")
        for label in bins:
            s = summary.get((metric, label))
            if s is None or s["n"] == 0:
                continue
            wc = "★ OBB" if s["winner_conf"] == "obb" else "  POSE"
            wf = "★ OBB" if s["winner_fps"]  == "obb" else "  POSE"
            print(f"  {label:<16} {s['n']:>5}  "
                  f"{s['obb_mean_conf']:>9.3f}  {s['obb_fps']:>8.1f}  "
                  f"{s['pose_mean_conf']:>9.3f}  {s['pose_fps']:>8.1f}  "
                  f"{wc:>13}  {wf:>11}")
 
 
def suggest_rules(summary: dict):
    """Print suggested switching rules based on findings."""
    print(f"\n{'═'*72}")
    print("  SUGGESTED AUTO-SWITCHING RULES")
    print(f"{'═'*72}")
 
    rules = []
    for metric in BINS:
        for _, _, label in BINS[metric]:
            s = summary.get((metric, label))
            if s is None or s["n"] < 10:
                continue
            diff_conf = s["obb_mean_conf"] - s["pose_mean_conf"]
            diff_fps  = s["obb_fps"]       - s["pose_fps"]
            winner    = s["winner_conf"]
            margin    = abs(diff_conf)
 
            # Only suggest if margin is meaningful
            if margin < 0.03:
                verdict = "  → similar performance, current model fine"
            else:
                verdict = (f"  → prefer {winner.upper()} "
                           f"(+{margin:.2f} conf, "
                           f"fps diff {diff_fps:+.1f})")
            rules.append(f"  {metric:<14} {label:<16}{verdict}")
 
    for r in rules:
        print(r)
 
    print(f"\n  Copy relevant thresholds into SWITCH_RULES in hula_ui.py")
    print(f"{'═'*72}\n")
 
 
# ─────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────
def make_charts(rows: list[dict], summary: dict, out_dir: Path):
    if not HAS_MPL or not rows:
        return
 
    _make_scatter_charts(rows, out_dir)
    _make_bar_charts(summary, out_dir)
    _make_fps_confidence_tradeoff(summary, out_dir)
 
 
def _styled_ax(ax, title):
    ax.set_facecolor(BG2)
    for s in ax.spines.values():
        s.set_edgecolor("#1e1e2e")
    ax.tick_params(colors="#aaaacc", labelsize=8)
    ax.grid(color="#1e1e2e", linewidth=0.5)
    ax.set_title(title, color=ACCENT, fontsize=10, fontweight="bold", pad=8)
    ax.xaxis.label.set_color("#aaaacc")
    ax.yaxis.label.set_color("#aaaacc")
 
 
def _make_scatter_charts(rows, out_dir):
    """Scatter: condition value vs confidence, coloured by model."""
    condition_metrics = list(BINS.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(BG)
    axes = axes.flatten()
 
    for ax, metric in zip(axes, condition_metrics):
        xs = [r[metric]         for r in rows]
        ys_obb  = [r["obb_mean_conf"]  for r in rows]
        ys_pose = [r["pose_mean_conf"] for r in rows]
 
        ax.scatter(xs, ys_obb,  c=C_OBB,  alpha=0.35, s=8,  label="OBB")
        ax.scatter(xs, ys_pose, c=C_POSE, alpha=0.35, s=8,  label="Pose")
 
        # Rolling trend lines
        if len(xs) > 20:
            order  = np.argsort(xs)
            xs_s   = np.array(xs)[order]
            for ys, c, lbl in [(ys_obb, C_OBB, "OBB trend"),
                                (ys_pose, C_POSE, "Pose trend")]:
                ys_s = np.array(ys)[order]
                # Smooth with uniform window
                w = max(1, len(xs_s) // 20)
                ys_smooth = np.convolve(ys_s, np.ones(w)/w, mode='valid')
                xs_smooth = xs_s[w//2: w//2 + len(ys_smooth)]
                ax.plot(xs_smooth, ys_smooth, color=c, linewidth=2,
                        label=lbl, alpha=0.9)
 
        _styled_ax(ax, f"{metric.capitalize()} vs Confidence")
        ax.set_xlabel(metric)
        ax.set_ylabel("Mean confidence")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, facecolor=BG2, labelcolor="#aaaacc",
                  edgecolor="#1e1e2e")
 
    fig.suptitle("Per-Frame Condition vs Confidence",
                 color=ACCENT, fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "scatter_condition_vs_confidence.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[CHART] Scatter → {path}")
 
 
def _make_bar_charts(summary, out_dir):
    """Grouped bar chart: mean confidence per bin per model."""
    metrics = list(BINS.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(BG)
    axes = axes.flatten()
 
    for ax, metric in zip(axes, metrics):
        bins   = [label for _, _, label in BINS[metric]]
        labels = [b for b in bins if summary.get((metric, b), {}).get("n", 0) > 0]
        if not labels:
            ax.set_visible(False)
            continue
 
        x      = np.arange(len(labels))
        w      = 0.35
        obb_c  = [summary[(metric, l)]["obb_mean_conf"]  for l in labels]
        pose_c = [summary[(metric, l)]["pose_mean_conf"] for l in labels]
 
        bars_obb  = ax.bar(x - w/2, obb_c,  w, color=C_OBB,  alpha=0.8, label="OBB")
        bars_pose = ax.bar(x + w/2, pose_c, w, color=C_POSE, alpha=0.8, label="Pose")
 
        for bar, val in list(zip(bars_obb, obb_c)) + list(zip(bars_pose, pose_c)):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom",
                        fontsize=7, color="#cccccc")
 
        _styled_ax(ax, f"{metric.capitalize()} — Mean Confidence by Bin")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Mean confidence")
        ax.legend(fontsize=8, facecolor=BG2, labelcolor="#aaaacc",
                  edgecolor="#1e1e2e")
 
        # N labels
        for i, l in enumerate(labels):
            n = summary[(metric, l)]["n"]
            ax.text(i, -0.08, f"n={n}", ha="center", va="top",
                    fontsize=6, color="#555577",
                    transform=ax.get_xaxis_transform())
 
    fig.suptitle("Mean Confidence per Condition Bin",
                 color=ACCENT, fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "bar_confidence_by_condition.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[CHART] Bar     → {path}")
 
 
def _make_fps_confidence_tradeoff(summary, out_dir):
    """
    Bubble chart: OBB vs Pose across all bins.
    X = mean confidence, Y = mean FPS, bubble size = n_frames.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(BG)
    _styled_ax(ax, "Confidence vs FPS Trade-off (all condition bins)")
 
    plotted = {"OBB": False, "Pose": False}
    for (metric, label), s in summary.items():
        if s["n"] < 5:
            continue
        size = max(20, s["n"] * 2)
        tag  = f"{metric[:3]}:{label}"
 
        ax.scatter(s["obb_mean_conf"], s["obb_fps"],
                   s=size, color=C_OBB, alpha=0.6, edgecolors="white", linewidths=0.5,
                   label="OBB" if not plotted["OBB"] else "_")
        ax.annotate(tag, (s["obb_mean_conf"], s["obb_fps"]),
                    fontsize=6, color=C_OBB, alpha=0.8,
                    xytext=(4, 4), textcoords="offset points")
 
        ax.scatter(s["pose_mean_conf"], s["pose_fps"],
                   s=size, color=C_POSE, alpha=0.6, edgecolors="white", linewidths=0.5,
                   label="Pose" if not plotted["Pose"] else "_")
        ax.annotate(tag, (s["pose_mean_conf"], s["pose_fps"]),
                    fontsize=6, color=C_POSE, alpha=0.8,
                    xytext=(4, -8), textcoords="offset points")
 
        plotted["OBB"] = plotted["Pose"] = True
 
    ax.set_xlabel("Mean confidence (higher = better detection)")
    ax.set_ylabel("Inference FPS (higher = faster)")
    ax.legend(fontsize=9, facecolor=BG2, labelcolor="#aaaacc", edgecolor="#1e1e2e")
    ax.text(0.01, 0.99,
            "Bubble size ∝ frame count  |  tag = condition bin",
            transform=ax.transAxes, fontsize=7, color="#555577",
            va="top")
 
    plt.tight_layout()
    path = out_dir / "tradeoff_fps_vs_confidence.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[CHART] Tradeoff → {path}")
 
 
# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Condition vs model performance analyser")
    p.add_argument("--source", default="0",
                   help="Video file, webcam index, or 'drone'")
    p.add_argument("--obb",    default=r"C:\Users\rizro\HulaDroneControlApp-desktop\runs\obb\train35\weights\best.pt",
                   help="OBB model weights")
    p.add_argument("--pose",   default=r"C:\Users\rizro\HulaDroneControlApp-desktop\runs\pose\train16\weights\best.pt",
                   help="Pose model weights")
    p.add_argument("--frames", type=int, default=500,
                   help="Max frames to analyse (default 500)")
    p.add_argument("--conf",   type=float, default=0.25,
                   help="Confidence threshold (default 0.25)")
    p.add_argument("--imgsz",  type=int, default=1280,
                   help="Inference image size (default 1280)")
    p.add_argument("--out",    default="condition_analysis",
                   help="Output folder (default condition_analysis/)")
    p.add_argument("--skip",   type=int, default=2,
                   help="Process every Nth frame (default 2, for speed)")
    return p.parse_args()
 
 
def main():
    args = parse_args()
 
    # Use video filename (without extension) as subfolder name so each
    # video gets its own tidy directory inside the main output folder.
    # e.g. condition_analysis/hula_indoor_dim/raw_frames.csv
    if args.source not in ("0", "drone") and Path(args.source).is_file():
        video_stem = Path(args.source).stem
    elif args.source == "drone":
        video_stem = f"drone_{time.strftime('%Y%m%d_%H%M%S')}"
    else:
        video_stem = f"webcam_{time.strftime('%Y%m%d_%H%M%S')}"
 
    out_dir = Path(args.out) / video_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output folder: {out_dir}")
 
    print(f"\n[INFO] Loading OBB model:  {args.obb}")
    obb_model = YOLO(args.obb)
    print(f"[INFO] Loading Pose model: {args.pose}")
    pose_model = YOLO(args.pose)
 
    source  = open_source(args.source)
 
    # ── Diagnostic: test both models on first frame before full run ───
    print("\n[DIAG] Running diagnostic on first frame...")
    diag_source = open_source(args.source)
    diag_frame  = None
    for f in diag_source:
        diag_frame = f
        break
    if hasattr(diag_source, "release"):
        diag_source.release()
 
    if diag_frame is not None:
        for label, mdl in [("OBB", obb_model), ("Pose", pose_model)]:
            print(f"\n  [{label}] model type : {type(mdl.model).__name__}")
            print(f"  [{label}] task       : {mdl.task}")
            print(f"  [{label}] conf used  : {args.conf}")
            is_obb_mdl = getattr(mdl, "task", "") == "obb"
            raw_results = list(mdl.predict(
                source=diag_frame, verbose=False,
                conf=0.01,           # very low — catch anything
                imgsz=args.imgsz, stream=True))
            for r in raw_results:
                preds = r.obb if is_obb_mdl else r.boxes
                if preds is not None and len(preds):
                    confs = preds.conf.cpu().tolist()
                    print(f"  [{label}] detections: {len(preds)}  "
                          f"confs: {[round(c,3) for c in confs[:5]]}")
                else:
                    print(f"  [{label}] detections: 0  (nothing found even at conf=0.01)")
                    print(f"  [{label}] obb attr   : {hasattr(r, 'obb')} "
                          f"  obb data: "
                          f"{r.obb.data.shape if hasattr(r,'obb') and r.obb is not None else 'None'}")
    print("\n[DIAG] Done — starting full analysis...\n")
 
    rows    = []
    n       = 0
    skipped = 0
 
    print(f"\n[INFO] Analysing up to {args.frames} frames "
          f"(every {args.skip} frame(s))...\n")
 
    try:
        for frame in source:
            skipped += 1
            if skipped % args.skip != 0:
                continue
 
            n += 1
            if n > args.frames:
                break
 
            if n % 50 == 0 or n == 1:
                print(f"  Frame {n}/{args.frames} ...")
 
            conds    = measure_conditions(frame)
            obb_res  = run_model(obb_model,  frame, args.conf, args.imgsz)
            pose_res = run_model(pose_model, frame, args.conf, args.imgsz)
 
            rows.append({
                "frame":            n,
                **conds,
                **{f"obb_{k}":  v for k, v in obb_res.items()},
                **{f"pose_{k}": v for k, v in pose_res.items()},
                "brightness_bin":  bin_value("brightness",  conds["brightness"]),
                "blur_bin":        bin_value("blur",        conds["blur"]),
                "contrast_bin":    bin_value("contrast",    conds["contrast"]),
                "saturation_bin":  bin_value("saturation",  conds["saturation"]),
            })
 
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted — analysing collected frames...")
    finally:
        if hasattr(source, "release"):
            source.release()
 
    if not rows:
        print("[ERROR] No frames collected.")
        return
 
    print(f"\n[INFO] Collected {len(rows)} frames. Running analysis...\n")
 
    summary = analyse(rows)
    print_summary(summary)
    suggest_rules(summary)
    save_csv(rows, summary, out_dir)
    make_charts(rows, summary, out_dir)
 
    print(f"\n[DONE] All outputs saved to: {out_dir}/\n")
 
 
if __name__ == "__main__":
    main()