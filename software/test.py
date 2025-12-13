#!/usr/bin/env python3
import os
import csv
import time
import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
import requests
from ultralytics import YOLO

print("===================================================")
print("   ONNX CNN + HYBRID LOCAL/REMOTE YOLO PIPELINE    ")
print("      (Jetson + Windows + A6000, timed logging)    ")
print("===================================================\n")

# ============================================================
# CONFIG
# ============================================================
JPEG_DIR      = "/frames"
CNN_ONNX_PATH = "simplecnn_scene_selector.onnx"

# Windows relay (Jetson sees Windows at this IP)
RELAY_BASE_URL        = "http://IP:PORT" #TODO change IP and PORT 
YOLO_REMOTE_INFER_URL = f"{RELAY_BASE_URL}/infer"
RELAY_PING_URL        = f"{RELAY_BASE_URL}/ping"

# Local YOLO model (11n) on Jetson
YOLO_LOCAL_MODEL_PATH = "yolo11n.pt"

OUTPUT_CSV    = "result.csv"

# Decision parameters (real deployment budget)
LATENCY_BUDGET_MS     = 90.0   # total allowed for (cloud model + Jetson<->Windows RTT)
CLOUD_PROBE_INTERVAL  = 10     # frames between probes when cloud is "too slow"

# ============================================================
# STATE FOR DECISIONS / STATS
# ============================================================
last_cloud_model_ms   = None   # last YOLO11x model time on A6000 (ms)
last_cloud_total_ms   = None   # last measured total cloud RTT (ms)
last_cloud_bridge_ms  = None   # last measured bridge-only time (ms)

frames_since_cloud    = 0      # frames since last cloud attempt

# bridge stats (Windows <-> A6000 + overhead)
bridge_ms_sum         = 0.0
bridge_ms_count       = 0

# latest RTT (Jetson <-> Windows) used for decisions
last_rtt_ms           = None

# RTT stats (for summary)
rtt_ms_sum            = 0.0
rtt_ms_count          = 0
rtt_ms_min            = None
rtt_ms_max            = None

# ============================================================
# SAFE IMAGE CHECK
# ============================================================
def safe_imread(path):
    try:
        img = cv2.imread(path)
        if img is None:
            raise ValueError("cv2 returned None (image unreadable)")
        return img
    except Exception as e:
        print(f"[ERROR] BAD IMAGE — Skipping {path}: {e}")
        return None

# ============================================================
# ONNX CNN: load session
# ============================================================
print(f"[DEBUG] Loading ONNX CNN from: {CNN_ONNX_PATH}")
cnn_session = ort.InferenceSession(
    CNN_ONNX_PATH,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
print("[DEBUG] ONNX CNN loaded.\n")

def preprocess_for_cnn(jpeg_path):
    img = Image.open(jpeg_path).convert("RGB")
    img = img.resize((128, 128))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)   # (1, 3, 128, 128)
    return arr

def run_cnn_onnx(jpeg_path):
    try:
        inp = preprocess_for_cnn(jpeg_path)
        logits = cnn_session.run(None, {"input": inp})[0]  # [1, 2]
        pred = int(np.argmax(logits, axis=1)[0])
        return pred
    except Exception as e:
        print(f"[ERROR] CNN ONNX failed on {jpeg_path}: {e}")
        return None

# ============================================================
# LOCAL YOLO11n (Jetson)
# ============================================================
yolo_local = None

def load_local_yolo():
    global yolo_local
    if yolo_local is None:
        print("[DEBUG] Loading local YOLO11n on Jetson…")
        t0 = time.time()
        yolo_local = YOLO(YOLO_LOCAL_MODEL_PATH)
        print(f"[DEBUG] Local YOLO11n loaded in {time.time() - t0:.2f}s\n")
    return yolo_local

def run_local_yolo(jpeg_path):
    """
    Run YOLO11n locally on Jetson (fallback / edge).
    Returns:
      (num_objects, num_vehicles, num_pedestrians, yolo_time_sec)
    """
    if safe_imread(jpeg_path) is None:
        print(f"[WARN] Local YOLO skipped (bad image): {jpeg_path}")
        return 0, 0, 0, 0.0

    try:
        model = load_local_yolo()
        print(f"[DEBUG] Running LOCAL YOLO11n on: {jpeg_path}")

        t0 = time.time()
        # device=0 -> Jetson GPU; use 'cpu' if needed
        results = model(jpeg_path, device=0, verbose=False)[0]
        elapsed = time.time() - t0

        num_objects = len(results.boxes)
        num_vehicles = 0
        num_pedestrians = 0

        for box in results.boxes:
            cls = int(box.cls.cpu().item())
            name = results.names[cls].lower()
            if "car" in name or "truck" in name or "bus" in name:
                num_vehicles += 1
            if "person" in name:
                num_pedestrians += 1

        print(
            f"[DEBUG] LOCAL YOLO11n: objects={num_objects}, "
            f"vehicles={num_vehicles}, pedestrians={num_pedestrians}, "
            f"time={elapsed:.4f}s"
        )

        return num_objects, num_vehicles, num_pedestrians, elapsed

    except Exception as e:
        print(f"[ERROR] Local YOLO11n failed for {jpeg_path}: {e}")
        return 0, 0, 0, 0.0

# ============================================================
# NETWORK AVAILABILITY & RTT (Jetson -> Windows relay)
# ============================================================
def check_network_available(timeout=0.3):
    """
    Returns:
      (available: bool, ping_ms: float or None)
    where ping_ms is the Jetson↔Windows RTT for /ping.
    """
    try:
        t0 = time.time()
        r = requests.get(RELAY_PING_URL, timeout=timeout)
        elapsed_ms = (time.time() - t0) * 1000.0
        if r.status_code == 200:
            return True, elapsed_ms
        else:
            return False, None
    except requests.RequestException:
        return False, None

# ============================================================
# REMOTE YOLO (Jetson -> Windows relay -> A6000)
# ============================================================
def call_remote_yolo(jpeg_path, timeout=0.5):
    """
    Try to run YOLO11x on A6000 via Windows relay.

    Returns:
      (num_objects, num_vehicles, num_pedestrians,
       yolo_time_sec, total_roundtrip_sec, success_flag)
    """
    if safe_imread(jpeg_path) is None:
        print(f"[WARN] Remote YOLO skipped (bad image): {jpeg_path}")
        return 0, 0, 0, 0.0, 0.0, False

    try:
        with open(jpeg_path, "rb") as f:
            files = {"file": (os.path.basename(jpeg_path), f, "image/jpeg")}
            t0 = time.time()
            r = requests.post(YOLO_REMOTE_INFER_URL, files=files, timeout=timeout)
            total = time.time() - t0

        r.raise_for_status()
        data = r.json()

        num_objects     = data.get("num_objects", 0)
        num_vehicles    = data.get("num_vehicles", 0)
        num_pedestrians = data.get("num_pedestrians", 0)
        yolo_time_sec   = data.get("yolo_time_sec", total)

        print(
            f"[DEBUG] REMOTE YOLO11x: objects={num_objects}, "
            f"vehicles={num_vehicles}, pedestrians={num_pedestrians}, "
            f"yolo_time={yolo_time_sec:.4f}s, roundtrip={total:.4f}s"
        )

        return num_objects, num_vehicles, num_pedestrians, yolo_time_sec, total, True

    except Exception as e:
        print(f"[ERROR] Remote YOLO11x failed for {jpeg_path}: {e}")
        return 0, 0, 0, 0.0, 0.0, False

# ============================================================
# MAIN LOOP
# ============================================================
print(f"[DEBUG] Scanning JPEG directory: {JPEG_DIR}")

jpg_files = []
for root, dirs, files in os.walk(JPEG_DIR):
    for f in files:
        if f.lower().endswith(".jpg"):
            jpg_files.append(os.path.join(root, f))

jpg_files.sort()
print(f"[DEBUG] Found {len(jpg_files)} JPEG images.\n")

rows = []

for idx, jpeg_path in enumerate(jpg_files, 1):
    jpeg_name = os.path.basename(jpeg_path)
    print("\n===================================================")
    print(f"[INFO] Processing {idx}/{len(jpg_files)} → {jpeg_name}")

   # --------------------------------------------------------
    # 1) Check network availability and measure RTT
    # --------------------------------------------------------
    network_available, ping_ms = check_network_available()
    print(f"[DEBUG] Network available: {network_available}")

    if network_available and ping_ms is not None:
        last_rtt_ms = ping_ms  # update most recent RTT used for decisions

        # update RTT stats
        rtt_ms_sum   += ping_ms
        rtt_ms_count += 1
        if rtt_ms_min is None or ping_ms < rtt_ms_min:
            rtt_ms_min = ping_ms
        if rtt_ms_max is None or ping_ms > rtt_ms_max:
            rtt_ms_max = ping_ms

        print(f"[DEBUG] Jetson<->Windows RTT this frame: {ping_ms:.2f} ms")
    else:
        print("[DEBUG] Network ping failed or no RTT measurement.")

    # For CSV logging
    ping_ms_inst_for_csv = ping_ms if ping_ms is not None else -1.0
    ping_ms_last_for_csv = last_rtt_ms if last_rtt_ms is not None else -1.0

    # --------------------------------------------------------
    # 2) Decide if cloud is allowed this frame (using latest RTT & latest model time)
    # --------------------------------------------------------
    force_local      = False
    allow_cloud      = False
    est_total_ms_dec = -1.0  # for CSV logging (what we compared to 90 ms)

    if not network_available:
        print("[DEBUG] Network down → FORCE LOCAL YOLO11n.")
        force_local = True
    else:
        # Network is up
        if last_cloud_model_ms is None or last_rtt_ms is None:
            # No model timing or RTT stats yet → allow cloud to collect stats
            print("[DEBUG] No previous cloud model time or RTT → allow cloud to measure.")
            allow_cloud = True
        else:
            # Use latest YOLO11x model time + last RTT (no averaging)
            est_total_ms_dec = last_cloud_model_ms + last_rtt_ms
            print(
                f"[DEBUG] Last cloud model={last_cloud_model_ms:.2f} ms, "
                f"last RTT={last_rtt_ms:.2f} ms → est_total={est_total_ms_dec:.2f} ms"
            )

            if est_total_ms_dec < LATENCY_BUDGET_MS:
                print(
                    f"[DEBUG] est_total={est_total_ms_dec:.2f} ms < "
                    f"{LATENCY_BUDGET_MS:.2f} ms (budget) → allow cloud."
                )
                allow_cloud = True
            elif frames_since_cloud >= CLOUD_PROBE_INTERVAL:
                print(
                    f"[DEBUG] est_total={est_total_ms_dec:.2f} ms ≥ {LATENCY_BUDGET_MS:.2f} ms "
                    f"but frames_since_cloud={frames_since_cloud} ≥ {CLOUD_PROBE_INTERVAL} → PROBE CLOUD."
                )
                allow_cloud = True
            else:
                print(
                    f"[DEBUG] est_total={est_total_ms_dec:.2f} ms ≥ {LATENCY_BUDGET_MS:.2f} ms "
                    f"and frames_since_cloud={frames_since_cloud} < {CLOUD_PROBE_INTERVAL} → FORCE LOCAL."
                )
                force_local = True

    # --------------------------------------------------------
    # 3) CNN decision (if not forced local)
    # --------------------------------------------------------
    cnn_result = None
    if not force_local:
        cnn_result = run_cnn_onnx(jpeg_path)
        if cnn_result is None:
            print("[WARN] CNN failure — falling back to LOCAL YOLO11n.")
            force_local = True

    # --------------------------------------------------------
    # 4) Final decision: cloud vs local
    # --------------------------------------------------------
    if force_local:
        print("[DECISION] Using LOCAL YOLO11n (override).")
        model_choice = "local_11n_override"
        num_objects, num_vehicles, num_pedestrians, yolo_time_sec = run_local_yolo(jpeg_path)

    else:
        # CNN result is valid
        if cnn_result == 1 and allow_cloud:
            # CNN says "cloud" and cloud is allowed for this frame
            print("[DECISION] CNN → CLOUD path (remote YOLO11x with fallback).")
            (num_objects,
             num_vehicles,
             num_pedestrians,
             yolo_time_sec,
             total_roundtrip_sec,
             success) = call_remote_yolo(jpeg_path, timeout=0.5) #TODO 
            if success:
                model_choice = "remote_11x"

                model_ms  = yolo_time_sec * 1000.0
                total_ms  = total_roundtrip_sec * 1000.0
                bridge_ms = max(total_ms - model_ms, 0.0)

                last_cloud_model_ms  = model_ms
                last_cloud_total_ms  = total_ms
                last_cloud_bridge_ms = bridge_ms

                frames_since_cloud = 0

                bridge_ms_sum   += bridge_ms
                bridge_ms_count += 1

                print(
                    f"[DEBUG] Cloud split: model={model_ms:.2f} ms, "
                    f"bridge={bridge_ms:.2f} ms, total={total_ms:.2f} ms"
                )
                print(f"[DEBUG] Updated last_cloud_model_ms = {last_cloud_model_ms:.2f} ms")
            else:
                print("[WARN] Remote YOLO failed or timed out → FALLBACK to LOCAL YOLO11n.")
                model_choice = "local_11n_fallback"
                num_objects, num_vehicles, num_pedestrians, yolo_time_sec = run_local_yolo(jpeg_path)
                frames_since_cloud = 0  # just attempted cloud

        else:
            # Either CNN said local, or cloud not allowed for this frame
            if cnn_result == 1 and not allow_cloud:
                print("[DECISION] CNN suggested CLOUD but cloud not allowed → LOCAL YOLO11n.")
            else:
                print("[DECISION] CNN → LOCAL path (YOLO11n on Jetson).")
            model_choice = "local_11n"
            num_objects, num_vehicles, num_pedestrians, yolo_time_sec = run_local_yolo(jpeg_path)

    print(
        f"[DEBUG] FINAL OUTPUT — model={model_choice}, "
        f"objects={num_objects}, vehicles={num_vehicles}, "
        f"pedestrians={num_pedestrians}, inference_time={yolo_time_sec:.4f}s"
    )

    # --------------------------------------------------------
    # 5) CSV row logging (include all timing components)
    # --------------------------------------------------------
    row_last_cloud_model_ms  = last_cloud_model_ms  if last_cloud_model_ms  is not None else -1.0
    row_last_cloud_total_ms  = last_cloud_total_ms  if last_cloud_total_ms  is not None else -1.0
    row_last_cloud_bridge_ms = last_cloud_bridge_ms if last_cloud_bridge_ms is not None else -1.0

    rows.append([
        jpeg_name,                         # 0
        cnn_result if cnn_result is not None else -1,  # 1
        model_choice,                      # 2
        num_objects,                       # 3
        num_vehicles,                      # 4
        num_pedestrians,                   # 5
        yolo_time_sec,                     # 6 (inference_time_sec of chosen model)
        network_available,                 # 7
        ping_ms_inst_for_csv,              # 8  (instant Jetson<->Windows RTT)
        ping_ms_last_for_csv,              # 9  (last RTT used in decisions)
        row_last_cloud_model_ms,           # 10 (last YOLO11x model_ms)
        row_last_cloud_total_ms,           # 11 (last total cloud RTT in sim)
        row_last_cloud_bridge_ms,          # 12 (last bridge_ms = total - model)
        est_total_ms_dec,                  # 13 (model_ms + RTT used for 90ms decision)
        frames_since_cloud,                # 14
    ])

    # Increment frames_since_cloud only if we did NOT use remote_*
    if not model_choice.startswith("remote_"):
        frames_since_cloud += 1

# ============================================================
# SAVE CSV
# ============================================================
print(f"\n[DEBUG] Saving CSV: {OUTPUT_CSV}")

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "jpeg_name",                # 0
        "cnn_result",               # 1
        "model_choice",             # 2
        "num_objects",              # 3
        "num_vehicles",             # 4
        "num_pedestrians",          # 5
        "inference_time_sec",       # 6
        "network_available",        # 7
        "ping_ms_instant",          # 8  Jetson<->Windows RTT this frame
        "ping_ms_last",             # 9  last Jetson<->Windows RTT
        "last_cloud_model_ms",      # 10 last YOLO11x model time (ms)
        "last_cloud_total_ms",      # 11 last total RTT measured (ms)
        "last_cloud_bridge_ms",     # 12 last bridge-only time (ms)
        "decision_est_total_ms",    # 13 model_ms + RTT used for 90ms decision
        "frames_since_cloud",       # 14
    ])
    writer.writerows(rows)

# ============================================================
# BRIDGE / CLOUD / RTT STATS
# ============================================================
if bridge_ms_count > 0:
    avg_bridge_ms = bridge_ms_sum / bridge_ms_count
    print("\n[STATS] Cloud bridge overhead (Windows↔A6000 + overhead):")
    print(f"        calls: {bridge_ms_count}")
    print(f"        avg bridge-only latency: {avg_bridge_ms:.2f} ms")
else:
    print("\n[STATS] No successful cloud calls → cannot compute bridge overhead yet.")

if rtt_ms_count > 0:
    avg_rtt_ms = rtt_ms_sum / rtt_ms_count
    print("\n[STATS] Jetson ↔ Windows RTT (based on ping_ms_instant):")
    print(f"        samples: {rtt_ms_count}")
    print(f"        avg RTT: {avg_rtt_ms:.2f} ms")
    print(f"        min RTT: {rtt_ms_min:.2f} ms")
    print(f"        max RTT: {rtt_ms_max:.2f} ms")
else:
    print("\n[STATS] No Jetson ↔ Windows RTT samples collected.")

print("\n===================================================")
print("   COMPLETED HYBRID LOCAL/REMOTE YOLO PIPELINE     ")
print(f"             Results saved → {OUTPUT_CSV}")
print("===================================================\n")
