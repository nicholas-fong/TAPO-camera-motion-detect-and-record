import cv2
import time
import datetime
import subprocess
import numpy as np
from zoneinfo import ZoneInfo

# --- CONFIG ---

# RTSP streams
rtsp_url_ffmpeg = "rtsp://nicholasfong:sau1kwan@192.168.1.96/stream1"  # high-res for recording
rtsp_url_cv     = "rtsp://nicholasfong:sau1kwan@192.168.1.96/stream2"  # low-res for motion detection

# Motion gate (efficient pre-filter)
MOTION_THRESHOLD    = 80   # pixel-count threshold on 640x360 diff. 200 OK to ignore cars, 80 for high sens but high CPU load
MEAN_SENSITIVITY    = 0.3   # motion gatekeeper: skip math if mean change < this. 0.2 OK to ignore cars
MOTION_FRAMES_START = 1     # how many frames above threshold to start recording
MOTION_FRAMES_STOP  = 15    # how many frames below threshold to consider motion ended 10 OK to ignore cars
MIN_RECORDING_SECONDS  = 15 # minimum recording time in seconds
FRAME_SKIP_INTERVAL = 1     # process every nth frame (TAPO ~30 FPS -> 15 FPS effective) (2 to lower CPU load)

# Person detection (YOLO ONNX)
YOLO_MODEL_PATH = "/home/nicholas/models/yolov8n.onnx"  # update to your actual path, use full path

# COCO person class id is 0
PERSON_CLASS_ID = 0

# YOLO thresholds
YOLO_CONFIDENCE_THRESH = 0.22  # confidence threshold for person OEM 0.4. TAPO 640x320 0.25 or 0.22
YOLO_IOU_NMS_THRESH    = 0.45 # IoU threshold for NMS (not strictly needed for single-person check)
MIN_PERSON_BBOX_AREA   = 1000 # ignore very small detections (tune as needed) OEM 3000, 1500 for TAPO 640x320. 1000 for smaller people
MIN_PERSON_ASPECT_RATIO = 0.45 # require tall-ish boxes: h/w >= this. OEM 1  try 0.5 as a test to detect person on TAPO 640x320
MIN_PERSON_HEIGHT = 45   # pixels in ORIGINAL frame; tune 40–80 depending on camera distance 60 or 45

# DNN throttling
PERSON_DETECT_EVERY_N = 1  # run DNN on every Nth motion frame (1 = every time) 2 or 3 to reduce CPU load
person_persistence = 0  # global or module-level variable

# --- STATE MACHINE STATES ---
STATE_IDLE      = 0  # not recording
STATE_RECORDING = 1  # recording while motion is present
STATE_COOLDOWN  = 2  # motion stopped, but continue until MIN_RECORDING_SECONDS
state = STATE_IDLE

# --- STATE MACHINE VARIABLES ---
last_motion          = None
last_mean            = None
motion_counter       = 0
no_motion_counter    = 0
recording            = None
recording_start_time = None
no_frame_counter     = 0
frame_count          = 0

# For DNN throttling
motion_frame_counter_for_dnn = 0

# --- LOAD YOLO MODEL ONCE ---
print("Loading YOLO ONNX model...")
dnn_available = False
net = None
try:
    net = cv2.dnn.readNetFromONNX(YOLO_MODEL_PATH)
    # Optional: use CPU explicitly
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    dnn_available = True
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    dnn_available = False
    net = None

# YOLO input size (depends on your model, common: 640x640)
YOLO_INPUT_WIDTH  = 640
YOLO_INPUT_HEIGHT = 640

# --- FUNCTIONS ----

def open_capture():
    """Open RTSP stream with retry logic and minimized buffer."""
    while True:
        cap = cv2.VideoCapture(rtsp_url_cv)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        print("Failed to open RTSP stream — retry in 2s")
        time.sleep(2)


def start_recording():
    """Launch ffmpeg and return the process handle and start time."""
    ts = datetime.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d_%H%M%S")
    filename = f"/mnt/storage/videos/tapo/cam1_{ts}.mp4"

    try:
        proc = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats",
            "-rtsp_transport", "tcp", "-rtsp_flags", "prefer_tcp",
            "-timeout", "4000000", "-fflags", "+genpts",
            "-use_wallclock_as_timestamps", "1", "-i", rtsp_url_ffmpeg,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "32k",
            "-reset_timestamps", "1", filename
        ])
        return proc, time.monotonic()
    except Exception as e:
        print(f"Failed to start ffmpeg: {e}")
        return None, None


def stop_recording():
    """Stop ffmpeg safely and reset recording state."""
    global recording, recording_start_time
    if recording is not None:
        try:
            recording.terminate()
            recording.wait(timeout=1)
        except Exception:
            recording.kill()
    print(f"Motion gone — recording stopped")
    recording = None
    recording_start_time = None


def update_state_machine():
    """
    Clean, explicit state machine for:
      - IDLE → waiting for person-gated motion
      - RECORDING → ffmpeg running while motion persists
      - COOLDOWN → motion ended, but minimum recording time not reached
    """

    global state, recording, recording_start_time
    global motion_counter, no_motion_counter

    # ------------------------------------------------------------
    # STATE: IDLE
    # ------------------------------------------------------------
    if state == STATE_IDLE:

        # Condition to start recording:
        # Enough consecutive person-gated motion frames
        if motion_counter >= MOTION_FRAMES_START and recording is None:
            print("Motion (person) detected — starting recording")

            recording, recording_start_time = start_recording()

            if recording is None:
                print("Failed to start ffmpeg — staying in IDLE")
                return

            state = STATE_RECORDING
            return

        # Otherwise remain idle
        return

    # ------------------------------------------------------------
    # STATE: RECORDING
    # ------------------------------------------------------------
    if state == STATE_RECORDING:

        # If ffmpeg died → stop and go idle
        if recording is None or recording.poll() is not None:
            print("ffmpeg died unexpectedly — stopping recording")
            stop_recording()
            state = STATE_IDLE
            return

        # If motion has stopped long enough → enter cooldown
        if no_motion_counter >= MOTION_FRAMES_STOP:
            print("Motion ended — entering COOLDOWN")
            state = STATE_COOLDOWN
            return

        # Otherwise remain in RECORDING
        return

    # ------------------------------------------------------------
    # STATE: COOLDOWN
    # ------------------------------------------------------------
    if state == STATE_COOLDOWN:

        # If ffmpeg died → stop and go idle
        if recording is None or recording.poll() is not None:
            print("ffmpeg died during cooldown — stopping")
            stop_recording()
            state = STATE_IDLE
            return

        # If motion returns → go back to RECORDING
        if motion_counter >= MOTION_FRAMES_START:
            print("Motion (person) returned — back to RECORDING")
            state = STATE_RECORDING
            return

        # If cooldown time satisfied → stop recording
        if recording_start_time is not None:
            elapsed = time.monotonic() - recording_start_time
            if elapsed >= MIN_RECORDING_SECONDS:
                print("Cooldown complete — stopping recording")
                stop_recording()
                state = STATE_IDLE
                return

        # Otherwise remain in COOLDOWN
        return

# --- PERSON DETECTION PERSISTENCE ---
person_persistence = 0

def detect_person_yolo(frame):
    """
    Run YOLOv8 ONNX on the frame and return True if a person is detected.
    Uses:
      - best-detection-per-frame logic
      - bounding box filters (area, aspect ratio, height)
      - persistence smoothing to avoid flicker
    """
    global person_persistence

    if not dnn_available or net is None:
        return False

    try:
        h, w = frame.shape[:2]

        # --- Letterbox resize to 640x640 ---
        scale = min(YOLO_INPUT_WIDTH / w, YOLO_INPUT_HEIGHT / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h))
        canvas = np.full((YOLO_INPUT_HEIGHT, YOLO_INPUT_WIDTH, 3), 114, dtype=np.uint8)

        dw = (YOLO_INPUT_WIDTH - new_w) // 2
        dh = (YOLO_INPUT_HEIGHT - new_h) // 2
        canvas[dh:dh + new_h, dw:dw + new_w] = resized

        # --- Preprocess for YOLO ---
        blob = cv2.dnn.blobFromImage(
            canvas,
            scalefactor=1/255.0,
            size=(YOLO_INPUT_WIDTH, YOLO_INPUT_HEIGHT),
            swapRB=True,
            crop=False
        )
        net.setInput(blob)
        outputs = net.forward()  # expected (1, 84, N) or (84, N)

        # --- Normalize output to (N, 84) ---
        outputs = np.squeeze(outputs)
        if outputs.shape[0] == 84:
            outputs = outputs.T  # (N, 84)

        # --- Track best detection this frame ---
        best_conf = 0
        best_box_h = 0

        # --- Parse detections ---
        for det in outputs:
            cx, cy, bw, bh = det[0:4]
            cls_scores = det[4:]

            class_id = int(np.argmax(cls_scores))
            conf = float(cls_scores[class_id])

            if class_id != PERSON_CLASS_ID:
                continue
            if conf < YOLO_CONFIDENCE_THRESH:
                continue

            # Convert to corners in letterboxed space
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

            # Undo letterbox
            x1 = (x1 - dw) / scale
            y1 = (y1 - dh) / scale
            x2 = (x2 - dw) / scale
            y2 = (y2 - dh) / scale

            # Clamp to frame
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            box_w = max(0, x2 - x1)
            box_h = max(0, y2 - y1)
            area = box_w * box_h
            aspect_ratio = box_h / box_w if box_w > 0 else 0

            # --- Filters tuned for TAPO low-res stream ---
            if area < MIN_PERSON_BBOX_AREA:
                continue
            if aspect_ratio < MIN_PERSON_ASPECT_RATIO:
                continue
            if box_h < MIN_PERSON_HEIGHT:   # <-- NEW: snow/rain/IR immunity
                continue

            # Keep the strongest detection
            if conf > best_conf:
                best_conf = conf
                best_box_h = box_h

        # --- Decide if a person was detected this frame ---
        person_detected_this_frame = (best_conf >= YOLO_CONFIDENCE_THRESH)

        # --- Persistence smoothing ---
        if person_detected_this_frame:
            person_persistence = min(person_persistence + 1, 5)
        else:
            person_persistence = max(person_persistence - 1, 0)

        # --- Final decision ---
        return person_persistence >= 1   # 1 = fast response, stable with persistence smoothing

    except Exception as e:
        print(f"Error during YOLO person detection: {e}")
        return False


# --- MAIN LOOP RUNS FOREVER ---

capture = open_capture()

while True:
    frame_count += 1

    # --- HANDLE FRAME FAILURES ---
    ret_grab = capture.grab()

    if not ret_grab:
        no_frame_counter += 1
        if no_frame_counter < 10:
            time.sleep(0.1)
            update_state_machine()
            continue

        print("Camera offline — reconnect using RTSP")
        no_frame_counter = 0
        try:
            capture.release()
        except Exception:
            pass

        capture = open_capture()
        last_motion = None
        last_mean = None
        # Treat as no motion
        motion_counter = 0
        no_motion_counter += 1
        update_state_machine()
        continue

    # --- FRAME SKIP LOGIC ---
    if frame_count % FRAME_SKIP_INTERVAL != 0:
        update_state_machine()
        continue

    ret, frame = capture.retrieve()
    if not ret:
        update_state_machine()
        continue

    no_frame_counter = 0
    if frame_count > 10000:
        frame_count = 0

    # --- 1. NUCLEAR OPTION (Mean Check) ---
    tiny_gray = cv2.cvtColor(cv2.resize(frame, (80, 45)), cv2.COLOR_BGR2GRAY)
    current_mean = tiny_gray.mean()

    motion_pixels = 0  # default when we skip heavy CV

    if last_mean is not None:
        if abs(current_mean - last_mean) < MEAN_SENSITIVITY:
            # Mean barely changed → treat as no motion, skip heavy work
            last_mean = current_mean
            no_motion_counter += 1
            motion_counter = 0

            update_state_machine()
            time.sleep(0.02)
            continue

    last_mean = current_mean

    # --- 2. FULL MOTION DETECTION (only if mean-gate says "something changed") ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)

    if last_motion is None:
        last_motion = blur
        update_state_machine()
        time.sleep(0.02)
        continue

    diff = cv2.absdiff(last_motion, blur)
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    motion_pixels = cv2.countNonZero(thresh)

    # --- 3. PERSON-GATED MOTION LOGIC (with YOLO) ---

    if motion_pixels > MOTION_THRESHOLD:
        # We have significant motion → consider running YOLO
        motion_frame_counter_for_dnn += 1

        run_dnn_this_frame = (
            PERSON_DETECT_EVERY_N <= 1 or
            (motion_frame_counter_for_dnn % PERSON_DETECT_EVERY_N == 0)
        )

        if run_dnn_this_frame:
            person_present = detect_person_yolo(frame)
        else:
            # If skipping DNN this frame, assume last known state
            person_present = (person_persistence >= 2)

        if person_present:
            # Person confirmed → this is real motion
            motion_counter += 1
            no_motion_counter = 0
        else:
            # Motion but no person → ignore (cars, shadows, pets, etc.)
            no_motion_counter += 1
            motion_counter = 0

    else:
        # Not enough motion to consider running YOLO
        no_motion_counter += 1
        motion_counter = 0
        motion_frame_counter_for_dnn = 0  # reset DNN throttle

        # --- UPDATE RECORDING STATE MACHINE ---
        update_state_machine()

        # --- UPDATE BASELINE ONLY AFTER SEVERAL STABLE FRAMES ---
        if no_motion_counter >= 3:
            last_motion = blur

        # Small sleep to reduce CPU load
        time.sleep(0.02)
