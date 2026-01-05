from datetime import datetime
import cv2
import time
import subprocess
import sys
import numpy as np
from zoneinfo import ZoneInfo
import logging
import onnxruntime as ort
import threading
print ("STARTING script")
logging.basicConfig(level=logging.INFO, force=True)

# tuning variable for MEAN-GATE
SENSITIVITY_THRESHOLD = 1.0  ## 2.0 sensitive, 1.5 quite, 1.0 very, 0.5 extreme

CONF_THRESHOLD = 0.2  ## confidence level for YOLO detection 0.15

YOLO_MODEL_PATH = "/home/yourname/models/yolov5s.onnx" 

is_recording = False

# --- OPEN RTSP video CAPTURE ---
def open_capture():
    #logging.info("capture video using TAPO low-res stream")
    """Open RTSP stream with retry logic and minimized buffer."""

    RTSP_URL = "rtsp://....../stream2"   #use low res stream for motion detection

    while True:
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap

        print("Failed to open RTSP stream — retry in 2s")
        time.sleep(2)

# --- RTSP FRAME GRAB ---
def get_rtsp_frame(cap):
    # Warm up the RTSP stream
    for _ in range(5):
        ret, _ = cap.read()
        if not ret:
            continue
        time.sleep(0.03)  # 30 ms to let decoder stabilize

    # Now grab the real frame
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    return frame

# --- MEAN GATE ---
def mean_gate(frame, last_mean, gate_threshold):     
    tiny = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    tiny_gray = cv2.cvtColor(tiny, cv2.COLOR_BGR2GRAY)
    current_mean = float(tiny_gray.mean())
    
    
    if last_mean is None:
        return current_mean, True  # treat first frame as "motion"

    mean_delta = abs(current_mean - last_mean)
    motion = mean_delta >= gate_threshold
    # logging.info(f"mean-gate  {current_mean} {motion}")
    return current_mean, motion

# --- Letterbox ---- 
# --- resize frame from video feed (640x360) to 640x640 required by VOLOv8
def preprocess_letterbox(frame):
    """
    Letterbox a 640x360 frame into a 640x640 YOLO input tensor.
    Preserves aspect ratio and avoids distortion.
    """
    # logging.info(f"letterbox frame shape={frame.shape}, min={frame.min()}, max={frame.max()}")

    h, w = frame.shape[:2]   # h=360, w=640
    size = 640               # YOLO input size

    # Scale factor (fits longest side to 640)
    scale = size / max(h, w)     # scale = 640 / 640 = 1.0
    nh, nw = int(h * scale), int(w * scale)  # nh=360, nw=640

    # Resize without distortion
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # Create a 640x640 black canvas
    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    # Center vertically (top/bottom padding)
    top = (size - nh) // 2       # (640 - 360) // 2 = 140
    canvas[top:top+nh, :nw] = resized

    # Convert to YOLO format
    img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)   # Add batch dim

    return img.astype(np.float32)


# --- YOLO inference function (person detection) ----

def yolo_detect_person(frame, session, input_name, output_name, conf_threshold):
    global is_recording
    if is_recording:
        return False
        
    if session is None:
        logging.warning("YOLO session is None")
        return False

    # --- PREPROCESS ---
    img = preprocess_letterbox(frame)  # You can keep this exactly as-is

    # --- INFERENCE ---
    try:
        outputs = session.run([output_name], {input_name: img})[0]  # (1, 25200, 85)
    except Exception as e:
        logging.error(f"YOLO inference failed: {e}")
        return False

    preds = outputs[0]  # shape (25200, 85)

    # --- PARSE DETECTIONS ---
    for i, det in enumerate(preds[:50]):  # log first 50 rows max
        obj_conf = det[4]
        class_scores = det[5:]
        class_id = int(np.argmax(class_scores))
        class_conf = class_scores[class_id]

        if not is_recording:
            if class_conf > conf_threshold * 0.8:
                logging.info(f"confidence={class_conf:.3f}")  #log confidence for testing, elimiate in production.

        # Person = class 0
        if class_id == 0 and class_conf > conf_threshold:
            if not is_recording:
                logging.info("PERSON DETECTED")
            return True

    # logging.info("YOLO: no person detected")
    return False

# --- TIMESTAMPED FILENAME ---
def timestamped_filename():
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"/mnt/storage/videos/tapo/camera_{ts}.mp4"

# --- RECORD CLIP from high res TAPO stream1 ---
def record_clip():
    global is_recording
    is_recording = True

    RTSP_URL_stream1 = "rtsp://....../stream1"   # use camera high res stream
    OUTPUT_FILE = timestamped_filename()
    RECORD_SECONDS = 12

    # warmup removed to avoid second ffmpeg session

    # ---------------------------------------------------------
    # 2. Real FFmpeg recording
    # ---------------------------------------------------------
    cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-max_delay", "500000",
        "-loglevel", "warning",
        "-use_wallclock_as_timestamps", "1",
        "-fflags", "+genpts",
        "-i", RTSP_URL_stream1,
        "-t", str(RECORD_SECONDS),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "64k",
        "-movflags", "+faststart",
        OUTPUT_FILE
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Give FFmpeg a moment to fail fast
        time.sleep(0.3)

        if proc.poll() is not None:
            err = proc.stderr.read().decode()
            logging.error(f"FFmpeg startup error:\n{err}")
            is_recording = False
            return

        logging.info(f"Start recording")
        sys.stdout.flush()

        # ---------------------------------------------------------
        # 3. Wait with timeout protection
        # ---------------------------------------------------------
        try:
            proc.communicate(timeout=RECORD_SECONDS + 5)
        except subprocess.TimeoutExpired:
            proc.kill()
            logging.warning("FFmpeg timed out, killed process")

        # ---------------------------------------------------------
        # 4. Log result
        # ---------------------------------------------------------
        if proc.returncode == 0:
            logging.info(f"Saved video clip")
        else:
            logging.error(f"FFmpeg exited with code {proc.returncode}")

        sys.stdout.flush()

    except Exception as e:
        logging.error(f"Exception during recording: {e}")

    finally:
        is_recording = False
        
                
################  Before main()  ###################

try:
    session = ort.InferenceSession(
        YOLO_MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # logging.info("YOLO model loaded successfully.")
    yolo_available = True

except Exception as e:
    logging.error(f"Failed to load YOLO model: {e}")
    session = None
    yolo_available = False


################  main loop ###################

if __name__ == "__main__":
    cap = open_capture()
    last_mean = None
    cooldown_until = 0  # timestamp to prevent back-to-back recordings

    while True:

        # --- 1. Grab RTSP frame ---
        frame = get_rtsp_frame(cap)
        if frame is None:
            logging.warning("RTSP frame grab failed, retrying...")
            time.sleep(0.2)
            continue

        # --- FIRST FRAME INITIALIZATION ---
        if last_mean is None:
            last_mean = frame.mean()
            continue

        # --- 2. Mean-gate (cheap early exit) ---
        last_mean, motion = mean_gate(frame, last_mean, SENSITIVITY_THRESHOLD)
        if not motion:
            time.sleep(0.02)
            continue

        # --- 3. Cooldown check BEFORE YOLO ---
        now = time.time()
        if now < cooldown_until:
            continue

        # --- 4. YOLO person detection ---
        person = yolo_detect_person(frame, session, input_name, output_name, CONF_THRESHOLD)
        if not person:
            continue

        # --- 5. Record clip ---
        threading.Thread(target=record_clip, daemon=True).start()

        # --- 6. Set cooldown (cooldown for 20 seconds) ---
        cooldown_until = now + 20