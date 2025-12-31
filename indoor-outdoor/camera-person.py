import cv2
import time
import datetime
import subprocess
import numpy as np
from zoneinfo import ZoneInfo

# --- CONFIG ---

# RTSP streams
rtsp_url_ffmpeg = "rtsp://username:password@<ip_address>/stream1"  # high-res for recording
rtsp_url_cv     = "rtsp://username:password@<ip_address>/stream2"  # low-res for motion detection

# Motion gate (efficient pre-filter)
MOTION_THRESHOLD       = 200      # pixel-count threshold on 640x360 diff
MEAN_SENSITIVITY       = 0.2      # "Nuclear" motion gatekeeper: skip math if mean change < this
MOTION_FRAMES_START    = 1        # how many frames above threshold to start recording
MOTION_FRAMES_STOP     = 10       # how many frames below threshold to consider motion ended
MIN_RECORDING_SECONDS  = 15       # minimum recording time in seconds
FRAME_SKIP_INTERVAL    = 2        # process every nth frame (TAPO ~30 FPS -> 15 FPS effective)

# Person detection (MobileNet-SSD Caffe)

PROTOTXT_PATH = "~/models/mobilenet_ssd/MobileNetSSD_deploy.prototxt"
CAFFEMODEL_PATH = "~/models/mobilenet_ssd/MobileNetSSD_deploy.caffemodel"

PERSON_CLASS_ID        = 15       # Standard MobileNet-SSD class id for 'person'
DNN_CONFIDENCE_THRESH  = 0.3      # low-res video use 0.3, high-res video use 0.5
MIN_PERSON_BBOX_AREA   = 1500     # ignore very small detections (tune if needed) 1000 for very sensitive
PERSON_DETECT_EVERY_N  = 1        # run DNN on every Nth motion frame (1 = every time)

# --- STATE MACHINE STATES ---
STATE_IDLE      = 0  # not recording
STATE_RECORDING = 1  # recording while motion is present
STATE_COOLDOWN  = 2  # motion stopped, but continue until MIN_RECORDING_SECONDS

state = STATE_IDLE

# --- STATE VARIABLES ---
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

# --- LOAD DNN MODEL ONCE ---

print("Loading MobileNet-SSD model...")
try:
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
    dnn_available = True
    print("MobileNet-SSD loaded successfully.")
except Exception as e:
    print(f"Failed to load MobileNet-SSD model: {e}")
    net = None
    dnn_available = False


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
    ts = datetime.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d_%H%M%S")  # modify your time zone
    filename = f"~/tapo/cam1_{ts}.mp4"  # you decide where to store the video clips

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
    Drive the recording state machine based on current global state:
    - state
    - motion_counter
    - no_motion_counter
    - recording, recording_start_time
    """
    global state, recording, recording_start_time

    # --- IDLE: wait for motion to start recording ---
    if state == STATE_IDLE:
        if motion_counter >= MOTION_FRAMES_START and recording is None:
            print(f"Motion (person) detected — start recording")
            recording, recording_start_time = start_recording()

            if recording is None:
                print("Failed to start ffmpeg — continue IDLE")
                state = STATE_IDLE
            else:
                state = STATE_RECORDING

        return

    # --- RECORDING: ffmpeg is running, motion ongoing ---
    if state == STATE_RECORDING:
        # If ffmpeg died unexpectedly, clean up and go idle
        if recording is None or recording.poll() is not None:
            print("ffmpeg died unexpectedly — returning to IDLE")
            stop_recording()
            state = STATE_IDLE
            return

        # If motion has stopped long enough, enter cooldown
        if no_motion_counter >= MOTION_FRAMES_STOP:
            state = STATE_COOLDOWN

        return

    # --- COOLDOWN: motion ended, but we may need to keep recording ---
    if state == STATE_COOLDOWN:
        # ffmpeg died during cooldown
        if recording is None or recording.poll() is not None:
            print("ffmpeg died during cooldown — returning to IDLE")
            stop_recording()
            state = STATE_IDLE
            return

        # If motion returns, go back to RECORDING
        if motion_counter >= MOTION_FRAMES_START:
            print("Motion (person) detected during cooldown — back to RECORDING")
            state = STATE_RECORDING
            return

        # If cooldown time satisfied, stop recording and go idle
        if recording_start_time is not None:
            if time.monotonic() - recording_start_time >= MIN_RECORDING_SECONDS:
                print("Motion stopped — cooldown then stop recording")
                stop_recording()
                state = STATE_IDLE

        return


def detect_person(frame):
    """
    Run MobileNet-SSD on the frame and return True if a person is detected.
    Uses standard MobileNet-SSD VOC classes where class 15 = 'person'.
    """
    if not dnn_available or net is None:
        return False

    try:
        # MobileNet-SSD expects 300x300 BGR, scale 1/255 ~ 0.007843, mean 127.5
        resized = cv2.resize(frame, (300, 300))
        blob = cv2.dnn.blobFromImage(resized, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        (h, w) = frame.shape[:2]

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < DNN_CONFIDENCE_THRESH:
                continue

            class_id = int(detections[0, 0, i, 1])
            if class_id != PERSON_CLASS_ID:
                continue

            # Scale bounding box back to original frame size
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            box_w = max(0, endX - startX)
            box_h = max(0, endY - startY)
            area = box_w * box_h

            # Ignore very small detections (e.g., noise)
            if area < MIN_PERSON_BBOX_AREA:
                continue

            # At least one valid person detected
            return True

        return False

    except Exception as e:
        # Fail-safe: log once and treat as no person
        print(f"Error during person detection: {e}")
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
            update_state_machine()  # still give state machine a chance to stop recording if needed
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

    # --- 3. PERSON-GATED MOTION LOGIC ---

    # We still use motion as a cheap pre-filter to avoid running DNN every frame.
    if motion_pixels > MOTION_THRESHOLD:
        # Only consider this as "real motion" if a person is detected.
        motion_frame_counter_for_dnn += 1

        run_dnn_this_frame = (
            PERSON_DETECT_EVERY_N <= 1 or
            (motion_frame_counter_for_dnn % PERSON_DETECT_EVERY_N == 0)
        )

        person_detected = False
        if run_dnn_this_frame:
            person_detected = detect_person(frame)

        if person_detected:
            # Person present → this counts as motion for the state machine
            motion_counter += 1
            no_motion_counter = 0
        else:
            # Motion but no person → ignore (likely insects/fog)
            no_motion_counter += 1
            motion_counter = 0
    else:
        # No significant motion
        no_motion_counter += 1
        motion_counter = 0
        motion_frame_counter_for_dnn = 0  # reset DNN counter when motion disappears

    # --- UPDATE RECORDING STATE MACHINE ---
    update_state_machine()

    # --- UPDATE BASELINE ONLY AFTER SEVERAL STABLE FRAMES ---
    if no_motion_counter >= 3:
        last_motion = blur

    # Small sleep to reduce CPU load
    time.sleep(0.02)
