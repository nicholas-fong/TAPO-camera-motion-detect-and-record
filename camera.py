import cv2
import time
import datetime
import subprocess
from zoneinfo import ZoneInfo

# --- CONFIG ---
rtsp_url_ffmpeg = "rtsp://username:password@<ip_address>/stream1"  # high-res for recording
rtsp_url_cv     = "rtsp://username:password@<ip_address>/stream2"  # low-res for motion detection

MOTION_THRESHOLD       = 500      # pixel-count threshold on 640x360 diff
MEAN_SENSITIVITY       = 0.3      # "Nuclear" gatekeeper: skip math if mean change < this
MOTION_FRAMES_START    = 1        # frames above threshold to start recording
MOTION_FRAMES_STOP     = 10       # frames below threshold to consider motion ended
MIN_RECORDING_SECONDS  = 10       # minimum recording duration
FRAME_SKIP_INTERVAL    = 2        # process every nth frame (from ~30 FPS -> 15 FPS effective)

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


# --- FUNCTIONS ----
def open_capture():
    """ Open RTSP stream with retry logic and minimized buffer."""
    while True:
        cap = cv2.VideoCapture(rtsp_url_cv)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        print("Failed to open RTSP stream — retry in 2s")
        time.sleep(2)


def start_recording():
    """ Launch ffmpeg and return the process handle and start time."""
    ts = datetime.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d_%H%M%S")  # modify your time zone
    filename = f"~/tapo/cam1_{ts}.mp4"     # you decide where to store the video clips, you can store as .mkv if you want

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
            print(f"Motion detected — start recording")
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
            print("Motion detected during cooldown — start recording")
            state = STATE_RECORDING
            return

        # If cooldown time satisfied, stop recording and go idle
        if recording_start_time is not None:
            if time.monotonic() - recording_start_time >= MIN_RECORDING_SECONDS:
                print("Motion stopped — cooldown then stop recording")
                stop_recording()
                state = STATE_IDLE

        return


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

    if motion_pixels > MOTION_THRESHOLD:
        motion_counter += 1
        no_motion_counter = 0
    else:
        no_motion_counter += 1
        motion_counter = 0

    # --- UPDATE RECORDING STATE MACHINE ---
    update_state_machine()

    # --- UPDATE BASELINE ONLY AFTER SEVERAL STABLE FRAMES ---
    if no_motion_counter >= 3:
        last_motion = blur

    # Small sleep to reduce CPU load
    time.sleep(0.02)
    
