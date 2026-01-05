## TAPO Camera Detect and Record script  
### Motion Detection only - motion.py

This project provides detection Python scripts mostly for TAPO security cameras. The script reads the camera’s RTSP sub‑stream and performs real‑time analysis to determine whether meaningful activity is occurring.


- **Motion detection** using OpenCV library


---

## 📦 Detection Script

### motion.py — OpenCV Motion Detection  
A lightweight, CPU‑efficient motion detector using frame differencing.

**Features**
- Uses system Python (`/usr/bin/python3`)
- Extremely fast and low‑CPU
- Ideal for general motion detection
- Tunable sensitivity (threshold, min area)
- Cons: Prone to false positives (shadows, rain, headlights)

**script**
1. Read RTSP frame  
2. Convert to grayscale  
3. Apply Gaussian blur  
4. Compute frame difference  
5. Threshold + dilate  
6. Find contours  
7. If contour area > MIN_AREA → motion detected  
8. Record


---

## ⚙️ Systemd Setup


### 1. Install systemd service files (run in background)

```
sudo cp systemd/motion.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start motion.service
```

---

## ▶️ Running the Motion Detector

Start:

```
sudo systemctl start detect1.service
```

Stop:

```
sudo systemctl stop detect1.service
```

### motion.service (example)

```
[Unit]
Description=OpenCV Motion Detection
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /path/to/motion.py
Restart=always
RestartSec=2
User=yourusername
WorkingDirectory=/path/to/

[Install]
WantedBy=multi-user.target
```

---

## 🧪 Debugging

### Motion detector:

```
python3 /path/to/motion.py --debug
```

Microsoft Copilot helped to shape the code structure and generate README
