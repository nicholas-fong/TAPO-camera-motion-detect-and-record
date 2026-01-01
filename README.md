## TAPO Camera Detection Pipeline  
### Motion or Person Detection Using detect1.py and detect2.py

This project provides two independent detection scripts for TAPO security cameras. Each script reads the camera’s RTSP sub‑stream and performs real‑time analysis to determine whether meaningful activity is occurring.

Only **one** script runs at a time, depending on whether you want:

- **Motion detection** (OpenCV)  
- **Person detection** (YOLOv8n ONNX)

Each script runs as a standalone systemd service.  

---

## 📦 Detection Scripts

### 1. detect1.py — OpenCV Motion Detection  
A lightweight, CPU‑efficient motion detector using frame differencing.

**Features**
- Uses system Python (`/usr/bin/python3`)
- Extremely fast and low‑CPU
- Ideal for general motion detection
- Adjustable sensitivity (threshold, min area)
- Prone to false positives (shadows, rain, headlights)

**script**
1. Read RTSP frame  
2. Convert to grayscale  
3. Apply Gaussian blur  
4. Compute frame difference  
5. Threshold + dilate  
6. Find contours  
7. If contour area > MIN_AREA → motion detected  

---

### 2. detect2.py — YOLOv8n ONNX Person Detection  
A modern neural detector that identifies **people only**, using the YOLOv8n model exported to ONNX.

**Features**
- Requires setup of virtual environment (yolo-env)
- High accuracy on low‑resolution TAPO streams
- Fewer false positives than motion detection
- Adjustable confidence and NMS thresholds
- Optional debug mode with bounding boxes

**script**
1. Read RTSP frame  
2. Resize + normalize to YOLO input  
3. Run ONNX inference  
4. Parse model outputs  
5. Filter for class `person`  
6. Apply NMS  
7. If any person detected → event logged  

---

## 📁 Directory Structure

```
.
├── detect1.py        # OpenCV motion detection
├── detect2.py        # YOLOv8n ONNX person detection
├── models/
│   └── yolov8n.onnx
├── systemd/
│   ├── detect1.service
│   └── detect2.service
└── README.md
```

---

## ⚙️ Systemd Setup

Each script has its own systemd service.  
Only **one** should be enabled at a time.

### 1. Install service files

```
sudo cp systemd/detect1.service /etc/systemd/system/
sudo cp systemd/detect2.service /etc/systemd/system/
sudo systemctl daemon-reload
```

---

## ▶️ Running the Motion Detector (detect1.py)

Enable and start:

```
sudo systemctl enable --now detect1.service
```

Disable:

```
sudo systemctl disable --now detect1.service
```

### detect1.service (example)

```
[Unit]
Description=OpenCV Motion Detection (detect1.py)
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /path/to/detect1.py
Restart=always
RestartSec=2
User=yourusername
WorkingDirectory=/path/to/

[Install]
WantedBy=multi-user.target
```

Replace `/path/to/` and `yourusername` as needed.

---

## ▶️ Running the Person Detector (detect2.py)

Enable and start:

```
sudo systemctl enable --now detect2.service
```

Disable:

```
sudo systemctl disable --now detect2.service
```

### detect2.service (example)

```
[Unit]
Description=YOLOv8n ONNX Person Detection (detect2.py)
After=network.target

[Service]
Type=simple
ExecStart=/full/path/to/python /full/path/to/detect2.py
Restart=always
RestartSec=2
User=yourusername
WorkingDirectory=/path/to/

[Install]
WantedBy=multi-user.target
```

Example ExecStart:

```
ExecStart=/home/nicholas/miniconda3/envs/yolo/bin/python /home/nicholas/projects/camera/detect2.py
```

---

## 🔄 Switching Between Detection Modes

Motion → Person:

```
sudo systemctl disable --now detect1.service
sudo systemctl enable --now detect2.service
```

Person → Motion:

```
sudo systemctl disable --now detect2.service
sudo systemctl enable --now detect1.service
```

Only one service should run at a time.

---

## 🧪 Debugging

### Motion detector:

```
python3 detect1.py --debug
```

### YOLO detector:

```
/full/path/to/python detect2.py --debug
```

Debug mode overlays bounding boxes and prints detection logs.

Microsoft Copilot helped to shape the code structure.
