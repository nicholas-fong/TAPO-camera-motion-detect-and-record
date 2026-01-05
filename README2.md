# Spin (export) a YOLO model
# YOLOv5s ONNX Export (FP32, CPU‑Compatible, Opset 12)

This repository documents a **fully reproducible procedure** for exporting a clean, CPU‑compatible (as opposed to GPU only) **YOLOv5s.onnx** model that has these features:

- **FP32 precision**  
- **Opset 12** (for maximum compatibility with ONNX Runtime and OpenCV DNN)  
- **CPU‑only export**  
- **Minimal, deterministic Python environment**  

The resulting `yolov5s.onnx` is suitable for embedded inference, OpenCV DNN, ONNX Runtime, and CPU‑only deployments.

---

## 📦 1. Create a Clean Python Environment

```bash
cd ~
python3 -m venv yolov5-export-env
source yolov5-export-env/bin/activate
pip install --upgrade pip
```

---

## 📥 2. Clone YOLOv5 and Download Weights

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5

wget https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt
```

---

## 📄 3. Create a Minimal, Deterministic `requirements.txt`

Create or edit the file:

```bash
nano requirements.txt
```

Paste the following, delete the other useless lines:

```
# YOLOv5 requirements
# Usage: pip install -r requirements.txt
# the order of the modules is important
numpy
opencv-python
pandas
seaborn
tqdm
ultralytics
onnx
onnxscript
```

Install:

```bash
pip install -r requirements.txt
```

(Installation takes a couple of minutes.)

---

## 🛠️ 4. Export (spin) YOLOv5s to ONNX format that runs on CPU

Run the export:

```bash
python export.py --weights yolov5s.pt --include onnx --opset 12 --device cpu
```

During export, safely ignore **RuntimeError** during export. It doesn't affet the export of the model.

In the export log, you should see a line similar to:

```
YOLOV5 v7.0-453-geed9bc19 Python-3.12.3 torch-2.9.1+cu128 CPU
```

---

## 🔐 5. Verify Model Integrity (SHA‑256)

```bash
sha256sum yolov5s.onnx yolov5s.pt
```

Correct checksums:

```
8dc6207beff9ae317fd0ee91a1dcc52e2cde4ee56e6d34cf3387bdc3d47c6399  yolov5s.onnx
8b3b748c1e592ddd8868022e8732fde20025197328490623cc16c6f24d0782ee  yolov5s.pt
```

---

## 🧪 6. Optional, confirm key package versions

```bash
pip freeze | grep -E '^(numpy|onnx|opencv-python)'
```

Expecte to see:

```
numpy==2.2.6
onnx==1.20.0
onnx-ir==0.1.13
onnxscript==0.5.7
opencv-python==4.12.0.88
```

---

## 💾 7. Save Your Model

Copy these files to a very safe location:

- `yolov5s.onnx`
- `yolov5s.pt`

---

## 🧹 8. Clean Up the Export Environment

```bash
deactivate
rm -rf yolov5 yolov5-export-env
```

Your  `yolov5s.onnx` is now ready for deployment.

---

## ✔ Summary

This procedure guarantees:

- Deterministic ONNX export  
- CPU‑compatible FP32 model  
- Opset 12 for maximum compatibility  
- Clean, minimal environment  
- Verified SHA‑256 hashes  
- Reproducible results  
