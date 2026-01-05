In order to use person.py to recognize a person and start recording, a YOLO model saved in onnx format is needed. In addition, a virtual environment is also needed. Once the virtual environment is created, the easiest way is start the script is via systemd because the necessary path is wired in the systemd service file. But you can also start it manually, see Step 6.

---

# 🧱 1. Create/install a virtual environment

```bash
cd ~
python3 -m venv yolo-infer
```

This creates:

```
~/yolo-infer/
    bin/
    lib/
    pyvenv.cfg
```

---

# 🔌 2. Activate the environment

```bash
source yolo-infer/bin/activate
```

Your shell prompt would change to:

```
(yolo-infer) yourname@server:~$
```

---

# 📦 3. Upgrade pip

```bash
pip install --upgrade pip
```

---

# 📚 4. Install the YOLO inference dependencies

For runtime YOLO model/inference, kept things minimal — no training stack, no torch, no ultralytics.
A typical runtime inference dependency modules:

```bash
pip install numpy opencv-python onnxruntime
```

You add missing package if system complains

```bash
pip install whatever_package_is_missing
```

Use your YOLOv5s model (yolov5.onnx) in your python script.
```
YOLO_MODEL_PATH = "/home/yourname/models/yolov5s.onnx" 
```
---

# 🧪 5. Verify the environment

```bash
pip freeze | grep -E '^(numpy|onnxruntime|opencv-python-headless)'
```

Correct versions that yolov5s.oonx is happy with:
<br>Note:  yolov5s.onnx crashes with numpy 2.x, numpy 1.26.4 is happy.
```
numpy==1.26.4  (important: if you see 2.x, pip install numpy==1.26.4)
onnxruntime==1.20.0
opencv-python-headless==4.12.0.88
```

---

# 📁 6. Run your person detector and recording script

Active virtual environment, then invoke

```bash
cd ~
source yolo-infer/bin/activate
(yolo-infer) yourname@server:~$ python motion.py
```
Or start your script with systemd, in that case, there is no need to activate an environment first.

# 📴 7. Deactivate the environment

```bash
deactivate
```

Your prompt returns to normal.

---

# 🎯 Summary


```bash
cd ~
python3 -m venv yolo-infer
source yolo-infer/bin/activate
pip install --upgrade pip
pip install numpy opencv-python-headless onnxruntime
pip freeze | grep -E '^(numpy|onnxruntime|opencv-python-headless)'
python person.py
deactivate
```


