In order to use this python script to recognize a person and start recording a YOLO model saved in onnx format is needed. IN addition, a virtual environment is also needed. Once the virtual environment is created, the easiest way is to start the script using systemd because the necessary path is given in the systemd service file.

---

# 🧱 1. Create the virtual environment

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
(yolo-infer) nicholas@server3:~
```

---

# 📦 3. Upgrade pip

```bash
pip install --upgrade pip
```

---

# 📚 4. Install the inference dependencies

For inference, you kept things minimal — no training stack, no torch, no ultralytics.  
Your typical inference stack looks like:

```bash
pip install numpy opencv-python onnxruntime
```

Sometimes you also add missing package if system complains

```bash
pip install whatever_package_is_missing
```

Use your ONNX YOLOv5s model (yolov5.onnx) in your python script.

---

# 🧪 5. Verify the environment

```bash
pip freeze | grep -E '^(numpy|onnxruntime|opencv-python-headless)'
```

Expected versions:

```
numpy==1.26.4  (important, if you see 2.x, pip install numpy==1.26.4)
onnxruntime==1.20.0
opencv-python-headless==4.12.0.88
```

---

# 📁 6. Run your inference script

Active the virtual environment, see 2. above

```bash
python motion.py
```
Or run your script with systemd, there is no need to activate environment.

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


