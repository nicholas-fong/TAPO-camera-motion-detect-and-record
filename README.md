# TAPO camera motion detect record <br>
Python script using RTSP (Real-Time Streaming Protocol) to (1) acquire video frames, (2) detect motion using opencv library (Open-Source Computer Vision Library), and (3) save video clips locally or to a NAS using ffmpeg. 
<br>
The script is tuned for consumer TAPO cameras (e.g. C200) and optimized to keep CPU load to a minimum.
<br>
Edit the script for your local environment:
<br>
Your TAPO camera's username and password and IP address.
<br>
The path you want to save the video clips (inside start_recording() function).
<br>
The timezone you are in (inside start_recording() function).
<br>
In the CONFIG block, you can fine tune the parameters for detection sensitivity.
<br><br>
sudo apt install python3-opencv ffmpeg
<br>
<br>
This Python script works well in day night or at night with visible LED lighting. However, when the TAPO cameera's IR LED are illuminated as the sole source of lighting, motion detection becomes hyper-sensitive to flying insects or fog droplets, causing too massive positive recordings.

