# TAPO indoor camera : motion-detection -> video-recording <br>

For TAPO camera indoor use only. 

Rasons: due to TAPO camers's night time IR illumination, flying insects, bugs, spider webs, fog droplets and snow will cause excessive false positives.


Python script using RTSP (Real-Time Streaming Protocol) to:

(1) acquire video frames from camera.

(2) detect motion using opencv library (Open-Source Computer Vision Library). 

(3) save video clips locally or to a NAS using ffmpeg. 


The script is tuned for consumer TAPO cameras (e.g. C200) and optimized to keep CPU load to a minimum.


Edit the script for your local environment as follows:

(1) Your TAPO camera's IP address, username and password and IP address that you created using the TAPO app.

(2) Inside start_recording(), edit the path you want to store the video clips and change the timezone parameters.


# Sensitivity Tuning:

In the CONFIG block, fine tune the parameters for motion detection sensitivity.

The following two packages are required: <br>
*sudo apt install python3-opencv ffmpeg*
