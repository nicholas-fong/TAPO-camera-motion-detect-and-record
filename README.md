# TAPO camera: <br>motion-detection -> video-recording <br>

The script detects motion using Open Source Computer Vision Library, after detection, starts recording video.

The following two packages are required: <br>
*sudo apt install python3-opencv ffmpeg*

Not suitable for outdoor camera with built-in IR: due to TAPO camers's night time IR illumination, flying insects, bugs, spider webs, fog droplets and snow will cause excessive false positives.


Python script using RTSP (Real-Time Streaming Protocol) to:

(1) acquire video frames from camera.
<br>
(2) detect motion using opencv library (Open-Source Computer Vision Library). 
<br>
(3) save video clips locally or to a NAS using ffmpeg. 


This script is tuned for consumer TAPO cameras (e.g. C200, C210) and optimized to keep CPU load to a minimum.


Edit the script for your local environment as follows:

(1) Your TAPO camera's ip address, username and password that you created using the TAPO app.
<br>
(2) Inside start_recording() procedure, edit the path you want to store the video clips and change the timezone parameter.

# Sensitivity Tuning:
-In the Motion gate block, fine tune MOTION_THRESHOLD and MEAN_SENSITIVITY.
<BR><BR>

# TAPO camera: <br> person-detection -> video-recording <br>

This scriipt detects a person using Open Source Computer Vision Library, after detecting motion and person, starts recording video. Small objects like flying insects or fog droplets will not cause false positives.

The following three packages are required: <br>
*sudo apt install python3-opencv ffmpeg numpy*

Two MobileNet_SSD files are required: <br>
(0) Create a sub-directory, e.g. ~\models\mobilenet_ssd\ 
<br>
(1) download MobileNetSSD_deploy (31k bytes) 
<br>
(2) download MobileNetSSD_deploy.caffemodel (22 Mbytes).  

https://github.com/chuanqi305/MobileNet-SSD  or  <br>https://github.com/lironghua318/MobileNet-SSD-1

Edit the script for your local environment as follows:

(1) Your TAPO camera's ip address, username and password that you created using the TAPO app.
<br>
(2) Inside start_recording() procedure, edit the path you want to store the video clips and change the timezone parameters.
<br>
(3) Make sure the PROTOTXT_PATH and CAFFEMODEL_PATH are pointing to the correct directory.

# Sensitivity Tuning:

-In the Motion gate block, fine tune MOTION_THRESHOLD and MEAN_SENSITIVITY.
<br>
-In the Person detection block, fine tune MIN_PERSON_BBOX_AREA and DNN_CONFIDENCE_THRESH.
