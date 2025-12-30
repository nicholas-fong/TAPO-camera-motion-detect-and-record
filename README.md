# TAPO camera: <br>motion-detection -> video-recording (for indoor use) 

The script detects motion, then start recording video.

The following two packages are required: <br>
*sudo apt install python3-opencv ffmpeg*

Rason for just indoor use: due to TAPO camers's night time IR illumination, flying insects, bugs, spider webs, fog droplets and snow will cause excessive false positives.


Python script using RTSP (Real-Time Streaming Protocol) to:

(1) acquire video frames from camera.

(2) detect motion using opencv library (Open-Source Computer Vision Library). 

(3) save video clips locally or to a NAS using ffmpeg. 


The script is tuned for consumer TAPO cameras (e.g. C200) and optimized to keep CPU load to a minimum.


Edit the script for your local environment as follows:

(1) Your TAPO camera's ip address, username and password that you created using the TAPO app.

(2) Inside start_recording(), edit the path you want to store the video clips and change the timezone parameters.


# Sensitivity Tuning:

In the CONFIG block, fine tune the parameters for motion detection sensitivity.


# TAPO camera: <br> person-detection -> video-recording (for indoor/outdoor use)

This scriipt detects a person, then start recording video. Small objects like flying insects or fog droplets will not cause false positives.

The following two packages are required: <br>
*sudo apt install python3-opencv ffmpeg numpy*

Need two MobileNet_SSD files. 

Suggested sub-directory is ~\models\mobilenet_ssd to store the two necessary files.

(1) MobileNetSSD_deploy (31k bytes) 

(2) MobileNetSSD_deploy.caffemodel (22 Mbytes).  

https://github.com/chuanqi305/MobileNet-SSD  and  https://github.com/lironghua318/MobileNet-SSD-1

Edit the script for your local environment:

(1) Your TAPO camera's ip address, username and password that you created using the TAPO app.

(2) Inside start_recording(), edit the path you want to store the video clips and change the timezone parameters.

# Sensitivity Tuning:

In the Motion gate block, fine tune MOTION_THRESHOLD and MEAN_SENSITIVITY .
in the Person detection block, fine tune MIN_PERSON_BBOX_AREA and DNN_CONFIDENCE_THRESH.
