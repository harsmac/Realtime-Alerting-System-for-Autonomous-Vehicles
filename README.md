# Realtime-Alerting-System-for-Autonomous-Vehicles
The following repository contains the code for the paper: **"AN EVALUATION METRIC FOR OBJECT DETECTION ALGORITHMS IN AUTONOMOUS NAVIGATION SYSTEMS AND ITS APPLICATION TO A REAL-TIME ALERTING SYSTEM"** which is to be presented at **IEEE ICIP,2018**. https://2018.ieeeicip.org/Papers/ViewPapers_MS.asp?PaperNum=2723 
IEEE Xplore link : https://ieeexplore.ieee.org/document/8451718

## The repository may be divided into three sections:
### 1. metric_codes:
This folder contains the code for scoring the clone and bad image weather conditions.It also contains precalculated scores on VKITTI dataset for the 0001,0002,0006,0018,0020 folder.

### 2. realtime_application: 
The second folder contains the script for running a realtime alerting system using the precalculated distance vectors from VGG 16 which are made available here:
https://drive.google.com/open?id=1TO6UEdBqJaRKV9AMWFd9nffi7oXk09Bk

### 3. realtime_result_images:
The third folder contains few of the test images from the Oxford Dataset, their scores and the explanations for the same.

