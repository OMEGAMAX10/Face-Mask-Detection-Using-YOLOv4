# Face-Mask-Detection-Using-YOLOv4
Because of the COVID-19 pandemic of 2020, more and more people are concerned with protecting themselves using masks, thus the need of software capable of monitoring whether the people are wearing masks or not. That is why I created a PyQt5 application using OpenCV (with CUDA support) based on the YOLOv4 algorithm, capable of monitoring the safety level of a space with video surveillance through multiple video cameras, connected either via USB or IP to the system.

**How to use the application:**

1. Execute the Python Script from command line like this:
```console
foo@bar:~$ python .\face_mask_detection.py
```
or
```console
foo@bar:~$ python3 .\face_mask_detection.py
```

2. From the start menu, you can add or delete a camera from the camera list. When creating a camera, a name and an ID must be provided. The ID must be from one of these categories:
    - **integer (e.g. 0, 1, 2...):** A camera with this ID represents a video recording device physically connected to the system which uses the application. For instance, if you want to use the webcam of a laptop, you must create a camera with an ID of 0 (an explanation would be that, in particular for Ubuntu, the integrated camera of a laptop is interpreted as /dev/video0).
    - **IP address (e.g. https://192.168.43.1:8080/video):** A camera with this ID represents a video recording device connected to the same network as the system which uses the application. For example, one can connect an Android device as a remote camera using "IP Webcam" Google Playstore app: https://play.google.com/store/apps/details?id=com.pas.webcam&hl=ro&gl=US.
    - **Video file location:** A camera with this ID represents a locally stored video, on which our application will run the detection. This case is useful whenever we have an already pre-recorded video file, possibly from a camera that was nou connected to a system with this application.
    
3. When you think that the camera list is ready, you can access the main menu

4. 

The dataset used for training this model is the one from Kaggle: https://www.kaggle.com/alexandralorenzo/maskdetection

The trained YOLOv4 weights, together with the configuration file can be found at this link: https://mega.nz/folder/SwADAYzR#Xv9Wz6wjW4iYpfx4W_0gZg
