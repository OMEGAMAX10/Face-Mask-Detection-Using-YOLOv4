# Face Mask Detector using YOLOv4
Because of the COVID-19 pandemic of 2020, more and more people are concerned with protecting themselves using masks, thus the need of software capable of monitoring whether the people are wearing masks or not. That is why I created a PyQt5 application using OpenCV (with CUDA support) based on the YOLOv4 algorithm, capable of monitoring the safety level of a space with video surveillance through multiple video cameras, connected either via USB or IP to the system.


### Installing dependencies:
#### For Windows:
1. Check if the GPU of the system supports CUDA by checking if it is in this list: https://developer.nvidia.com/cuda-gpus
2. If the GPU supports CUDA, install it using this guide: https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781
3. Install the latest Python 3 version from https://www.python.org/downloads/
4. Run this command in command line:
    ```powershell
    PS C:> pip install numpy PyQt5 PyQt5-stubs pyqt5-tools
    ```
5. Install OpenCV with CUDA support using this tutorial: https://www.youtube.com/watch?v=TT3_dlPL4vo&list=LL&index=1&t=41s

#### For Ubuntu/Debian:
1. Check if the GPU of the system supports CUDA by checking if it is in this list: https://developer.nvidia.com/cuda-gpus
2. If the GPU supports CUDA, install it using this guide: https://medium.com/analytics-vidhya/installing-tensorflow-with-cuda-cudnn-gpu-support-on-ubuntu-20-04-f6f67745750a
3. Run these commands:
```console
foo@bar:~$ sudo chmod +x install_dependencies_ubuntu.sh    # make the script executable
foo@bar:~$ sudo ./install_dependencies_ubuntu.sh           # run the script to install the dependencies of the application
```


### Guide for using the Face Mask Detector application:

1. Execute the Python Script from command line like this:
```console
foo@bar:~$ python .\face_mask_detection.py
```
or
```console
foo@bar:~$ python3 .\face_mask_detection.py
```

2. From the start menu, you can add or delete a camera from the camera list. When creating a camera, a name and an ID must be provided. The ID must be from one of these categories:
    - **integer (e.g.: 0, 1, 2...):** A camera with this ID represents a video recording device physically connected to the system which uses the application. For instance, if you want to use the webcam of a laptop, you must create a camera with an ID of 0 (an explanation would be that, in particular for Ubuntu, the integrated camera of a laptop is interpreted as /dev/video0).
    - **IP address (e.g.: https://192.168.43.1:8080/video):** A camera with this ID represents a video recording device connected to the same network as the system which uses the application. For example, one can connect an Android device as a remote camera using "IP Webcam" Google Playstore app: https://play.google.com/store/apps/details?id=com.pas.webcam&hl=ro&gl=US.
    - **Video file location:** A camera with this ID represents a locally stored video, on which our application will run the detection. This case is useful whenever we have an already pre-recorded video file, possibly from a camera that was nou connected to a system with this application.
    
3. When you think that the camera list is ready, you can access the main menu.

4. In the main menu, you will have the following elements: 
    - **A main panel** where the selected camera will be displayed together with the detections of the masked or unmasked faces from it, as well as real time statistics of their numbers and its status ("Not Connected" - the camera is not connected to the system, "Safe" - all people wear mask, "Warning" - 1 or 2 people do not wear a mask, "Danger" - more than 3 people do not wear a mask);
    - **A small selection menu** in the upper part of the window from where you can select which camera do you whish to visualize;
    - **A camera control panel** in the right part of the window where the camera list is displayed together with the status of every camera in real time;
    - **A "Take Photo" button** placed in the lower part of the window used for taking photos for further analysis in order to identify the persons not wearing a mask.
    
5. Despite the fact that you can take photos manually from the main menu, the application makes each of the connected cameras capable of taking photos automatically with whenever its state switches to "Warning" or "Danger".


### Datasets and weights used by the Face Mask Detector application:
The dataset used for training this model is the one from Kaggle: https://www.kaggle.com/alexandralorenzo/maskdetection

The trained YOLOv4 weights, together with the configuration file can be found at this link: https://mega.nz/folder/SwADAYzR#Xv9Wz6wjW4iYpfx4W_0gZg
