<h1 align="center"> ArUco Markers Detection </h1>

<h2 align="center"> Table of Contents </h2>

----------------------------------------------------------------------------------------------------------------------------------------------------------

- [Project Overview](#project-overview)
- [Features](#features)
- [Preparation](#preparation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How Detection Looks Like](#how-detection-looks-like)
- [Performance In Milliseconds](#performance-in-milliseconds)

----------------------------------------------------------------------------------------------------------------------------------------------------------

## Project Overview

This project involves the detection of ArUco markers in video files using OpenCV. The goal is to process each frame of the video to detect and annotate ArUco markers, and then log the performance of this processing. The project also includes a summary of the average frame processing times for multiple input videos.

Eventually, it aligns the drone camera to markers so it can be directed to perform a path based on the alignments and distances.

## Features
----------------------------------------------------------------------------------------------------------------------------------------------------------

- Detection and annotation of ArUco markers in video frames.
- Logging of processing times for each frame.
- Summary of average processing times for multiple input videos.
- Output of annotated videos and detection logs for each input video.
- Directs a live camera to perform a path/lap based on a known video. 

## Preparation
----------------------------------------------------------------------------------------------------------------------------------------------------------

1. **Clone the repository:**
   ```sh
   git clone https://github.com/osamaghaliah/aruco_marker_detection.git
   cd aruco_marker_detection
   ```

2. **Install required dependencies:**
   ```bash
    pip install opencv-python opencv-contrib-python numpy argparse packaging
   ```

3. **Camera Matrix:**

   |                |                |                |
   |:--------------:|:--------------:|:--------------:|
   | **728.49671263** | **0.000000**   | **640.000000** |
   | **0.000000**     | **728.49671263** | **360.000000** |
   | **0.000000**     | **0.000000**     | **1.000000**   |


## Usage
----------------------------------------------------------------------------------------------------------------------------------------------------------
   
1. **Run the main script:**
   ```sh
   python main.py
   ```
   
   - **Which processes input1 to input6 using 'ArUcoDetector.py' and creates output1 to output6 containing the following:**
      - annotated_video.mp4
      - 'frames' directory of the annotated video.
      - detected_markers.csv
      - processing_times.log
   

    - **How The Terminal Should Look Like During Detection:**
  
       ```
       Running command for input1...
       input1: Frame Processing Time Average = 18.28 ms
       Running command for input2...
       input2: Frame Processing Time Average = 10.21 ms
       Running command for input3...
       input3: Frame Processing Time Average = 14.08 ms
       Running command for input4...
       .
       .
       .
       Summary of processing times has been saved to outputs\performance.csv
       ```

2. **Run DroneAlignment.py:**
    ```sh
   python DroneAlignment.py
   ```
    
   - **Creates 'MovementCommands.csv' which holds movement commands for each frame that had detected markers using the following 8 possible movements:**
      - Move Up.
      - Move Down.
      - Move Right.
      - Move Left.
      - Turn Right.
      - Turn Left.
      - Move Forward.
      - Move Backward.

## Project Structure
----------------------------------------------------------------------------------------------------------------------------------------------------------

The project is structured as follows:

```
aruco_markers_detection/
├── inputs/
│   ├── input1/
│   │   ├── input1.mp4
│   │   └── input1.csv
│   ├── input2/
│   │   ├── input2.mp4
│   │   └── input2.csv
│   ├── input3/
│   │   ├── input3.mp4
│   │   └── input3.csv
│   ├── input4/
│   │   ├── input4.mp4
│   │   └── input4.csv
│   ├── input5/
│   │   └── input5.mp4
│   ├── input6/
│   │   └── input6.mp4
├── outputs/
│   ├── calibration.yml
│   ├── output1/
│   │   ├── frames/
│   │   ├── annotated_video.mp4
│   │   ├── detected_markers.csv
│   │   └── processing_times.log
│   ├── output2/
│   │   ├── frames/
│   │   ├── annotated_video.mp4
│   │   ├── detected_markers.csv
│   │   └── processing_times.log
│   ├── output3/
│   │   ├── frames/
│   │   ├── annotated_video.mp4
│   │   ├── detected_markers.csv
│   │   └── processing_times.log
│   ├── output4/
│   │   ├── frames/
│   │   ├── annotated_video.mp4
│   │   ├── detected_markers.csv
│   │   └── processing_times.log
│   ├── output5/
│   │   ├── frames/
│   │   ├── annotated_video.mp4
│   │   ├── detected_markers.csv
│   │   └── processing_times.log
│   ├── output6/
│   │   ├── frames/
│   │   ├── annotated_video.mp4
│   │   ├── detected_markers.csv
│   │   └── processing_times.log
│   └── performance.csv
├── ArUcoDetector.py
├── CameraMatrix.py
├── DroneAlignment.py
├── main.py
└── README.md
```

## How Detection Looks Like:
----------------------------------------------------------------------------------------------------------------------------------------------------------
![frame_0005](https://github.com/osamaghaliah/aruco_markers_detection/assets/75171676/484d67a7-66aa-453a-809b-eb2d87879392)

![frame_0144](https://github.com/osamaghaliah/aruco_markers_detection/assets/75171676/ed984ee7-c1cb-4640-b952-2dd19ae1ce12)

![frame_0012](https://github.com/osamaghaliah/aruco_markers_detection/assets/75171676/a4d2e990-ffdf-45d9-b181-37e43f0bfe7c)

![frame_0463](https://github.com/osamaghaliah/aruco_markers_detection/assets/75171676/60118e24-fd1c-486a-9eb7-9ddc685672f8)


## Performance In Milliseconds
----------------------------------------------------------------------------------------------------------------------------------------------------------

The following table summarizes the frame processing time averages for each input video:

| INPUT # | Frame Processing Time Average |
|---------|-------------------------------|
| input1  |           18.28 ms            |
| input2  |           10.21 ms            |
| input3  |           14.08 ms            |
| input4  |           10.51 ms            |
| input5  |            5.07 ms            |
| input6  |            4.57 ms            |
