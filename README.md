<h1 align="center"> ArUco Markers Detection </h1>

<h2 align="center"> Table of Contents </h2>

----------------------------------------------------------------------------------------------------------------------------------------------------------

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How Detection Looks Like](#how-detection-looks-like)
- [Performance In Milliseconds](#performance-in-milliseconds)

----------------------------------------------------------------------------------------------------------------------------------------------------------

## Project Overview

This project involves the detection of ArUco markers in video files using OpenCV. The goal is to process each frame of the video to detect and annotate ArUco markers, and then log the performance of this processing. The project also includes a summary of the average frame processing times for multiple input videos.

## Features
----------------------------------------------------------------------------------------------------------------------------------------------------------

- Detection and annotation of ArUco markers in video frames.
- Logging of processing times for each frame.
- Summary of average processing times for multiple input videos.
- Output of annotated videos and detection logs for each input video.

## Installation
----------------------------------------------------------------------------------------------------------------------------------------------------------

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/aruco-marker-detection.git
   cd aruco-marker-detection
   ```

2. **Install required dependencies:**
   ```bash
    pip install opencv-python opencv-contrib-python numpy argparse packaging
   ```

3. **Ensure you have the necessary calibration file:**
   - Place your calibration file (`calibration.yml`) in the `outputs` directory.

## Usage
----------------------------------------------------------------------------------------------------------------------------------------------------------

1. **Prepare your input files:**
   - Place your input video files and corresponding CSV drone logs in the `inputs` directory. Each input should be in a separate folder named `input1`, `input2`, ..., `input5`.

2. **Calibrate the Camera:**
   
   Use the checkerboard images in the `calibration_images/` directory to calibrate your camera. Run the `CameraCalibration.py` script to generate `calibration.yml`.
   ```sh
   python CameraCalibration.py --image_dir calibration_images --image_format jpg --prefix image --square_size 1.0 --save_file calibration.yml
   ```

   ![image](https://github.com/osamaghaliah/aruco_markers_detection/assets/75171676/b89adfe6-f381-49a5-ae2f-a10f3e972b34)

   
    - **How The Terminal Should Look Like After Calibration Is Finished:**
  
       ```
       Number of successful detections: 11
       Calibration is finished. RMS:  1.1957113482361996
       ```
   
   
3. **Run the main script:**
   
   ```sh
   python main.py
   ```

    - **How The Terminal Should Look Like After Detection Is Finished:**
  
       ```
       Running command for input1...
       input1: Frame Processing Time Average = 18.28 ms
       Running command for input2...
       input2: Frame Processing Time Average = 10.21 ms
       Running command for input3...
       input3: Frame Processing Time Average = 14.08 ms
       Running command for input4...
       input4: Frame Processing Time Average = 10.51 ms
       Running command for input5...
       input5: Frame Processing Time Average = 5.07 ms
    
       Summary of processing times has been saved to outputs\performance.csv
       ```

## Project Structure
----------------------------------------------------------------------------------------------------------------------------------------------------------

The project is organized as follows:

```
aruco_markers_detection/
├── calibration_images/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   ├── image4.jpg
│   ├── image5.jpg
│   ├── image6.jpg
│   ├── image7.jpg
│   ├── image8.jpg
│   ├── image9.jpg
│   ├── image10.jpg
│   └── image11.jpg
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
│   └── performance.csv
├── CameraCalibration.py
├── ArUcoDetector.py
├── main.py
└── README.md
```

## How Detection Looks Like:
----------------------------------------------------------------------------------------------------------------------------------------------------------
![frame_0005](https://github.com/osamaghaliah/aruco_markers_detection/assets/75171676/484d67a7-66aa-453a-809b-eb2d87879392)

![frame_0252](https://github.com/osamaghaliah/aruco_markers_detection/assets/75171676/4009d5e3-b5d4-46d3-9d52-f5cd11fc9476)

![frame_0144](https://github.com/osamaghaliah/aruco_markers_detection/assets/75171676/ed984ee7-c1cb-4640-b952-2dd19ae1ce12)

![frame_0012](https://github.com/osamaghaliah/aruco_markers_detection/assets/75171676/a4d2e990-ffdf-45d9-b181-37e43f0bfe7c)

![frame_0390](https://github.com/osamaghaliah/aruco_markers_detection/assets/75171676/18ee2312-e7dc-47fa-aef8-5e5c0631a9b4)

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
