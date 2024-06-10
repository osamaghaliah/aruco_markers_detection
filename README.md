<h1 align="center"> ArUco Markers Detection </h1>

## Project Overview

This project involves the detection of ArUco markers in video files using OpenCV. The goal is to process each frame of the video to detect and annotate ArUco markers, and then log the performance of this processing. The project also includes a summary of the average frame processing times for multiple input videos.

## Table of Contents
----------------------------------------------------------------------------------------------------------------------------------------------------------

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Performance In Milliseconds](#performance-in-milliseconds)

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