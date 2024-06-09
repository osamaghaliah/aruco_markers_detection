import cv2
import numpy as np
import argparse
import csv
import os
import time
import logging
from collections import defaultdict
from packaging import version

def load_coefficients(file_path):
    """Load camera matrix and distortion coefficients from a YML file."""
    cv_file = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("K").mat()
    dist_coeffs = cv_file.getNode("D").mat()
    cv_file.release()
    return camera_matrix, dist_coeffs

def draw_axes(img, camera_matrix, dist_coeffs, rvec, tvec, length=0.1):
    """Draw the axis on the image manually."""
    points = np.float32([
        [0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, -length]
    ]).reshape(-1, 3)
    imgpts, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs)

    # Check for invalid values
    if not np.all(np.isfinite(imgpts)):
        print(f"Invalid values encountered in projected points: {imgpts}")
        return img

    imgpts = np.int32(imgpts).reshape(-1, 2)

    img = cv2.line(img, tuple(imgpts[0]), tuple(imgpts[1]), (0, 0, 255), 3)
    img = cv2.line(img, tuple(imgpts[0]), tuple(imgpts[2]), (0, 255, 0), 3)
    img = cv2.line(img, tuple(imgpts[0]), tuple(imgpts[3]), (255, 0, 0), 3)
    return img

def detect_aruco_markers(frame, frame_id, aruco_dict, aruco_params, camera_matrix, dist_coeffs, marker_size, detection_stats, csv_writer, drone_position):
    start_time = time.time()  # Start timing
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

    marker_positions = []

    if len(corners) > 0:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

        for rvec, tvec, marker_id, corner in zip(rvecs, tvecs, ids, corners):
            frame = draw_axes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_size / 2)
            cv2.aruco.drawDetectedMarkers(frame, [corner])

            # Draw green rectangle around the marker
            corner = corner.reshape(4, 2).astype(int)
            frame = cv2.polylines(frame, [corner], isClosed=True, color=(0, 255, 0), thickness=2)

            # Compute 3D information
            distance = np.linalg.norm(tvec)
            rmat, _ = cv2.Rodrigues(rvec)
            yaw, pitch, roll = rotationMatrixToEulerAngles(rmat)

            # Add text annotation
            cv2.putText(frame,
                        f"ID: {marker_id[0]}",
                        (corner[0][0], corner[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            detection_stats[marker_id[0]] += 1

            # Transform marker position to world coordinates
            marker_position_world = drone_position + tvec[0]
            marker_positions.append((marker_id[0], marker_position_world))

            # Write to CSV
            qr_2d = [list(corner[0]), list(corner[1]), list(corner[2]), list(corner[3])]
            qr_3d = [distance, yaw, pitch, roll]
            csv_writer.writerow([frame_id, marker_id[0], qr_2d, qr_3d])
    else:
        detection_stats['none'] += 1

    end_time = time.time()  # End timing
    processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
    logging.info(f"Frame {frame_id} processing time: {processing_time:.2f} ms")

    return frame, marker_positions, processing_time

def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.degrees(x), np.degrees(y), np.degrees(z)

def main(video_path, calibration_file, marker_size, output_dir, drone_log_file=None):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frames_dir = os.path.join(output_dir, 'frames')
    if os.path.exists(frames_dir):
        for f in os.listdir(frames_dir):
            os.remove(os.path.join(frames_dir, f))
    else:
        os.makedirs(frames_dir)

    video_output_path = os.path.join(output_dir, 'annotated_video.mp4')
    csv_output_path = os.path.join(output_dir, 'detected_markers.csv')
    log_output_path = os.path.join(output_dir, 'processing_times.log')

    if version.parse(cv2.__version__) >= version.parse("4.7.0"):
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        aruco_params = cv2.aruco.DetectorParameters()
    else:
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
        aruco_params = cv2.aruco.DetectorParameters_create()

    camera_matrix, dist_coeffs = load_coefficients(calibration_file)

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    detection_stats = defaultdict(int)
    processing_times = []

    if drone_log_file:
        with open(drone_log_file, mode='r') as log_file:
            log_reader = csv.DictReader(log_file)
            log_entries = list(log_reader)
    else:
        log_entries = None

    with open(csv_output_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Frame ID', 'QR ID', 'QR 2D', 'QR 3D'])

        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if log_entries and frame_id < len(log_entries):
                log_entry = log_entries[frame_id]
                drone_position = np.array([0, float(log_entry['height']), 0])  # Assuming Y is up, height gives Y
            else:
                # Dummy drone position (assuming it stays at the origin for simplicity)
                drone_position = np.array([0, 0, 0])

            frame, marker_positions, processing_time = detect_aruco_markers(frame, frame_id, aruco_dict, aruco_params, camera_matrix, dist_coeffs, marker_size, detection_stats, csv_writer, drone_position)
            processing_times.append(processing_time)

            # Save the frame
            frame_path = os.path.join(frames_dir, f'frame_{frame_id:04d}.png')
            cv2.imwrite(frame_path, frame)

            out.write(frame)  # Write the frame to the output video

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_id += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Compute and save the average processing time
    average_processing_time = np.mean(processing_times)
    with open(log_output_path, 'a') as log_file:
        log_file.write(f"\nFrame Processing Time Average = {average_processing_time:.2f} ms\n")

    # Print summary of detection statistics
    print("\nDetection Summary:")
    for marker_id, count in detection_stats.items():
        print(f"Marker ID {marker_id}: {count} detections")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArUco Marker Detection in Video')
    parser.add_argument('--video', type=str, required=True, help='Path to the video file')
    parser.add_argument('--calibration_file', type=str, required=True, help='Path to calibration YML file')
    parser.add_argument('--marker_size', type=float, default=0.19, help='Marker size in meters (default: 0.19 for A4 size)')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--drone_log', type=str, help='Path to the drone log CSV file (optional)')
    args = parser.parse_args()

    # Debugging: Print the received arguments
    print("Received arguments:")
    print(f"Video Path: {args.video}")
    print(f"Calibration File: {args.calibration_file}")
    print(f"Marker Size: {args.marker_size}")
    print(f"Output Directory: {args.output_dir}")
    if args.drone_log:
        print(f"Drone Log: {args.drone_log}")
    else:
        print("Drone Log: Not provided")

    # Ensure output directory exists before setting up logging
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Setup logging
    logging.basicConfig(filename=os.path.join(args.output_dir, 'processing_times.log'), filemode='w', level=logging.INFO, format='%(asctime)s %(message)s')

    main(args.video, args.calibration_file, args.marker_size, args.output_dir, args.drone_log)