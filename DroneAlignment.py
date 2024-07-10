import cv2
import numpy as np
import pandas as pd
import time
from collections import deque

# Using the updated camera matrix and distortion coefficients
camera_matrix = np.array([
    [728.49671263, 0.000000, 640.000000],
    [0.000000, 728.49671263, 360.000000],
    [0.000000, 0.000000, 1.000000]
])
dist_coeffs = np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000])

# Define a fixed known marker size in meters
KNOWN_MARKER_SIZE = 0.09  # for example, 9 cm

# Pose history for smoothing
pose_history = deque(maxlen=10)


def draw_axes(img, camera_matrix, dist_coeffs, rvec, tvec, length=0.1):
    """Draw the axis on the image manually."""
    points = np.float32([
        [0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, -length]
    ]).reshape(-1, 3)
    imgpts, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs)

    if not np.all(np.isfinite(imgpts)):
        return img

    imgpts = np.int32(imgpts).reshape(-1, 2)

    img = cv2.line(img, tuple(imgpts[0]), tuple(imgpts[1]), (0, 0, 255), 3)
    img = cv2.line(img, tuple(imgpts[0]), tuple(imgpts[2]), (0, 255, 0), 3)
    img = cv2.line(img, tuple(imgpts[0]), tuple(imgpts[3]), (255, 0, 0), 3)
    return img


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


def detect_aruco_markers(frame, aruco_dict, aruco_params):
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

    marker_positions = []
    if len(corners) > 0:
        for marker_id, corner in zip(ids, corners):
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers([corner], KNOWN_MARKER_SIZE, camera_matrix,
                                                                  dist_coeffs)
            for rvec, tvec in zip(rvecs, tvecs):
                frame = draw_axes(frame, camera_matrix, dist_coeffs, rvec, tvec, KNOWN_MARKER_SIZE / 2)
                cv2.aruco.drawDetectedMarkers(frame, [corner])

                corner = corner.reshape(4, 2).astype(int)
                frame = cv2.polylines(frame, [corner], isClosed=True, color=(0, 255, 0), thickness=2)

                distance = np.linalg.norm(tvec)
                rmat, _ = cv2.Rodrigues(rvec)
                yaw, pitch, roll = rotationMatrixToEulerAngles(rmat)

                marker_positions.append((marker_id[0], [distance, yaw, pitch, roll, rvec, tvec]))
    return frame, marker_positions


def average_marker_positions(marker_positions):
    """Compute the weighted average position of all detected markers."""
    if not marker_positions:
        return None

    total_weight = sum(1 / pos[1][0] for pos in marker_positions)  # Weight inversely proportional to distance
    avg_distance = sum((1 / pos[1][0]) * pos[1][0] for pos in marker_positions) / total_weight
    avg_yaw = sum((1 / pos[1][0]) * pos[1][1] for pos in marker_positions) / total_weight
    avg_pitch = sum((1 / pos[1][0]) * pos[1][2] for pos in marker_positions) / total_weight
    avg_roll = sum((1 / pos[1][0]) * pos[1][3] for pos in marker_positions) / total_weight
    avg_rvec = sum((1 / pos[1][0]) * pos[1][4] for pos in marker_positions) / total_weight
    avg_tvec = sum((1 / pos[1][0]) * pos[1][5] for pos in marker_positions) / total_weight

    return [avg_distance, avg_yaw, avg_pitch, avg_roll, avg_rvec, avg_tvec]


def smooth_pose(new_pose):
    pose_history.append(new_pose)
    avg_distance = np.mean([pose[0] for pose in pose_history])
    avg_yaw = np.mean([pose[1] for pose in pose_history])
    avg_pitch = np.mean([pose[2] for pose in pose_history])
    avg_roll = np.mean([pose[3] for pose in pose_history])
    avg_rvec = np.mean([pose[4] for pose in pose_history], axis=0)
    avg_tvec = np.mean([pose[5] for pose in pose_history], axis=0)
    return [avg_distance, avg_yaw, avg_pitch, avg_roll, avg_rvec, avg_tvec]


def droneAlignment():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    fps = 30  # Assuming 30 FPS if the original video FPS is unknown

    targetPose = None
    movementCommands = []

    # Use the correct function based on OpenCV version
    if hasattr(cv2.aruco, 'getPredefinedDictionary'):
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        aruco_params = cv2.aruco.DetectorParameters()
    else:
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
        aruco_params = cv2.aruco.DetectorParameters_create()

    while True:
        startTime = time.time()
        ret, currentFrame = cap.read()
        if not ret:
            break

        currentFrame, marker_positions = detect_aruco_markers(currentFrame, aruco_dict, aruco_params)

        if marker_positions:
            avg_pose = average_marker_positions(marker_positions)
            smoothed_pose = smooth_pose(avg_pose)
            if targetPose is None:
                targetPose = smoothed_pose
            movementCommand = determineMovementCommands(smoothed_pose, targetPose, smoothed_pose[0], targetPose[0])
            movementCommandString = " | ".join(movementCommand)
            print(f"Movement Command: [{movementCommandString}]")

            # Append only relevant frames
            movementCommands.append([time.time(), f"[{movementCommandString}]"])
        else:
            movementCommandString = "No Marker"

        combinedHeight, combinedWidth = currentFrame.shape[:2]
        maxWidth, maxHeight = 1920, 1080
        scaleFactor = min(maxWidth / combinedWidth, maxHeight / combinedHeight)
        newWidth = int(combinedWidth * scaleFactor)
        newHeight = int(combinedHeight * scaleFactor)
        resizedCombinedFrame = cv2.resize(currentFrame, (newWidth, newHeight))

        cv2.putText(resizedCombinedFrame, f"Movement: [{movementCommandString}]", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Live Video', resizedCombinedFrame)

        elapsedTime = time.time() - startTime
        delay = max(1, int((1 / fps - elapsedTime) * 1000))  # Calculate delay to maintain the frame rate
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save movement commands to CSV file
    movementCommandsPath = 'MovementCommands.csv'
    movementDF = pd.DataFrame(movementCommands, columns=['Time', 'Movement Commands'])
    movementDF.to_csv(movementCommandsPath, index=False)


def determineMovementCommands(currentPose, targetPose, currentDistance, targetDistance):
    """
    Calculate movement commands based on the difference between current and target positions.

    Args:
    currentPose (list): Current pose of the drone.
    targetPose (list): Target pose of the drone.
    currentDistance (float): Current distance of the drone from the marker.
    targetDistance (float): Target distance of the drone from the marker.

    Returns:
    list: List of movement commands.
    """
    distanceDifference = targetDistance - currentDistance
    yawDifference = targetPose[1] - currentPose[1]
    pitchDifference = targetPose[2] - currentPose[2]
    rollDifference = targetPose[3] - currentPose[3]

    commands = []

    # Determine forward/backward movement based on distance difference
    if abs(distanceDifference) > 0.2:
        if distanceDifference > 0:
            commands.append("Move Backward")
        else:
            commands.append("Move Forward")

    # Determine left/right movement based on roll difference
    if abs(rollDifference) > 0.25:
        if rollDifference > 0:
            commands.append("Move Left")
        else:
            commands.append("Move Right")

    # Determine turn direction based on yaw difference
    if abs(yawDifference) > 5:
        if yawDifference > 0:
            commands.append("Turn Left")
        else:
            commands.append("Turn Right")

    # Determine up/down movement based on pitch difference
    if abs(pitchDifference) > 5:
        if pitchDifference > 0:
            commands.append("Move Down")
        else:
            commands.append("Move Up")

    return commands if commands else ["In Position"]


if __name__ == "__main__":
    droneAlignment()
