import cv2
import numpy as np
import argparse
import os
import pandas as pd
import time


def loadAnnotatedFrames(directory):
    """
    Load frames from the specified directory.

    Args:
    directory (str): Path to the directory containing frame images.

    Returns:
    list: List of frames read from the directory.
    """
    frameFiles = sorted([f for f in os.listdir(directory) if f.endswith('.png')])
    frames = []
    for frameFile in frameFiles:
        framePath = os.path.join(directory, frameFile)
        frame = cv2.imread(framePath)
        frames.append(frame)
    return frames


def convertQR3DStringToList(qr3DString):
    """
    Convert QR 3D data string to a list.

    Args:
    qr3DString (str): String representation of QR 3D data.

    Returns:
    list: List of QR 3D data.
    """
    qr3DString = qr3DString.replace('NaN', 'None')
    qr3D = eval(qr3DString)
    return qr3D


def findClosestMarker(frameData):
    """
    Select the primary marker from the frame data based on minimum distance.

    Args:
    frameData (DataFrame): DataFrame containing markers' data for a frame.

    Returns:
    Series: Row containing data for the closest marker.
    """
    closestMarker = None
    minimumDistance = float('inf')
    for index, row in frameData.iterrows():
        qr3D = convertQR3DStringToList(row['QR 3D'])
        distance = qr3D[0]
        if distance is not None and distance < minimumDistance:
            closestMarker = row
            minimumDistance = distance
    return closestMarker


def droneAlignment(framesDirectory, markersCSVPath):
    """
    Main function to align drone based on annotated frames and detected markers.

    Args:
    framesDirectory (str): Directory containing annotated frames.
    markersCSVPath (str): Path to the CSV file containing detected markers.
    """
    frames = loadAnnotatedFrames(framesDirectory)
    markerData = pd.read_csv(markersCSVPath)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    fps = 30  # Assuming 30 FPS if the original video FPS is unknown

    targetPose = None
    frameIndex = 0
    movementCommands = []

    while True:
        startTime = time.time()
        ret, currentFrame = cap.read()
        if not ret or frameIndex >= len(frames):
            break

        targetFrame = frames[frameIndex]
        frameData = markerData[markerData['Frame ID'] == frameIndex]

        closestMarker = findClosestMarker(frameData)

        currentFrame = cv2.resize(currentFrame, (targetFrame.shape[1], targetFrame.shape[0]))

        if closestMarker is not None:
            currentQR3D = convertQR3DStringToList(closestMarker['QR 3D'])
            if targetPose is None:
                targetPose = currentQR3D
            movementCommand = determineMovementCommands(currentQR3D, targetPose, currentQR3D[0], targetPose[0])
            movementCommandString = " | ".join(movementCommand)
            print(f"Frame {frameIndex + 1} Movement Command: [{movementCommandString}]")

            # Append only relevant frames
            movementCommands.append([frameIndex + 1, f"[{movementCommandString}]"])
        else:
            movementCommandString = "No Marker"

        combinedFrame = np.hstack((currentFrame, targetFrame))

        combinedHeight, combinedWidth = combinedFrame.shape[:2]
        maxWidth, maxHeight = 1920, 1080
        scaleFactor = min(maxWidth / combinedWidth, maxHeight / combinedHeight)
        newWidth = int(combinedWidth * scaleFactor)
        newHeight = int(combinedHeight * scaleFactor)
        resizedCombinedFrame = cv2.resize(combinedFrame, (newWidth, newHeight))

        cv2.putText(resizedCombinedFrame, f"Movement: [{movementCommandString}]", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Live Video', resizedCombinedFrame)

        elapsedTime = time.time() - startTime
        delay = max(1, int((1 / fps - elapsedTime) * 1000))  # Calculate delay to maintain the frame rate
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

        frameIndex += 1

    cap.release()
    cv2.destroyAllWindows()

    # Save movement commands to CSV file
    movementCommandsPath = 'MovementCommands.csv'
    movementDF = pd.DataFrame(movementCommands, columns=['Frame #', 'Movement Commands'])
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
    parser = argparse.ArgumentParser(description='Drone Alignment using Live Video Feed')
    parser.add_argument('--frames_directory', type=str, required=True,
                        help='Path to the frames directory of the annotated video')
    args = parser.parse_args()

    markersCSVPath = os.path.join(os.path.dirname(args.frames_directory), 'detected_markers.csv')

    droneAlignment(args.frames_directory, markersCSVPath)
