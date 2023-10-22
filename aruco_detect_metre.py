import cv2
import numpy as np
import math

# Initialize camera or space image source
camera = cv2.VideoCapture(1)

# ArUco dictionary and parameters
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
aruco_params = cv2.aruco.DetectorParameters_create()

# Initialize the PID controller
Kp = 0.1  # Proportional gain
Kd = 0.01  # Derivative gain
Ki = 0.005  # Integral gain
integral = 0  # Integral term
previous_error = 0  # Previous error term

# Size of the ArUco marker in meters
marker_size = 0.1  # Assuming the marker is 10 cm in size

# Focal length of the camera (in pixels)
focal_length = 500  # Adjust this based on your camera's specifications

# Main loop
while True:
    # Capture frame from camera or space image source
    ret, frame = camera.read()

    # Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

    # If any markers are detected
    if ids is not None:
        # Calculate the centroid of the first marker
        centroid = np.mean(corners[0][0], axis=0)
        cx, cy = centroid[0], centroid[1]

        # Calculate the error terms for PID control
        error = cx - frame.shape[1] / 2  # Error is the difference between the centroid and the center of the frame
        derivative = error - previous_error
        integral += error

        # Apply PID control to steer the satellite
        output = Kp * error + Kd * derivative + Ki * integral

        # Calculate the distance to the marker
        marker_pixels = np.sqrt(np.sum((corners[0][0][0] - corners[0][0][2]) ** 2))
        distance = (marker_size * focal_length) / marker_pixels

        # Print the distance (change this to control the satellite)
        print("Distance:", distance, "meters")

        # Update the previous error term
        previous_error = error

        # Draw the marker and centroid on the frame
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)

    # Display the frame
    cv2.imshow("ArUco Marker Detection", frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera or space image source
camera.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
