import cv2
import numpy as np

# Initialize camera and ArUco marker parameters
camera = cv2.VideoCapture(1)
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
aruco_params = cv2.aruco.DetectorParameters_create()

# Define marker size (in meters) and desired distance for docking
marker_size = 0.1
desired_distance = 100  # in centimeters

# Main loop
while True:
    # Capture frame from camera
    ret, frame = camera.read()

    # Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

    # Check if any markers are found
    if ids is not None:
        # Initialize movement direction
        movement_direction = ""

        # Iterate over detected markers
        for i in range(len(ids)):
            # Get corner coordinates of the marker
            marker_corners = corners[i][0]

            # Calculate marker center
            marker_center = np.mean(marker_corners, axis=0)

            # Calculate marker direction from center of the frame
            frame_center = np.array([frame.shape[1] / 2, frame.shape[0] / 2])
            marker_direction = marker_center - frame_center

            # Normalize marker direction
            marker_direction /= np.linalg.norm(marker_direction)

            # Convert marker direction to centimeters
            marker_direction_cm = marker_direction * desired_distance

            # Display marker direction
            print("Marker direction (cm): ", marker_direction_cm)

            # Check if the camera moved
            if np.linalg.norm(marker_direction) > 0.1:
                print("Camera moved. Adjusting trajectory...")

                # Calculate the movement direction
                if marker_direction_cm[0] > 0:
                    movement_direction += "Right "
                elif marker_direction_cm[0] < 0:
                    movement_direction += "Left "

                if marker_direction_cm[1] > 0:
                    movement_direction += "Down"
                elif marker_direction_cm[1] < 0:
                    movement_direction += "Up"

                # Calculate the amount of movement in the frame
                movement_amount = np.linalg.norm(marker_direction_cm)

                # TODO: Implement navigation code to adjust trajectory
                # Example: adjust navigation based on desired position

        # Display movement information on the frame
        cv2.putText(frame, f"Movement: {movement_direction} ", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display frame with detected markers
    frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.imshow("ArUco Marker Detection", frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
camera.release()
cv2.destroyAllWindows()
