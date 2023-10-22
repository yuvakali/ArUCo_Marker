import cv2
import numpy as np

# Load ArUco detector
parameters = cv2.aruco.DetectorParameters_create()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)

# Load Cap
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    _, img = cap.read()

    # Get ArUco marker
    corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

    if ids is not None:
        # Iterate over detected markers
        for i in range(len(ids)):
            # Get corner coordinates of the marker
            marker_corners = corners[i][0]

            # Calculate marker size
            marker_size_pixels = np.linalg.norm(marker_corners[0] - marker_corners[1])
            marker_size_cm = marker_size_pixels / 20.0  # Assuming marker size in cm is 20

            # Display marker ID and size in cm
            cv2.putText(img, f"Marker ID: {ids[i][0]}", (int(marker_corners[0][0]), int(marker_corners[0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Width: {marker_size_cm:.2f} cm", (int(marker_corners[0][0]), int(marker_corners[0][1]) + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Height: {marker_size_cm:.2f} cm", (int(marker_corners[0][0]), int(marker_corners[0][1]) + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw polygon around the marker
        cv2.aruco.drawDetectedMarkers(img, corners, ids)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
