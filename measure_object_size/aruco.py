import cv2
import cv2.aruco as aruco

# Generate ARUCO markers
def generate_markers(output_dir, marker_size=200, num_markers=5):
    for i in range(num_markers):
        marker_id = i
        marker = aruco.drawMarker(aruco.Dictionary_get(aruco.DICT_4X4_50), marker_id, marker_size)
        marker_filename = f"{output_dir}/marker_{marker_id}.png"
        cv2.imwrite(marker_filename, marker)

# Detect ARUCO markers from webcam
def detect_markers():
    cap = cv2.VideoCapture(0)  # 0 indicates the default camera

    while True:
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Define the dictionary and parameters for detection
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters_create()

        # Detect markers
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # Draw markers on the frame
        aruco.drawDetectedMarkers(frame, corners, ids)

        # Display the resulting frame
        cv2.imshow('ARUCO Marker Detection', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

detect_markers()
