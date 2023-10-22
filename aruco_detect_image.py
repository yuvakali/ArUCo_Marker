import cv2
from cv2 import aruco
import matplotlib.pyplot as plt

def detect_markers(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    return corners, ids

def plot_image_with_markers(image, corners, ids):
    image_with_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)
    plt.imshow(cv2.cvtColor(image_with_markers, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def track_marker_movement(prev_corners, curr_corners):
    if prev_corners is None or curr_corners is None:
        return "No movement"

    prev_center = prev_corners.mean(axis=1).squeeze()
    curr_center = curr_corners.mean(axis=1).squeeze()

    x_diff = curr_center[0] - prev_center[0]
    y_diff = curr_center[1] - prev_center[1]

    if abs(x_diff) > abs(y_diff):
        if x_diff > 0:
            return "Move right"
        else:
            return "Move left"
    else:
        if y_diff > 0:
            return "Move down"
        else:
            return "Move up"

# Example usage
prev_frame = None

# Capture video from a source or load images
cap = cv2.VideoCapture(0)  # Uncomment this line to capture from webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Adjust width if needed
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Adjust height if needed

while True:
    ret, frame = cap.read()  # Uncomment this line to capture from webcam
    #frame = cv2.imread("image.jpg")  # Replace "image.jpg" with your image path

    corners, ids = detect_markers(frame)
    if ids is not None:
        if prev_frame is not None:
            prev_corners, _ = detect_markers(prev_frame)
            movement_direction = track_marker_movement(prev_corners, corners)
            print("Movement Direction:", movement_direction)

        prev_frame = frame.copy()
        plot_image_with_markers(frame, corners, ids)

    # Uncomment the following line and 'break' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
         break

# cap.release()  # Uncomment this line to release webcam capture
cv2.destroyAllWindows()
