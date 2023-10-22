import cv2
import matplotlib.pyplot as plt

# Initialize the video capture object
video_capture = cv2.VideoCapture(0)  # Use 0 for the default camera or provide the video file path

# Read the first frame
ret, frame = video_capture.read()

# Initialize the selected object bounding box
bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

# Convert the selected bounding box to tuple format (x, y, w, h)
bbox = tuple(map(int, bbox))

# Initialize the tracker
tracker = cv2.legacy_TrackerMOSSE.create()
tracker.init(frame, bbox)

# Initialize variables for tracking direction
prev_center = None
direction = ""

while True:
    # Read a new frame
    ret, frame = video_capture.read()

    # Break the loop if the video stream has ended
    if not ret:
        break

    # Update the tracker
    success, new_bbox = tracker.update(frame)

    if success:
        # Calculate the new object center
        center = (int(new_bbox[0] + new_bbox[2] / 2), int(new_bbox[1] + new_bbox[3] / 2))

        # Draw a bounding box and object center on the frame
        cv2.rectangle(frame, (int(new_bbox[0]), int(new_bbox[1])), (int(new_bbox[0] + new_bbox[2]), int(new_bbox[1] + new_bbox[3])), (0, 255, 0), 2)
        cv2.circle(frame, center, 2, (0, 255, 0), -1)

        # Check if it's the first frame or the object center has changed
        if prev_center is not None and center != prev_center:
            # Determine the direction of movement
            if center[0] > prev_center[0]:
                direction = "Right"
            elif center[0] < prev_center[0]:
                direction = "Left"
            else:
                direction = "No movement"

        prev_center = center

    # Display the frame
    cv2.imshow("Object Tracking", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
video_capture.release()
cv2.destroyAllWindows()
