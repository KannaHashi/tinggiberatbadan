#https://pysource.com
import cv2
from realsense_camera import *
import cv2
from mask_rcnn import *

# Load Mask RCNN model
mrcnn = MaskRCNN()

# Open the default camera (camera index 0)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, bgr_frame = cap.read()

    # Get object mask
    boxes, classes, contours, centers = mrcnn.detect_objects_mask(bgr_frame)

    # Draw object mask
    bgr_frame = mrcnn.draw_object_mask(bgr_frame)

    # Show depth info of the objects (commented out, as depth_frame is not available with cv2.VideoCapture(0))
    # mrcnn.draw_object_info(bgr_frame, depth_frame)

    # Display the frames
    cv2.imshow("Bgr frame", bgr_frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

# Release the camera capture object
cap.release()
cv2.destroyAllWindows()
