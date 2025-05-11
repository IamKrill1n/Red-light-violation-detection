import tarfile
import sys
sys.modules["backports.tarfile"] = tarfile

import cv2
import torch

# Add local YOLOv5 repository to path
sys.path.insert(0, 'd:/dev/project/Red-light-violation-detection/yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.models.common import AutoShape
# Choose device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model using YOLOv5's attempt_load (do not use map_location; use device keyword)
model = attempt_load('best.pt', device=device)
model = AutoShape(model)  # Convert model to Autoshape
model.eval()

# Define the line coordinates (e.g., a horizontal line in the middle of the frame)
# You'll need to get the frame height first, which can be done after reading the first frame,
# or you can set a fixed y-coordinate if you know your video dimensions.
line_y_coordinate = None

# Open the video file
cap = cv2.VideoCapture('video7.avi')  # Replace with your video path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Initialize line_y_coordinate with frame height if not already set
    if line_y_coordinate is None:
        frame_height = frame.shape[0]
        line_y_coordinate = frame_height // 3  # Example: horizontal line in the middle

    # Draw the defined line on the frame
    cv2.line(frame, (0, line_y_coordinate), (frame.shape[1], line_y_coordinate), (0, 0, 255), 2)  # Red line

    # Convert frame to RGB as required
    rgb_frame = frame[:, :, ::-1]

    # YOLOv5 expects a list of images; autoshape now handles preprocessing
    results = model([rgb_frame])
    
    # Assuming single image in the batch; results.xyxy is a list per image.
    detections = results.xyxy[0].cpu().numpy()
    
    # Draw bounding boxes for detected vehicles
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        
        # Check for intersection with the horizontal line
        # A simple check: if the line's y-coordinate is between the top (y1) and bottom (y2) of the box
        intersected = False
        if y1 < line_y_coordinate < y2:
            intersected = True
            # Optionally, change color or add text for intersected boxes
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for intersection
            cv2.putText(frame, "INTERSECTED", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box otherwise
        
        label = f"{conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # Display the frame
    cv2.imshow('Vehicle Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()