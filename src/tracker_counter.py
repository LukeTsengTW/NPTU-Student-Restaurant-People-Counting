# tracker_counter.py

import cv2
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict

model_path = Path(__file__).parent.parent / 'models' / 'yolov8s.pt'

# Loading model
model = YOLO(model_path)

# Open Camera
cap = cv2.VideoCapture(0)

# To save the track
track_history = defaultdict(lambda: [])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Using Tracking Mode（persist=True keep ID same）
    results = model.track(frame, classes=[0], conf=0.5, 
                          persist=True, verbose=False)
    
    # Get the detection box and tracking ID
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        # Draw the track
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            # Compute the center
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Save the track
            track = track_history[track_id]
            track.append((center_x, center_y))
            if len(track) > 30:  # Keep lastest 30 points
                track.pop(0)
            
            # Draw the track line
            points = [(int(x), int(y)) for x, y in track]
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], (0, 255, 0), 2)
        
        person_count = len(track_ids)
    else:
        person_count = 0
    
    # Draw detection box
    annotated_frame = results[0].plot()
    
    # Show the number of people
    cv2.putText(annotated_frame, f'People: {person_count}', 
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, (0, 255, 0), 3)
    
    cv2.imshow('People Tracker', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
