import cv2
import time
from ultralytics import YOLO

# Use yolov8s - better accuracy than nano model
# model = YOLO("yolov8s.pt")
model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture(0)
last_detection_time = 0
detection_interval = 1.0  # Run detection every 2 seconds
latest_annotated = None

print("Starting camera feed. Detection runs every 2 seconds.")
print("Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Resize for speed
    small = cv2.resize(frame, (512, 512))
    current_time = time.time()

    # Run detection only once per second
    if current_time - last_detection_time >= detection_interval:
        results = model(small, verbose=False)
        
        # Extract detailed object information
        detections = []
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            
            detection_info = {
                "object": class_name,
                "confidence": round(conf, 2),
                "coordinates": {
                    "x1": int(xyxy[0]),
                    "y1": int(xyxy[1]),
                    "x2": int(xyxy[2]),
                    "y2": int(xyxy[3])
                }
            }
            detections.append(detection_info)
        
        # Print the detection array
        print(f"\n[{time.strftime('%H:%M:%S')}] Detected {len(detections)} object(s):")
        for obj in detections:
            print(f"  - {obj['object']}: confidence={obj['confidence']}, "
                  f"bbox=({obj['coordinates']['x1']}, {obj['coordinates']['y1']}, "
                  f"{obj['coordinates']['x2']}, {obj['coordinates']['y2']})")
        
        # Update annotated frame for display
        latest_annotated = results[0].plot()
        last_detection_time = current_time

    # Display the latest annotated frame (or original if no detection yet)
    display_frame = latest_annotated if latest_annotated is not None else small
    cv2.imshow('Camera Feed', display_frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nCamera feed stopped.")
# 1.5 gb ram


