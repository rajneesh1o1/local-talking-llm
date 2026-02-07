import cv2
import mediapipe as mp
import math
import time
import os
import json
import logging
from collections import deque
import numpy as np
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== YOLO Object Detection Setup ==========
yolo_model = YOLO("yolov8s.pt")

# ========== MediaPipe Setup ==========
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ========== Face Recognition Setup ==========
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def load_known_faces(images_dir="../images"):
    """Load face images for comparison"""
    known_faces = []
    
    if not os.path.exists(images_dir):
        print(f"Warning: {images_dir} folder not found. Creating it...")
        os.makedirs(images_dir)
        return known_faces
    
    print("Loading known faces...")
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(filename)[0]
            filepath = os.path.join(images_dir, filename)
            
            try:
                img = cv2.imread(filepath)
                if img is None:
                    print(f"  ✗ Could not load: {filename}")
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
                    face_img = gray[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (100, 100))
                    
                    known_faces.append({
                        'name': name,
                        'face': face_img,
                        'histogram': cv2.calcHist([face_img], [0], None, [256], [0, 256])
                    })
                    print(f"  ✓ Loaded: {name}")
                else:
                    print(f"  ✗ No face found in: {filename}")
            except Exception as e:
                print(f"  ✗ Error loading {filename}: {e}")
    
    print(f"Loaded {len(known_faces)} known faces.\n")
    return known_faces

known_faces = load_known_faces()

def recognize_faces(frame, known_faces):
    """Detect and recognize faces using OpenCV"""
    if not known_faces:
        return []
    
    recognized = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (100, 100))
        face_hist = cv2.calcHist([face_img], [0], None, [256], [0, 256])
        
        best_match = None
        best_score = -1
        
        for known_face in known_faces:
            score = cv2.compareHist(face_hist, known_face['histogram'], cv2.HISTCMP_CORREL)
            if score > best_score:
                best_score = score
                best_match = known_face['name']
        
        name = best_match if best_score > 0.5 else "Unknown"
        
        recognized.append({
            "name": name,
            "location": (y, x+w, y+h, x),  # top, right, bottom, left
            "head_bbox": (x, y, x+w, y+h),  # x1, y1, x2, y2
            "confidence": best_score
        })
    
    return recognized

# ========== MediaPipe Activity Detection ==========
def dist3(a, b):
    return math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))

def dist2(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def detect_smile(face):
    lm = face.landmark
    mouth = dist2(lm[61], lm[291])
    face_w = dist2(lm[234], lm[454])
    return (mouth / face_w) > 0.38

def mouth_open(face):
    lm = face.landmark
    mouth_h = dist2(lm[13], lm[14])
    face_w = dist2(lm[234], lm[454])
    return (mouth_h / face_w) > 0.03

def head_direction(face):
    lm = face.landmark
    nose = lm[1]
    left = lm[33]
    right = lm[263]
    if nose.x < left.x:
        return "left"
    if nose.x > right.x:
        return "right"
    return "center"

WRIST = 0
FINGERS = {
    "thumb":  (4, 2),
    "index":  (8, 6),
    "middle": (12, 10),
    "ring":   (16, 14),
    "pinky":  (20, 18)
}

def finger_extended(hand, tip, pip):
    lm = hand.landmark
    return dist3(lm[tip], lm[WRIST]) > dist3(lm[pip], lm[WRIST])

def analyze_hand(hand):
    lm = hand.landmark
    fingers = {}
    for name, (tip, pip) in FINGERS.items():
        fingers[name] = finger_extended(hand, tip, pip)
    
    open_hand = sum(fingers.values()) >= 4
    fist = sum(fingers.values()) == 0
    middle_finger = (
        fingers["middle"] and
        not fingers["index"] and
        not fingers["ring"] and
        not fingers["pinky"]
    )
    
    return {
        "present": True,
        "fingers": fingers,
        "open": open_hand,
        "fist": fist,
        "middle_finger": middle_finger
    }

# ========== Shared Storage for Visual Context ==========
import threading
from datetime import datetime

VISUAL_CONTEXT_FILE = "../visual_context.json"
visual_context_lock = threading.Lock()

def save_visual_context(detections):
    """Save detection data to shared JSON file (maintains last 5 entries)"""
    try:
        with visual_context_lock:
            # Read existing data
            if os.path.exists(VISUAL_CONTEXT_FILE):
                with open(VISUAL_CONTEXT_FILE, 'r') as f:
                    data = json.load(f)
            else:
                data = {"visual_history": []}
            
            # Deduplicate detections by person name (including "Unknown")
            person_map = {}
            non_person_objects = []
            
            # for detection in detections:
            #     if detection.get("object") == "person" and detection.get("person_data"):
            #         person_data = detection["person_data"]
            #         name = person_data.get("name", "Unknown")
                    
            #         # If person with same name exists, override with latest detection
            #         person_map[name] = detection
            #     else:
            #         # Keep non-person objects as-is
            #         non_person_objects.append(detection)
            
            # # Combine deduplicated persons and other objects
            # deduplicated_detections = list(person_map.values()) + non_person_objects
            
            # Add new entry with timestamp
            entry = {
                "timestamp": datetime.now().isoformat(),
                "detections": detections
            }
            
            data["visual_history"].append(entry)
            
            # Keep only last 5 entries
            data["visual_history"] = data["visual_history"][-5:]
            
            # Write back to file
            with open(VISUAL_CONTEXT_FILE, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
    except Exception as e:
        logger.error(f"Failed to save visual context: {e}")

# ========== Video Source Configuration ==========
DRONE_IP = "192.168.1.1"
RTSP_PORT = 7070
DRONE_RTSP_URL = f"rtsp://{DRONE_IP}:{RTSP_PORT}/webcam"

# Video source: 0 = Mac camera, DRONE_RTSP_URL = drone camera
video_sources = {
    "mac": 0,
    "drone": DRONE_RTSP_URL
}
current_source = "mac"  # Start with Mac camera

def switch_video_source(cap, new_source):
    """Switch between video sources"""
    cap.release()
    time.sleep(0.2)  # Give time for release
    
    if new_source == "drone":
        new_cap = cv2.VideoCapture(DRONE_RTSP_URL, cv2.CAP_FFMPEG)
        new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for low latency
    else:
        new_cap = cv2.VideoCapture(0)
    
    return new_cap

# ========== Main Loop ==========
mouth_history = deque(maxlen=10)
hand_seen_time = {"left": 0, "right": 0}

detection_interval = 5.0  # 1 second
last_detection_time = 0
last_yolo_detections = []
last_people_data = []
last_recognized_faces = []

cap = cv2.VideoCapture(video_sources[current_source])

print("Starting combined observation system...")
print("Object Detection + Face Recognition + Activity Detection")
print("Detection interval: 1 second")
print(f"Video source: {current_source.upper()}")
print("\nControls:")
print("  [q] Quit")
print("  [c] Toggle camera (Mac ↔ Drone)")
print()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize frame for faster processing
    frame = cv2.resize(frame, (512, 512))
    current_time = time.time()
    
    # Run detections every 1 second
    if current_time - last_detection_time >= detection_interval:
        # 1. YOLO Object Detection
        yolo_results = yolo_model(frame, verbose=False)
        yolo_detections = []
        
        for box in yolo_results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = yolo_model.names[cls_id]
            
            detection_info = {
                "object": class_name,
                "confidence": round(conf, 2),
                "bbox": (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
            }
            yolo_detections.append(detection_info)
        
        # 2. Face Recognition
        last_recognized_faces = recognize_faces(frame, known_faces)
        
        # 3. MediaPipe Activity Detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        holistic_results = holistic.process(rgb)
        
        people_data = []
        for person in last_recognized_faces:
            person_data = {
                "name": person["name"],
                "face": None,
                "left_hand": None,
                "right_hand": None,
                "head_bbox": person["head_bbox"]
            }
            
            # Add face activity
            if holistic_results.face_landmarks:
                face = holistic_results.face_landmarks
                mouth_history.append(mouth_open(face))
                talking = sum(mouth_history) >= 3
                
                person_data["face"] = {
                    "smiling": detect_smile(face),
                    "talking": talking,
                    "head_direction": head_direction(face)
                }
            
            # Add hand data
            now = time.time()
            if holistic_results.left_hand_landmarks:
                person_data["left_hand"] = analyze_hand(holistic_results.left_hand_landmarks)
                hand_seen_time["left"] = now
            
            if holistic_results.right_hand_landmarks:
                person_data["right_hand"] = analyze_hand(holistic_results.right_hand_landmarks)
                hand_seen_time["right"] = now
            
            people_data.append(person_data)
        
        # If no face recognized but MediaPipe detected something
        if not people_data and (holistic_results.face_landmarks or 
                               holistic_results.left_hand_landmarks or 
                               holistic_results.right_hand_landmarks):
            unknown_person = {
                "name": "Unknown",
                "face": None,
                "left_hand": None,
                "right_hand": None,
                "head_bbox": None
            }
            
            if holistic_results.face_landmarks:
                face = holistic_results.face_landmarks
                mouth_history.append(mouth_open(face))
                talking = sum(mouth_history) >= 3
                unknown_person["face"] = {
                    "smiling": detect_smile(face),
                    "talking": talking,
                    "head_direction": head_direction(face)
                }
            
            now = time.time()
            if holistic_results.left_hand_landmarks:
                unknown_person["left_hand"] = analyze_hand(holistic_results.left_hand_landmarks)
            if holistic_results.right_hand_landmarks:
                unknown_person["right_hand"] = analyze_hand(holistic_results.right_hand_landmarks)
            
            people_data.append(unknown_person)
        
        # 4. Combine detections: Replace "person" from YOLO with detailed people_data
        combined_detections = []
        person_found = False
        
        for detection in yolo_detections:
            if detection["object"] == "person":
                # Replace with detailed person data
                if people_data:
                    for person in people_data:
                        combined_detections.append({
                            "object": "person",
                            "person_data": person,
                            "confidence": detection["confidence"],
                            "bbox": detection["bbox"]
                        })
                    person_found = True
                else:
                    # Keep YOLO person detection if no face recognized
                    combined_detections.append({
                        "object": "person",
                        "person_data": None,
                        "confidence": detection["confidence"],
                        "bbox": detection["bbox"]
                    })
            else:
                # Keep non-person objects
                combined_detections.append(detection)
        
        # If people detected but YOLO didn't detect person class, add them
        if people_data and not person_found:
            for person in people_data:
                bbox = person.get("head_bbox")
                if bbox:
                    # Expand bbox for full body estimate
                    x1, y1, x2, y2 = bbox
                    body_y2 = min(512, y2 + (y2 - y1) * 3)  # Estimate body height
                    combined_detections.append({
                        "object": "person",
                        "person_data": person,
                        "confidence": 0.30,
                        "bbox": (x1, y1, x2, body_y2)
                    })
        
        # 5. Print combined output
        print(f"\n[{time.strftime('%H:%M:%S')}] Detected {len(combined_detections)} object(s):")
        for detection in combined_detections:
            if detection["object"] == "person" and detection.get("person_data"):
                person = detection["person_data"]
                bbox = detection["bbox"]
                print(f"  - person: {person} confidence={detection['confidence']}, "
                      f"bbox={bbox}")
            else:
                bbox = detection["bbox"]
                print(f"  - {detection['object']}: confidence={detection['confidence']}, "
                      f"bbox={bbox}")
        
        # 6. Save to shared storage for LLM context
        save_visual_context(combined_detections)
        
        last_yolo_detections = yolo_detections
        last_people_data = people_data
        last_detection_time = current_time
    
    # Draw visualizations
    # Draw person detections with names
    for person in last_recognized_faces:
        top, right, bottom, left = person["location"]
        color = (0, 255, 0) if person["name"] != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        label = f"{person['name']}"
        cv2.putText(frame, label, (left, top - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Draw YOLO detections for non-person objects
    for detection in last_yolo_detections:
        if detection["object"] != "person":
            x1, y1, x2, y2 = detection["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{detection['object']} {detection['confidence']}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Add source indicator on frame
    source_text = f"Source: {current_source.upper()}"
    cv2.putText(frame, source_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow("Observer - Combined Detection", frame)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        # Toggle camera source
        current_source = "drone" if current_source == "mac" else "mac"
        print(f"\n🔄 Switching to {current_source.upper()} camera...")
        cap = switch_video_source(cap, current_source)
        if cap.isOpened():
            print(f"✓ Connected to {current_source.upper()} camera")
        else:
            print(f"✗ Failed to connect to {current_source.upper()} camera, reverting...")
            current_source = "drone" if current_source == "mac" else "mac"
            cap = switch_video_source(cap, current_source)
        print()

cap.release()
cv2.destroyAllWindows()
print("\nObservation stopped.")

