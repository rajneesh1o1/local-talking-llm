import cv2
import mediapipe as mp
import math
import time
import os
from collections import deque
import numpy as np

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ---------- Load Known Faces using OpenCV ----------
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
            name = os.path.splitext(filename)[0]  # Remove extension
            filepath = os.path.join(images_dir, filename)
            
            try:
                # Load image
                img = cv2.imread(filepath)
                if img is None:
                    print(f"  ✗ Could not load: {filename}")
                    continue
                
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Detect face
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    # Get the largest face
                    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
                    face_img = gray[y:y+h, x:x+w]
                    # Resize to standard size for comparison
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

# Load known faces at startup
known_faces = load_known_faces()

# ---------- Face Recognition using OpenCV ----------
def recognize_faces(frame, known_faces):
    """Detect and recognize faces using OpenCV"""
    if not known_faces:
        return []
    
    recognized = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (100, 100))
        face_hist = cv2.calcHist([face_img], [0], None, [256], [0, 256])
        
        # Compare with known faces using histogram correlation
        best_match = None
        best_score = -1
        
        for known_face in known_faces:
            # Compare histograms
            score = cv2.compareHist(face_hist, known_face['histogram'], cv2.HISTCMP_CORREL)
            
            if score > best_score:
                best_score = score
                best_match = known_face['name']
        
        # If similarity is high enough, recognize the face
        name = best_match if best_score > 0.5 else "Unknown"
        
        recognized.append({
            "name": name,
            "location": (y, x+w, y+h, x),  # top, right, bottom, left
            "confidence": best_score
        })
    
    return recognized

# ---------- Utils ----------
def dist3(a, b):
    return math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))

def dist2(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

# ---------- FACE ----------
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

# ---------- HAND ----------
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

# ---------- TEMPORAL BUFFERS ----------
mouth_history = deque(maxlen=10)
hand_seen_time = {"left": 0, "right": 0}

# Face recognition throttle (run every N frames to save CPU)
face_recog_interval = 30  # Every 30 frames (~1 second at 30fps)
frame_count = 0
last_recognized_faces = []

# ---------- MAIN ----------
cap = cv2.VideoCapture(0)

print("Starting camera with face recognition...")
print("Press 'q' to quit.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    # Resize frame for faster processing
    frame = cv2.resize(frame, (512, 512))
    
    # Run face recognition periodically
    if frame_count % face_recog_interval == 0 and known_faces:
        last_recognized_faces = recognize_faces(frame, known_faces)
        if last_recognized_faces:
            names = [p['name'] for p in last_recognized_faces]
            print(f"[Face Recognition] Detected: {names}")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    # ----- BUILD OUTPUT FOR EACH PERSON -----
    people = []
    
    for person in last_recognized_faces:
        person_data = {
            "name": person["name"],
            "face": None,
            "left_hand": None,
            "right_hand": None
        }
        
        # If MediaPipe detected face landmarks, add activity data
        if results.face_landmarks:
            face = results.face_landmarks
            mouth_history.append(mouth_open(face))
            talking = sum(mouth_history) >= 3

            person_data["face"] = {
                "smiling": detect_smile(face),
                "talking": talking,
                "head_direction": head_direction(face)
            }
        
        # Add hand data
        now = time.time()
        
        if results.left_hand_landmarks:
            person_data["left_hand"] = analyze_hand(results.left_hand_landmarks)
            hand_seen_time["left"] = now
        
        if results.right_hand_landmarks:
            person_data["right_hand"] = analyze_hand(results.right_hand_landmarks)
            hand_seen_time["right"] = now
        
        people.append(person_data)
    
    # If no one recognized but MediaPipe detected something, add as "Unknown"
    if not people and (results.face_landmarks or results.left_hand_landmarks or results.right_hand_landmarks):
        unknown_person = {
            "name": "Unknown",
            "face": None,
            "left_hand": None,
            "right_hand": None
        }
        
        if results.face_landmarks:
            face = results.face_landmarks
            mouth_history.append(mouth_open(face))
            talking = sum(mouth_history) >= 3
            unknown_person["face"] = {
                "smiling": detect_smile(face),
                "talking": talking,
                "head_direction": head_direction(face)
            }
        
        now = time.time()
        if results.left_hand_landmarks:
            unknown_person["left_hand"] = analyze_hand(results.left_hand_landmarks)
        if results.right_hand_landmarks:
            unknown_person["right_hand"] = analyze_hand(results.right_hand_landmarks)
        
        people.append(unknown_person)
    
    # Output format: list of people with their activities
    if people and frame_count % 30 == 0:  # Print every second
        print(people)

    # Draw rectangles around recognized faces
    for person in last_recognized_faces:
        top, right, bottom, left = person["location"]
        color = (0, 255, 0) if person["name"] != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Draw name label with background
        label = f"{person['name']} ({person['confidence']:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (left, top - 25), (left + label_size[0], top), color, -1)
        cv2.putText(frame, label, (left, top - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Camera - Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nCamera stopped.")
