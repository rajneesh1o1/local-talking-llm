# import cv2
# import mediapipe as mp
# import math
# import time

# mp_holistic = mp.solutions.holistic
# holistic = mp_holistic.Holistic(
#     static_image_mode=False,
#     model_complexity=1,          # 0 = lighter, 1 = balanced
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# def distance(p1, p2):
#     return math.hypot(p1.x - p2.x, p1.y - p2.y)

# def detect_face_activity(face):
#     if not face:
#         return {}

#     lm = face.landmark

#     mouth_open = distance(lm[13], lm[14]) > 0.015
#     smile = distance(lm[61], lm[291]) > 0.06

#     return {
#         "smiling": smile,
#         "talking": mouth_open
#     }

# def detect_head_direction(face):
#     if not face:
#         return "unknown"

#     nose = face.landmark[1]
#     left_eye = face.landmark[33]
#     right_eye = face.landmark[263]

#     if nose.x < left_eye.x:
#         return "looking left"
#     elif nose.x > right_eye.x:
#         return "looking right"
#     return "looking center"

# def detect_hand_activity(hand):
#     if not hand:
#         return None
#     return "hand detected"

# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = holistic.process(rgb)

#     activity = {}

#     # Face
#     face = results.face_landmarks
#     activity.update(detect_face_activity(face))
#     activity["head"] = detect_head_direction(face)

#     # Hands
#     if results.left_hand_landmarks:
#         activity["left_hand"] = "active"
#     if results.right_hand_landmarks:
#         activity["right_hand"] = "active"

#     print(activity)

#     cv2.imshow("Camera", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()





# import cv2
# import mediapipe as mp
# import math
# from collections import deque

# mp_holistic = mp.solutions.holistic
# holistic = mp_holistic.Holistic(
#     static_image_mode=False,
#     model_complexity=1,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# # ---------- helpers ----------
# def dist(a, b):
#     return math.hypot(a.x - b.x, a.y - b.y)

# def ratio(a, b, c):
#     return dist(a, b) / (dist(b, c) + 1e-6)

# # temporal buffers
# mouth_history = deque(maxlen=10)
# head_x_history = deque(maxlen=10)
# wrist_x_history = deque(maxlen=15)

# # ---------- face ----------
# def analyze_face(face):
#     if not face:
#         return {}

#     lm = face.landmark

#     # Mouth
#     mouth_open = dist(lm[13], lm[14]) > 0.015
#     smile = dist(lm[61], lm[291]) / dist(lm[0], lm[17]) > 0.45

#     mouth_history.append(mouth_open)
#     talking = sum(mouth_history) > 3

#     # Eyes
#     left_eye_open = dist(lm[159], lm[145]) > 0.008
#     right_eye_open = dist(lm[386], lm[374]) > 0.008
#     blinking = not left_eye_open or not right_eye_open

#     # Head direction
#     nose = lm[1]
#     left_eye = lm[33]
#     right_eye = lm[263]

#     head_dir = "center"
#     if nose.x < left_eye.x:
#         head_dir = "left"
#     elif nose.x > right_eye.x:
#         head_dir = "right"

#     return {
#         "smiling": smile,
#         "talking": talking,
#         "blinking": blinking,
#         "head_direction": head_dir
#     }

# # ---------- pose ----------
# def analyze_pose(pose):
#     if not pose:
#         return {}

#     lm = pose.landmark
#     nose = lm[0]
#     shoulders_mid = lm[11].x + lm[12].x

#     head_x_history.append(nose.x)

#     nodding = max(head_x_history) - min(head_x_history) < 0.01
#     leaning = "forward" if nose.z < -0.15 else "neutral"

#     return {
#         "nodding": nodding,
#         "leaning": leaning
#     }

# # ---------- hands ----------
# def analyze_hand(hand, label):
#     if not hand:
#         return None

#     lm = hand.landmark

#     # Finger states
#     fingers = {
#         "thumb": dist(lm[4], lm[2]) > 0.04,
#         "index": dist(lm[8], lm[6]) > 0.04,
#         "middle": dist(lm[12], lm[10]) > 0.04,
#         "ring": dist(lm[16], lm[14]) > 0.04,
#         "pinky": dist(lm[20], lm[18]) > 0.04
#     }

#     open_hand = sum(fingers.values()) >= 3
#     fist = sum(fingers.values()) <= 1
#     pointing = fingers["index"] and not fingers["middle"]

#     wrist_x_history.append(lm[0].x)
#     waving = max(wrist_x_history) - min(wrist_x_history) > 0.1

#     return {
#         "present": True,
#         "open": open_hand,
#         "fist": fist,
#         "pointing": pointing,
#         "waving": waving,
#         "fingers": fingers
#     }

# # ---------- main ----------
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = holistic.process(rgb)

#     state = {
#         "face": analyze_face(results.face_landmarks),
#         "pose": analyze_pose(results.pose_landmarks),
#         "left_hand": analyze_hand(results.left_hand_landmarks, "left"),
#         "right_hand": analyze_hand(results.right_hand_landmarks, "right")
#     }

#     print(state)

#     cv2.imshow("Camera", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




import cv2
import mediapipe as mp
import math
import time
from collections import deque

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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

# ---------- MAIN ----------
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    output = {}

    # ----- FACE -----
    if results.face_landmarks:
        face = results.face_landmarks

        mouth_history.append(mouth_open(face))
        talking = sum(mouth_history) >= 3

        output["face"] = {
            "smiling": detect_smile(face),
            "talking": talking,
            "head_direction": head_direction(face)
        }
    else:
        output["face"] = None

    now = time.time()

    # ----- LEFT HAND -----
    if results.left_hand_landmarks:
        output["left_hand"] = analyze_hand(results.left_hand_landmarks)
        hand_seen_time["left"] = now
    else:
        output["left_hand"] = (
            "lost" if now - hand_seen_time["left"] < 0.3 else None
        )

    # ----- RIGHT HAND -----
    if results.right_hand_landmarks:
        output["right_hand"] = analyze_hand(results.right_hand_landmarks)
        hand_seen_time["right"] = now
    else:
        output["right_hand"] = (
            "lost" if now - hand_seen_time["right"] < 0.3 else None
        )

    print(output)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

