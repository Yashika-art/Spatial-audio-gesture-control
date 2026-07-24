"""
Step 7: ML-powered gesture control
Replaces the hard-coded pinch-distance threshold with a trained ML
classifier (gesture_model.pkl from Day 2) that recognizes:
  fist, open_palm, peace_sign, pinch, point

Gesture -> action mapping (change these however you like):
  fist / pinch  -> mute
  open_palm     -> unmute (full sound)
  point         -> unmute (no special action yet, reserved for future use)
  peace_sign    -> unmute (also reserved for a future feature, e.g.
                   switching sound source or instrument)

Audio engine (pan/elevation/distance) is unchanged from Step 6.

Press 'q' to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import threading
import math
import pandas as pd
import joblib
from collections import deque, Counter

MODEL_FILE = "gesture_model.pkl"

# ---------- Load trained gesture classifier ----------
print(f"Loading gesture model from {MODEL_FILE}...")
gesture_model = joblib.load(MODEL_FILE)

# Must match the exact column order used during training in
# train_gesture_model.py (x0,y0,z0, x1,y1,z1, ... x20,y20,z20)
FEATURE_COLUMNS = []
for i in range(21):
    FEATURE_COLUMNS += [f"x{i}", f"y{i}", f"z{i}"]

# Gestures that should mute the audio.
MUTE_GESTURES = {"fist", "pinch"}

# ---------- Audio setup ----------
SAMPLE_RATE = 44100
FREQUENCY = 330.0
BLOCK_SIZE = 512

pan_position = 0.5
elevation_position = 0.5
distance_volume = 0.5
is_muted = False
state_lock = threading.Lock()

phase = 0.0
filter_state_l = 0.0
filter_state_r = 0.0


def audio_callback(outdata, frames, time_info, status):
    global phase, filter_state_l, filter_state_r
    if status:
        print(status)

    t = (np.arange(frames) + phase) / SAMPLE_RATE
    phase = (phase + frames) % SAMPLE_RATE
    tone = 0.18 * np.sin(2 * np.pi * FREQUENCY * t)

    with state_lock:
        pan = pan_position
        elev = elevation_position
        vol = distance_volume
        muted = is_muted

    if muted:
        vol = 0.0

    theta = pan * (np.pi / 2)
    left_gain = np.cos(theta) * vol
    right_gain = np.sin(theta) * vol

    left = tone * left_gain
    right = tone * right_gain

    alpha = 0.05 + elev * 0.95

    filtered_l = np.empty_like(left)
    filtered_r = np.empty_like(right)
    fl, fr = filter_state_l, filter_state_r
    for i in range(len(left)):
        fl = fl + alpha * (left[i] - fl)
        fr = fr + alpha * (right[i] - fr)
        filtered_l[i] = fl
        filtered_r[i] = fr

    if not (np.isfinite(fl) and np.isfinite(fr)):
        fl, fr = 0.0, 0.0
    filter_state_l, filter_state_r = fl, fr

    outdata[:, 0] = np.clip(np.nan_to_num(filtered_l), -1.0, 1.0)
    outdata[:, 1] = np.clip(np.nan_to_num(filtered_r), -1.0, 1.0)


stream = sd.OutputStream(
    samplerate=SAMPLE_RATE,
    blocksize=BLOCK_SIZE,
    channels=2,
    callback=audio_callback,
)
stream.start()

# ---------- Hand tracking setup ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam.")
    exit()

print("Running with ML gesture recognition. Press 'q' to quit.")

smoothed_x = 0.5
smoothed_y = 0.5
smoothed_size = 0.15
SMOOTHING = 0.2

# Majority-vote smoothing over recent gesture predictions, same idea as
# the pinch smoothing in Step 6 -- prevents a single noisy frame from
# flickering the detected gesture.
gesture_history = deque(maxlen=7)

MIN_SPAN = 0.08
MAX_SPAN = 0.35

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_bar(frame, label, value, x, y, w=200, h=20, color=(0, 255, 0)):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), 1)
    fill_w = int(w * max(0.0, min(1.0, value)))
    cv2.rectangle(frame, (x, y), (x + fill_w, y + h), color, -1)
    cv2.putText(frame, f"{label}: {value:.2f}", (x, y - 6), FONT, 0.5, (255, 255, 255), 1)


def draw_radar(frame, pan, distance_vol, muted, cx, cy, radius=80):
    cv2.circle(frame, (cx, cy), radius, (60, 60, 60), 2)
    cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)
    cv2.putText(frame, "YOU", (cx - 15, cy + radius + 20), FONT, 0.5, (200, 200, 200), 1)

    dx = (pan - 0.5) * 2 * radius
    dist_from_center = radius * (1.0 - distance_vol)
    dot_x = int(cx + dx * (dist_from_center / radius) if radius > 0 else cx)
    dot_y = int(cy - dist_from_center * 0.3)

    dot_color = (0, 0, 255) if muted else (0, 255, 120)
    cv2.circle(frame, (dot_x, dot_y), 8, dot_color, -1)
    cv2.line(frame, (cx, cy), (dot_x, dot_y), dot_color, 1)


def predict_gesture(hand_landmarks):
    """
    Extracts landmarks, normalizes relative to the wrist (matching the
    exact preprocessing used in train_gesture_model.py), and returns the
    predicted gesture label.
    """
    raw = []
    for lm in hand_landmarks.landmark:
        raw += [lm.x, lm.y, lm.z]

    wrist_x, wrist_y, wrist_z = raw[0], raw[1], raw[2]
    normalized = []
    for i in range(21):
        normalized += [
            raw[i * 3] - wrist_x,
            raw[i * 3 + 1] - wrist_y,
            raw[i * 3 + 2] - wrist_z,
        ]

    features_df = pd.DataFrame([normalized], columns=FEATURE_COLUMNS)
    prediction = gesture_model.predict(features_df)[0]
    return prediction


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    raw_span = None
    stable_gesture = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            wrist = hand_landmarks.landmark[0]
            middle_tip = hand_landmarks.landmark[12]

            wrist_x = max(0.0, min(1.0, wrist.x))
            wrist_y_inverted = max(0.0, min(1.0, 1.0 - wrist.y))

            # ML gesture prediction, replacing the old distance threshold.
            predicted = predict_gesture(hand_landmarks)
            gesture_history.append(predicted)
            # Majority vote across recent frames for stability.
            stable_gesture = Counter(gesture_history).most_common(1)[0][0]

            raw_span = math.hypot(middle_tip.x - wrist.x, middle_tip.y - wrist.y)
            span_normalized = (raw_span - MIN_SPAN) / (MAX_SPAN - MIN_SPAN)
            span_normalized = max(0.0, min(1.0, span_normalized))

            smoothed_x = SMOOTHING * wrist_x + (1 - SMOOTHING) * smoothed_x
            smoothed_y = SMOOTHING * wrist_y_inverted + (1 - SMOOTHING) * smoothed_y
            smoothed_size = SMOOTHING * span_normalized + (1 - SMOOTHING) * smoothed_size

            with state_lock:
                pan_position = smoothed_x
                elevation_position = smoothed_y
                distance_volume = smoothed_size
                is_muted = stable_gesture in MUTE_GESTURES

    # ---------- On-screen overlay ----------
    draw_bar(frame, "Pan (L-R)", smoothed_x, 20, 40, color=(255, 120, 0))
    draw_bar(frame, "Elevation", smoothed_y, 20, 90, color=(0, 200, 255))
    draw_bar(frame, "Distance/Vol", smoothed_size, 20, 140, color=(0, 255, 120))

    with state_lock:
        muted_for_radar = is_muted
    frame_h, frame_w, _ = frame.shape
    draw_radar(frame, smoothed_x, smoothed_size, muted_for_radar,
               cx=frame_w - 120, cy=100, radius=70)

    if stable_gesture is not None:
        cv2.putText(frame, f"Gesture: {stable_gesture}", (20, 180),
                    FONT, 0.7, (255, 255, 0), 2)

    with state_lock:
        muted_display = is_muted
    status_text = "MUTED" if muted_display else "SOUND ON"
    status_color = (0, 0, 255) if muted_display else (0, 255, 0)
    cv2.putText(frame, status_text, (20, 220), FONT, 0.7, status_color, 2)

    cv2.imshow("ML Gesture-Controlled Spatial Audio - press q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
stream.stop()
stream.close()