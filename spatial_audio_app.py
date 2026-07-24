"""
Step 8: Two-hand, two-source spatial audio
Each hand is now a fully independent sound source:
  - Its own pan (left-right), elevation (brightness), distance (volume)
  - Its own ML gesture recognition controlling its own mute state

The two sources use slightly different pitches (a musical fifth apart)
so you can tell them apart by ear, which also makes the "two independent
sources" effect obvious in a demo.

Hands are identified as "Left"/"Right" using MediaPipe's handedness
classification (based on the mirrored/flipped view you see on screen),
which stays consistent frame-to-frame -- more reliable than just using
detection order, which can swap between hands.

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

print(f"Loading gesture model from {MODEL_FILE}...")
gesture_model = joblib.load(MODEL_FILE)

FEATURE_COLUMNS = []
for i in range(21):
    FEATURE_COLUMNS += [f"x{i}", f"y{i}", f"z{i}"]

MUTE_GESTURES = {"fist", "pinch"}

# ---------- Audio setup ----------
SAMPLE_RATE = 44100
BLOCK_SIZE = 512

# Wider pitch separation (two octaves + a bit) AND different timbre per
# source -- pitch alone wasn't distinct enough. Left = warm sine-based
# tone, Right = brighter buzzy tone (extra harmonics), so they're
# unmistakable by ear even when both play together.
FREQ_LEFT = 220.0   # A3
FREQ_RIGHT = 660.0  # E5

state_lock = threading.Lock()

# Per-hand state, keyed by "Left" / "Right".
hand_state = {
    "Left": {"pan": 0.5, "elevation": 0.5, "distance": 0.5, "muted": True},
    "Right": {"pan": 0.5, "elevation": 0.5, "distance": 0.5, "muted": True},
}
# Start muted until a hand is actually detected, so silence plays if
# no hand is in frame instead of a default center tone.

phase_l = 0.0
phase_r = 0.0
filter_state = {
    "Left": {"l": 0.0, "r": 0.0},
    "Right": {"l": 0.0, "r": 0.0},
}


def synth_source(hand_key, frequency, frames, t, phase_local):
    """Generates one source's filtered stereo signal for this audio block."""
    with state_lock:
        s = hand_state[hand_key].copy()

    vol = 0.0 if s["muted"] else s["distance"]

    if hand_key == "Left":
        # Warm, simple sine -- soft and round sounding.
        tone = 0.15 * np.sin(2 * np.pi * frequency * t)
    else:
        # Brighter, buzzier tone: fundamental + a couple of harmonics,
        # like a soft square-ish wave. Clearly distinct from the Left
        # hand's plain sine even at a similar loudness.
        tone = (
            0.11 * np.sin(2 * np.pi * frequency * t) +
            0.05 * np.sin(2 * np.pi * frequency * 3 * t) +
            0.03 * np.sin(2 * np.pi * frequency * 5 * t)
        )

    theta = s["pan"] * (np.pi / 2)
    left = tone * np.cos(theta) * vol
    right = tone * np.sin(theta) * vol

    alpha = 0.05 + s["elevation"] * 0.95
    fl, fr = filter_state[hand_key]["l"], filter_state[hand_key]["r"]

    filtered_l = np.empty_like(left)
    filtered_r = np.empty_like(right)
    for i in range(len(left)):
        fl = fl + alpha * (left[i] - fl)
        fr = fr + alpha * (right[i] - fr)
        filtered_l[i] = fl
        filtered_r[i] = fr

    if not (np.isfinite(fl) and np.isfinite(fr)):
        fl, fr = 0.0, 0.0
    filter_state[hand_key]["l"], filter_state[hand_key]["r"] = fl, fr

    return filtered_l, filtered_r


def audio_callback(outdata, frames, time_info, status):
    global phase_l, phase_r
    if status:
        print(status)

    t_l = (np.arange(frames) + phase_l) / SAMPLE_RATE
    t_r = (np.arange(frames) + phase_r) / SAMPLE_RATE
    phase_l = (phase_l + frames) % SAMPLE_RATE
    phase_r = (phase_r + frames) % SAMPLE_RATE

    left_a, right_a = synth_source("Left", FREQ_LEFT, frames, t_l, phase_l)
    left_b, right_b = synth_source("Right", FREQ_RIGHT, frames, t_r, phase_r)

    mixed_l = left_a + left_b
    mixed_r = right_a + right_b

    outdata[:, 0] = np.clip(np.nan_to_num(mixed_l), -1.0, 1.0)
    outdata[:, 1] = np.clip(np.nan_to_num(mixed_r), -1.0, 1.0)


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
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam.")
    exit()

print("Running two-hand ML spatial audio. Press 'q' to quit.")

# Smoothed values and gesture history, per hand.
smoothed = {
    "Left": {"x": 0.5, "y": 0.5, "size": 0.15},
    "Right": {"x": 0.5, "y": 0.5, "size": 0.15},
}
gesture_history = {
    "Left": deque(maxlen=7),
    "Right": deque(maxlen=7),
}
SMOOTHING = 0.2
MIN_SPAN = 0.08
MAX_SPAN = 0.35

# Grace period: if a hand isn't detected for a few consecutive frames
# (e.g. brief occlusion when hands overlap/cross), don't mute
# immediately -- wait a few frames first. Prevents overlap from
# instantly cutting all sound.
GRACE_FRAMES = 15
frames_since_seen = {"Left": 0, "Right": 0}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_bar(frame, label, value, x, y, w=180, h=16, color=(0, 255, 0)):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), 1)
    fill_w = int(w * max(0.0, min(1.0, value)))
    cv2.rectangle(frame, (x, y), (x + fill_w, y + h), color, -1)
    cv2.putText(frame, f"{label}: {value:.2f}", (x, y - 5), FONT, 0.45, (255, 255, 255), 1)


def draw_radar(frame, sources, cx, cy, radius=80):
    cv2.circle(frame, (cx, cy), radius, (60, 60, 60), 2)
    cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)
    cv2.putText(frame, "YOU", (cx - 15, cy + radius + 20), FONT, 0.5, (200, 200, 200), 1)

    colors = {"Left": (255, 120, 0), "Right": (0, 200, 255)}
    for key, s in sources.items():
        dx = (s["pan"] - 0.5) * 2 * radius
        dist_from_center = radius * (1.0 - s["distance"])
        dot_x = int(cx + dx * (dist_from_center / radius) if radius > 0 else cx)
        dot_y = int(cy - dist_from_center * 0.3)
        color = (0, 0, 255) if s["muted"] else colors[key]
        cv2.circle(frame, (dot_x, dot_y), 8, color, -1)
        cv2.line(frame, (cx, cy), (dot_x, dot_y), color, 1)


TRAINING_HAND = "Right"  # which hand the gesture_model.pkl was trained on


def predict_gesture(hand_landmarks, hand_key):
    """
    Extracts landmarks, normalizes relative to the wrist, and returns the
    predicted gesture label.

    Mirror correction: the model was trained using only the
    TRAINING_HAND. A left hand's fist and a right hand's fist are mirror
    images of each other in landmark coordinates (x flips sign), so
    without correction, the "other" hand looks like unfamiliar data to
    the model. We flip the x-axis for any hand that isn't the training
    hand, making it look like the training hand's chirality before
    classifying.
    """
    raw = []
    for lm in hand_landmarks.landmark:
        raw += [lm.x, lm.y, lm.z]

    wrist_x, wrist_y, wrist_z = raw[0], raw[1], raw[2]
    mirror = (hand_key != TRAINING_HAND)

    normalized = []
    for i in range(21):
        dx = raw[i * 3] - wrist_x
        dy = raw[i * 3 + 1] - wrist_y
        dz = raw[i * 3 + 2] - wrist_z
        if mirror:
            dx = -dx
        normalized += [dx, dy, dz]

    features_df = pd.DataFrame([normalized], columns=FEATURE_COLUMNS)
    return gesture_model.predict(features_df)[0]


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    detected_this_frame = set()
    stable_gestures = {}

    if results.multi_hand_landmarks and results.multi_handedness:
        # Pair each hand's landmarks with its MediaPipe handedness label
        # (anatomical Left/Right, based on hand appearance -- NOT screen
        # position). This is what we need for mirror correction, since
        # chirality is a property of the physical hand, not where it is
        # on screen.
        hands_with_info = []
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            avg_x = sum(lm.x for lm in hand_landmarks.landmark) / 21
            # MediaPipe assumes the input is already mirrored (selfie-view).
            # We additionally flip the frame ourselves for display, which
            # double-mirrors it and inverts MediaPipe's Left/Right label.
            # Swap it back here so "Right" truly means your real right hand.
            raw_label = handedness.classification[0].label
            anatomical_label = "Left" if raw_label == "Right" else "Right"
            hands_with_info.append((avg_x, hand_landmarks, anatomical_label))

        hands_with_info.sort(key=lambda triple: triple[0])  # smallest x = leftmost on screen

        if len(hands_with_info) == 1:
            # Only one hand visible right now (common during partial
            # overlap/occlusion) -- assign its slot based on which half
            # of the screen it's actually in, instead of always
            # defaulting to "Left". This prevents the slot from jumping
            # around during exactly the moments hands are crossing.
            avg_x, hand_landmarks, anatomical_label = hands_with_info[0]
            source_key = "Left" if avg_x < 0.5 else "Right"
            assignments = [(source_key, hand_landmarks, anatomical_label)]
        else:
            source_keys_in_order = ["Left", "Right"][:len(hands_with_info)]
            assignments = [
                (source_key, hand_landmarks, anatomical_label)
                for source_key, (_, hand_landmarks, anatomical_label)
                in zip(source_keys_in_order, hands_with_info)
            ]

        for source_key, hand_landmarks, anatomical_label in assignments:
            hand_key = source_key  # used for audio/UI slot below
            detected_this_frame.add(hand_key)
            frames_since_seen[hand_key] = 0

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            wrist = hand_landmarks.landmark[0]
            middle_tip = hand_landmarks.landmark[12]

            wrist_x = max(0.0, min(1.0, wrist.x))
            wrist_y_inverted = max(0.0, min(1.0, 1.0 - wrist.y))

            predicted = predict_gesture(hand_landmarks, anatomical_label)
            gesture_history[hand_key].append(predicted)
            stable_gesture = Counter(gesture_history[hand_key]).most_common(1)[0][0]
            stable_gestures[hand_key] = stable_gesture

            raw_span = math.hypot(middle_tip.x - wrist.x, middle_tip.y - wrist.y)
            span_normalized = (raw_span - MIN_SPAN) / (MAX_SPAN - MIN_SPAN)
            span_normalized = max(0.0, min(1.0, span_normalized))

            sm = smoothed[hand_key]
            sm["x"] = SMOOTHING * wrist_x + (1 - SMOOTHING) * sm["x"]
            sm["y"] = SMOOTHING * wrist_y_inverted + (1 - SMOOTHING) * sm["y"]
            sm["size"] = SMOOTHING * span_normalized + (1 - SMOOTHING) * sm["size"]

            with state_lock:
                hand_state[hand_key]["pan"] = sm["x"]
                hand_state[hand_key]["elevation"] = sm["y"]
                hand_state[hand_key]["distance"] = sm["size"]
                hand_state[hand_key]["muted"] = stable_gesture in MUTE_GESTURES

    # Any hand NOT detected this frame counts up toward its grace period.
    # Only mute once it's been missing for GRACE_FRAMES in a row -- a
    # brief overlap/occlusion won't instantly silence that source.
    for key in ("Left", "Right"):
        if key not in detected_this_frame:
            frames_since_seen[key] += 1
            if frames_since_seen[key] > GRACE_FRAMES:
                with state_lock:
                    hand_state[key]["muted"] = True

    # ---------- On-screen overlay ----------
    y0 = 30
    cv2.putText(frame, "LEFT hand source", (20, y0), FONT, 0.55, (255, 120, 0), 2)
    draw_bar(frame, "Pan", smoothed["Left"]["x"], 20, y0 + 20, color=(255, 120, 0))
    draw_bar(frame, "Elevation", smoothed["Left"]["y"], 20, y0 + 55, color=(255, 120, 0))
    draw_bar(frame, "Distance", smoothed["Left"]["size"], 20, y0 + 90, color=(255, 120, 0))
    if "Left" in stable_gestures:
        cv2.putText(frame, f"Gesture: {stable_gestures['Left']}", (20, y0 + 120),
                    FONT, 0.5, (255, 255, 0), 1)

    y1 = 200
    cv2.putText(frame, "RIGHT hand source", (20, y1), FONT, 0.55, (0, 200, 255), 2)
    draw_bar(frame, "Pan", smoothed["Right"]["x"], 20, y1 + 20, color=(0, 200, 255))
    draw_bar(frame, "Elevation", smoothed["Right"]["y"], 20, y1 + 55, color=(0, 200, 255))
    draw_bar(frame, "Distance", smoothed["Right"]["size"], 20, y1 + 90, color=(0, 200, 255))
    if "Right" in stable_gestures:
        cv2.putText(frame, f"Gesture: {stable_gestures['Right']}", (20, y1 + 120),
                    FONT, 0.5, (255, 255, 0), 1)

    with state_lock:
        sources_snapshot = {k: v.copy() for k, v in hand_state.items()}
    frame_h, frame_w, _ = frame.shape
    draw_radar(frame, sources_snapshot, cx=frame_w - 120, cy=100, radius=70)

    cv2.imshow("Two-Hand ML Spatial Audio - press q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
stream.stop()
stream.close()