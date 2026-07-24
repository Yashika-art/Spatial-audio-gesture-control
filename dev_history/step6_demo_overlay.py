"""
Step 6: Demo-ready version with on-screen overlay
Same audio engine as Step 5 (pan, brightness, distance-volume), but now
shows live numeric values on screen -- essential for presenting, since
evaluators can SEE the numbers respond as you move your hand, not just
hear it.

Press 'q' to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import threading
import math

# ---------- Audio setup ----------
SAMPLE_RATE = 44100
FREQUENCY = 330.0  # slightly lower than before, easier on the ears for long sessions
BLOCK_SIZE = 512

pan_position = 0.5
elevation_position = 0.5
distance_volume = 0.5
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

print("Running. Press 'q' to quit.")
print("Move your hand around, then watch the printed span values below")
print("to help calibrate MIN_SPAN / MAX_SPAN for your setup.")

smoothed_x = 0.5
smoothed_y = 0.5
smoothed_size = 0.15
SMOOTHING = 0.2

# Calibrate these based on what you observe today (see console output).
MIN_SPAN = 0.08
MAX_SPAN = 0.35

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_bar(frame, label, value, x, y, w=200, h=20, color=(0, 255, 0)):
    """Draws a labeled horizontal bar from 0.0 to 1.0."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), 1)
    fill_w = int(w * max(0.0, min(1.0, value)))
    cv2.rectangle(frame, (x, y), (x + fill_w, y + h), color, -1)
    cv2.putText(frame, f"{label}: {value:.2f}", (x, y - 6), FONT, 0.5, (255, 255, 255), 1)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    raw_span = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            wrist = hand_landmarks.landmark[0]
            middle_tip = hand_landmarks.landmark[12]

            wrist_x = max(0.0, min(1.0, wrist.x))
            wrist_y_inverted = max(0.0, min(1.0, 1.0 - wrist.y))

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

    # ---------- On-screen overlay ----------
    draw_bar(frame, "Pan (L-R)", smoothed_x, 20, 40, color=(255, 120, 0))
    draw_bar(frame, "Elevation", smoothed_y, 20, 90, color=(0, 200, 255))
    draw_bar(frame, "Distance/Vol", smoothed_size, 20, 140, color=(0, 255, 120))

    if raw_span is not None:
        cv2.putText(frame, f"raw hand span: {raw_span:.3f}", (20, 180),
                    FONT, 0.5, (200, 200, 200), 1)
        # Print to console too, useful for calibrating MIN_SPAN/MAX_SPAN.
        print(f"raw span: {raw_span:.3f}", end="\r")

    cv2.imshow("Spatial Audio Demo - press q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
stream.stop()
stream.close()