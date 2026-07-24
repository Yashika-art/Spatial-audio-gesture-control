"""
Step 5: Hand distance controls volume (distance cue)
Completes all three spatial axes:
  - Hand x-position -> left/right pan (azimuth)
  - Hand y-position -> brightness filter (simplified elevation)
  - Hand size in frame -> volume (distance: closer hand = louder)

We estimate "distance" using the spread between landmarks (wrist to
middle fingertip) rather than MediaPipe's raw z, which is more stable
for a single webcam.

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
FREQUENCY = 440.0
BLOCK_SIZE = 512

pan_position = 0.5
elevation_position = 0.5
distance_volume = 0.5   # 0.0 = quiet/far, 1.0 = loud/close
state_lock = threading.Lock()

phase = 0.0
filter_state_l = 0.0
filter_state_r = 0.0


def audio_callback(outdata, frames, time_info, status):
    global phase, filter_state_l, filter_state_r
    if status:
        print(status)

    t = (np.arange(frames) + phase) / SAMPLE_RATE
    tone = 0.3 * np.sin(2 * np.pi * FREQUENCY * t)
    # Wrap phase so it doesn't grow unbounded over a long session
    # (large numbers lose floating point precision over time).
    phase = (phase + frames) % SAMPLE_RATE

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
    # Safety net: if anything went non-finite (NaN/Inf), reset the filter
    # state to 0 instead of letting bad values propagate forever.
    if not (np.isfinite(fl) and np.isfinite(fr)):
        fl, fr = 0.0, 0.0
    filter_state_l, filter_state_r = fl, fr

    # Clip to a safe audio range no matter what, as a final guard.
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

print("Running. x=pan, y=brightness, hand size=volume/distance. Press 'q' to quit.")

smoothed_x = 0.5
smoothed_y = 0.5
smoothed_size = 0.15  # starting guess for a mid-distance hand span
SMOOTHING = 0.2

# Calibration: tune these if volume feels wrong for your camera/distance.
# MIN_SPAN = hand span (normalized) when hand is far away (quiet)
# MAX_SPAN = hand span (normalized) when hand is close to camera (loud)
MIN_SPAN = 0.08
MAX_SPAN = 0.35

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            wrist = hand_landmarks.landmark[0]
            middle_tip = hand_landmarks.landmark[12]  # middle fingertip

            # Clamp to [0, 1] -- MediaPipe landmarks can drift slightly
            # outside this range near frame edges, which was causing the
            # filter math to become unstable (NaN/overflow).
            wrist_x = max(0.0, min(1.0, wrist.x))
            wrist_y_inverted = max(0.0, min(1.0, 1.0 - wrist.y))

            # Hand span = distance from wrist to middle fingertip,
            # normalized. Bigger span = hand is closer to camera.
            span = math.hypot(middle_tip.x - wrist.x, middle_tip.y - wrist.y)
            span_normalized = (span - MIN_SPAN) / (MAX_SPAN - MIN_SPAN)
            span_normalized = max(0.0, min(1.0, span_normalized))  # clamp 0-1

            smoothed_x = SMOOTHING * wrist_x + (1 - SMOOTHING) * smoothed_x
            smoothed_y = SMOOTHING * wrist_y_inverted + (1 - SMOOTHING) * smoothed_y
            smoothed_size = SMOOTHING * span_normalized + (1 - SMOOTHING) * smoothed_size

            with state_lock:
                pan_position = smoothed_x
                elevation_position = smoothed_y
                distance_volume = smoothed_size

    cv2.imshow("Full spatial hand control - press q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
stream.stop()
stream.close()