"""
Step 4: Hand height controls elevation (via brightness filter)
Combines with Step 3's left-right panning:
  - Hand x-position -> left/right pan (azimuth)
  - Hand y-position -> brightness (simplified elevation cue)

Move hand up -> tone gets brighter/sharper (simulates "above")
Move hand down -> tone gets darker/muffled (simulates "below")

This uses a simple one-pole low-pass filter as a stand-in for real
elevation cues. Real elevation perception needs HRTF data (next step).

Press 'q' to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import threading

# ---------- Audio setup ----------
SAMPLE_RATE = 44100
FREQUENCY = 440.0
BLOCK_SIZE = 512

pan_position = 0.5       # 0.0 = left, 1.0 = right
elevation_position = 0.5 # 0.0 = low/dark, 1.0 = high/bright
state_lock = threading.Lock()

phase = 0.0
filter_state_l = 0.0  # one-pole filter memory, left channel
filter_state_r = 0.0  # one-pole filter memory, right channel


def audio_callback(outdata, frames, time_info, status):
    global phase, filter_state_l, filter_state_r
    if status:
        print(status)

    t = (np.arange(frames) + phase) / SAMPLE_RATE
    tone = 0.3 * np.sin(2 * np.pi * FREQUENCY * t)
    phase += frames

    with state_lock:
        pan = pan_position
        elev = elevation_position

    # Equal-power panning (from Step 3)
    theta = pan * (np.pi / 2)
    left_gain = np.cos(theta)
    right_gain = np.sin(theta)

    left = tone * left_gain
    right = tone * right_gain

    # One-pole low-pass filter: alpha near 1.0 = bright/unfiltered (high hand),
    # alpha near 0.05 = heavily muffled (low hand). This is a crude but
    # audible stand-in for elevation brightness cues.
    alpha = 0.05 + elev * 0.95

    filtered_l = np.empty_like(left)
    filtered_r = np.empty_like(right)
    fl, fr = filter_state_l, filter_state_r
    for i in range(len(left)):
        fl = fl + alpha * (left[i] - fl)
        fr = fr + alpha * (right[i] - fr)
        filtered_l[i] = fl
        filtered_r[i] = fr
    filter_state_l, filter_state_r = fl, fr

    outdata[:, 0] = filtered_l
    outdata[:, 1] = filtered_r


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

print("Running. Move hand left/right = pan, up/down = brightness. Press 'q' to quit.")

smoothed_x = 0.5
smoothed_y = 0.5
SMOOTHING = 0.2

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
            wrist_x = wrist.x  # 0.0 (left) to 1.0 (right)
            # y is 0.0 at TOP of frame, 1.0 at BOTTOM -> invert so
            # "hand up" = higher elevation value = brighter.
            wrist_y_inverted = 1.0 - wrist.y

            smoothed_x = SMOOTHING * wrist_x + (1 - SMOOTHING) * smoothed_x
            smoothed_y = SMOOTHING * wrist_y_inverted + (1 - SMOOTHING) * smoothed_y

            with state_lock:
                pan_position = smoothed_x
                elevation_position = smoothed_y

    cv2.imshow("Hand-controlled spatial audio - press q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
stream.stop()
stream.close()