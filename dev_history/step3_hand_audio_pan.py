"""
Step 3: Hand position controls audio panning
Plays a continuous tone and pans it left/right in real time based on
your wrist's x-position. This is your first full "camera -> audio" loop.

Move your hand left -> sound moves to left ear.
Move your hand right -> sound moves to right ear.

Press 'q' (with the video window focused) to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import threading

# ---------- Audio setup ----------
SAMPLE_RATE = 44100
FREQUENCY = 440.0  # A4 tone, easy to hear clearly
BLOCK_SIZE = 512

# Shared value between the video thread and the audio thread.
# 0.0 = fully left, 1.0 = fully right, 0.5 = center.
pan_position = 0.5
pan_lock = threading.Lock()

phase = 0.0  # keeps the sine wave continuous across audio callback blocks


def audio_callback(outdata, frames, time_info, status):
    global phase
    if status:
        print(status)

    t = (np.arange(frames) + phase) / SAMPLE_RATE
    tone = 0.3 * np.sin(2 * np.pi * FREQUENCY * t)  # 0.3 = volume, keep it gentle
    phase += frames

    with pan_lock:
        pan = pan_position

    # Equal-power panning: uses a quarter sine/cosine curve instead of a
    # straight line, so total loudness stays constant across the pan range
    # (fixes the "quiet dip in the middle" you get with linear panning).
    theta = pan * (np.pi / 2)  # pan 0->1 maps to angle 0->90 degrees
    left_gain = np.cos(theta)
    right_gain = np.sin(theta)

    outdata[:, 0] = tone * left_gain
    outdata[:, 1] = tone * right_gain


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

print("Running. Move your hand left/right to pan the tone. Press 'q' to quit.")

# Light smoothing so the audio doesn't jitter with every tracking wobble.
smoothed_x = 0.5
SMOOTHING = 0.2  # lower = smoother but more lag, higher = more responsive

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

            wrist_x = hand_landmarks.landmark[0].x  # 0.0 (left) to 1.0 (right)

            # Exponential smoothing to reduce jitter.
            smoothed_x = SMOOTHING * wrist_x + (1 - SMOOTHING) * smoothed_x

            with pan_lock:
                pan_position = smoothed_x

    cv2.imshow("Hand-controlled panning - press q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
stream.stop()
stream.close()