"""
 Gesture data collection tool
Captures hand landmarks from your webcam and saves them to a CSV file
with a label, so you can later train a classifier on this data.

HOW TO USE:
1. Run the script.
2. Hold up the gesture you want to record (e.g. a fist).
3. Press the number key matching that gesture (see GESTURE_LABELS below).
4. While held down, it keeps saving samples every frame -- move your
   hand slightly (angle, distance, position) while holding the key so
   your dataset isn't just one exact pose (this makes the classifier
   more robust later).
5. Release the key to stop recording that gesture.
6. Repeat for each gesture. Aim for at least 100-150 samples per gesture.
7. Press 'q' to quit and save everything to gesture_data.csv.

Recommended: record each gesture at a few different distances/angles
from the camera for a more robust dataset.
"""

import cv2
import mediapipe as mp
import csv
import os

# ---------- Gesture labels ----------
# Map keyboard keys to gesture names. Add/change these as you like.
GESTURE_LABELS = {
    ord('1'): "fist",
    ord('2'): "open_palm",
    ord('3'): "point",
    ord('4'): "peace_sign",
    ord('5'): "pinch",
}

OUTPUT_FILE = "gesture_data.csv"

# ---------- Setup ----------
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

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Prepare CSV file. Each row = 21 landmarks x (x,y,z) = 63 values + label.
file_exists = os.path.exists(OUTPUT_FILE)
csv_file = open(OUTPUT_FILE, mode='a', newline='')
csv_writer = csv.writer(csv_file)

if not file_exists:
    header = []
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    header.append("label")
    header.append("session_id")
    csv_writer.writerow(header)

print("Gesture data collection started.")
print("Hold a number key to record that gesture. Release to stop. 'q' to quit.")
for key, name in GESTURE_LABELS.items():
    print(f"  Key '{chr(key)}' -> {name}")

sample_counts = {name: 0 for name in GESTURE_LABELS.values()}
current_key = None  # tracks which key is currently held down
was_recording_last_frame = False
session_id = 0  # increments each time you start a new key-hold "take"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key in GESTURE_LABELS:
        current_key = key
    elif key == 255:  # no key pressed this frame
        current_key = None

    recording_label = GESTURE_LABELS.get(current_key)
    is_recording_now = recording_label is not None

    # Start of a new "take": increment session_id so this recording burst
    # is grouped separately from the previous one, even for the same gesture.
    if is_recording_now and not was_recording_last_frame:
        session_id += 1
    was_recording_last_frame = is_recording_now

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if recording_label is not None:
                row = []
                for lm in hand_landmarks.landmark:
                    row += [lm.x, lm.y, lm.z]
                row.append(recording_label)
                row.append(session_id)
                csv_writer.writerow(row)
                sample_counts[recording_label] += 1

    # ---------- Overlay ----------
    y_offset = 30
    if recording_label:
        cv2.putText(frame, f"RECORDING: {recording_label}", (20, y_offset),
                    FONT, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Hold a number key to record", (20, y_offset),
                    FONT, 0.7, (200, 200, 200), 2)

    y_offset += 40
    for name, count in sample_counts.items():
        cv2.putText(frame, f"{name}: {count}", (20, y_offset), FONT, 0.5, (255, 255, 255), 1)
        y_offset += 22

    cv2.imshow("Gesture Data Collection - press q to quit", frame)

cap.release()
cv2.destroyAllWindows()
csv_file.close()

print("\nDone. Saved to", OUTPUT_FILE)
print("Sample counts:", sample_counts)
