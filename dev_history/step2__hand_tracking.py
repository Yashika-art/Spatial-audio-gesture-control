"""
Step 2: Hand tracking
Adds MediaPipe on top of the webcam feed from Step 1.
Draws 21 landmark points on your hand and prints the wrist's
x-position to the console so you can see raw numbers changing
as you move your hand left/right.

Press 'q' to quit.
"""

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up the hand tracker.
# max_num_hands=1 keeps things simple for now — we'll add a second
# hand later once single-hand tracking feels solid.
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    exit()

print("Hand tracking started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # MediaPipe expects RGB, OpenCV gives us BGR — convert.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the 21 landmark points + connections on the frame.
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Landmark 0 is the wrist. x and y are normalized (0.0 to 1.0)
            # across the frame width/height. z is relative depth.
            wrist = hand_landmarks.landmark[0]
            print(f"Wrist x: {wrist.x:.2f}  y: {wrist.y:.2f}  z: {wrist.z:.2f}")

    cv2.imshow("Hand Tracking - press q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()