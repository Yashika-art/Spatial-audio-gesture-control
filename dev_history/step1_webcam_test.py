"""
Step 1: Webcam test
Just confirms your camera works and OpenCV can read frames from it.
Press 'q' to quit the window.
"""

import cv2

# 0 = default webcam. If you have multiple cameras, try 1, 2, etc.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam. Check that no other app is using it,")
    print("and that you selected the right camera index.")
    exit()

print("Webcam opened successfully. Press 'q' in the video window to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Flip horizontally so it acts like a mirror (feels more natural)
    frame = cv2.flip(frame, 1)

    cv2.imshow("Webcam Test - press q to quit", frame)

    # Wait 1ms for a keypress; exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
