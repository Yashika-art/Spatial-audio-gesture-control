# Spatial-audio-gesture-control
Real-time spatial audio controlled by hand gestures via webcam. Uses MediaPipe for tracking, a custom-trained ML classifier for gesture recognition, and supports two-hand independent audio sources.

## Overview

This project lets you control spatial audio using only your hands and a standard webcam — no specialized hardware, depth camera, or motion capture equipment required. Hand position and gestures are translated into live audio parameters, creating an interactive, gesture-driven sound experience.

## Features

- **Real-time hand tracking** using MediaPipe, extracting 21 3D landmarks per hand
- **Three-axis spatial audio control:**
  - **Azimuth** — left/right panning (equal-power panning, hand x-position)
  - **Elevation** — brightness filter simulating vertical sound placement (hand y-position)
  - **Distance** — volume scaling based on hand size in frame (proxy for depth)
- **ML-based gesture recognition** — a custom-trained scikit-learn KNN classifier recognizes five gestures (fist, open palm, point, peace sign, pinch) from hand landmark data, used to control muting
- **Two-hand support** — each hand acts as an independent spatial audio source with its own pitch, timbre, and gesture control
- **Mirror correction** — accounts for left/right hand asymmetry so the gesture model generalizes across both hands
- **Overlap handling** — screen-position-based source assignment and grace-period muting keep audio stable when hands cross or briefly occlude each other
- **Live on-screen overlay** — visual bars and a top-down "radar" view show pan, elevation, distance, and detected gesture in real time

## Tech Stack

- Python 3.11
- [OpenCV](https://opencv.org/) — webcam capture and video display
- [MediaPipe](https://developers.google.com/mediapipe) — hand landmark detection
- [NumPy](https://numpy.org/) — signal processing and math
- [sounddevice](https://python-sounddevice.readthedocs.io/) — real-time audio output
- [scikit-learn](https://scikit-learn.org/) — gesture classification (KNN)
- [pandas](https://pandas.pydata.org/) — data handling for training
- [joblib](https://joblib.readthedocs.io/) — model persistence

## Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/gesture-spatial-audio.git
cd gesture-spatial-audio

# Create a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install mediapipe opencv-python numpy sounddevice scikit-learn pandas joblib
```

## Usage

### 1. Collect gesture training data
```bash
python collect_gesture_data.py
```
Hold number keys 1–5 to record each gesture (fist, open palm, point, peace sign, pinch). Move your hand slightly while recording for a more robust dataset.

### 2. Train the gesture classifier
```bash
python train_gesture_model.py
```
Trains a KNN classifier on the collected data and saves it as `gesture_model.pkl`. Prints an accuracy report and confusion matrix.

### 3. Run the live application
```bash
python step8_two_hand.py
```
Use headphones for the best spatial audio experience. Hold up one or both hands to control independent sound sources.

## Project Structure

```
├── collect_gesture_data.py     # Gesture training data collection tool
├── train_gesture_model.py      # Trains and evaluates the gesture classifier
├── step8_two_hand.py           # Main application (two-hand spatial audio + ML gestures)
├── gesture_data.csv            # Collected training data (generated)
├── gesture_model.pkl           # Trained classifier (generated)
└── README.md
```

## Known Limitations

- Elevation and distance use simplified DSP proxies (a brightness filter and hand-size-based volume) rather than true HRTF (Head-Related Transfer Function) convolution
- Single-camera tracking has inherent depth-estimation limits, especially during full hand occlusion
- Gesture recognition accuracy depends on training data diversity (angles, lighting, distances)

## Future Work

- Replace simplified elevation/distance cues with real HRTF-based binaural rendering
- Expand the gesture vocabulary and dataset size for higher classification accuracy
- Add support for real audio samples/instruments instead of synthesized tones
- Explore depth-camera integration for more robust distance and overlap handling

## License

MIT License (or update to whatever license you prefer)
