"""
Day 2: Train the gesture classifier
Loads gesture_data.csv (collected in Day 1), trains a classifier on the
hand landmark features, evaluates it, and saves the trained model to
disk so it can be loaded later inside the live spatial audio app.

Run this once after collecting data. Re-run any time you collect more
data or want to retrain.
"""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

DATA_FILE = "gesture_data.csv"
MODEL_FILE = "gesture_model.pkl"

# ---------- Load data ----------
print(f"Loading {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)
print(f"Loaded {len(df)} samples across {df['label'].nunique()} gestures.")
print(df['label'].value_counts())

# Features = landmark columns only (exclude label and session_id)
feature_cols = [c for c in df.columns if c not in ('label', 'session_id')]
X = df[feature_cols]
y = df['label']
groups = df['session_id']  # which recording "take" each row came from

# ---------- Normalize landmarks relative to the wrist ----------
# Raw x/y/z values depend on where your hand is in the frame, which we
# don't want the classifier learning (a fist should be a fist whether
# it's on the left or right side of the screen). We subtract the wrist
# position (landmark 0) from every other landmark so the features
# represent hand SHAPE, not hand POSITION.
def normalize_landmarks(row):
    wrist_x, wrist_y, wrist_z = row['x0'], row['y0'], row['z0']
    normalized = row.copy()
    for i in range(21):
        normalized[f'x{i}'] = row[f'x{i}'] - wrist_x
        normalized[f'y{i}'] = row[f'y{i}'] - wrist_y
        normalized[f'z{i}'] = row[f'z{i}'] - wrist_z
    return normalized

X = X.apply(normalize_landmarks, axis=1)

# ---------- Train/test split BY SESSION, not by random frame ----------
# Critical: consecutive frames from the same "take" are nearly identical
# (your hand barely moves between frames). A random per-frame split would
# leak near-duplicate frames into both train and test, causing artificially
# inflated (fake) accuracy. Splitting by session_id ensures entire
# recording sessions go fully into either train or test, giving us a
# realistic measure of how well the model generalizes to a hand pose it
# hasn't effectively already seen.
splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(f"\nTrain sessions: {groups.iloc[train_idx].nunique()}, "
      f"Test sessions: {groups.iloc[test_idx].nunique()}")

# ---------- Train ----------
# KNN is a good starting choice here: simple, fast, works well on small
# structured datasets like this without much tuning. You can swap this
# for an MLPClassifier (small neural net) later if you want to compare.
print("\nTraining KNN classifier...")
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# ---------- Evaluate ----------
y_pred = model.predict(X_test)
print("\n--- Evaluation on held-out test data ---")
print(classification_report(y_test, y_pred))

print("Confusion matrix (rows=actual, columns=predicted):")
labels_sorted = sorted(y.unique())
print("Labels order:", labels_sorted)
print(confusion_matrix(y_test, y_pred, labels=labels_sorted))

# ---------- Save model ----------
joblib.dump(model, MODEL_FILE)
print(f"\nModel saved to {MODEL_FILE}")