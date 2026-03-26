"""
BUILDCORED ORCAS — Day 02: AirCanvas
======================================
Track thumb-to-index finger distance via webcam.
When pinched, your fingertip becomes a stylus
drawing colored strokes on screen in real time.

Hardware concept: Coordinate mapping + sampling rate
This is identical to reading X/Y from a resistive
touchscreen through an ADC.

YOUR TASK:
1. Adjust the pinch threshold (line marked TODO #1)
2. Add a second drawing color (line marked TODO #2)
3. Run it: python day02_starter.py
4. Push to GitHub before midnight

CONTROLS:
- Pinch thumb + index finger → draw
- Release → stop drawing
- Press 'c' → clear canvas
- Press 'q' → quit
"""

import cv2
import mediapipe as mp
import numpy as np
import math

# ============================================================
# SETUP — you don't need to change this section
# ============================================================

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("ERROR: No webcam found.")
    exit(1)

# Get frame dimensions
ret, test_frame = cap.read()
FRAME_H, FRAME_W = test_frame.shape[:2]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

# Canvas — a transparent overlay we draw on
# This persists between frames so your drawing doesn't disappear
canvas = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

# Landmark indices
THUMB_TIP = 4
INDEX_TIP = 8

# Store previous fingertip position for smooth lines
prev_x, prev_y = None, None

# Track if we were drawing last frame (for line continuity)
was_drawing = False

# ============================================================
# TODO #1: Set your pinch threshold
# ============================================================
# Distance between thumb tip and index tip (in pixels)
# When distance is BELOW this value → pinch detected → draw
# When distance is ABOVE this value → not pinching → stop
#
# Too low = hard to trigger, need to squeeze fingers tightly
# Too high = triggers accidentally when fingers are apart
# Start with 40 and adjust
#
PINCH_THRESHOLD = 40  # <-- Adjust this value


# ============================================================
# TODO #2: Set up your drawing colors
# ============================================================
# Colors are in BGR format (not RGB!) because OpenCV uses BGR
# Blue = (255, 0, 0), Green = (0, 255, 0), Red = (0, 0, 255)
# Yellow = (0, 255, 255), Cyan = (255, 255, 0), White = (255, 255, 255)
#
# current_color_index tracks which color is active
# Press 1, 2, 3... to switch colors
#
# ADD MORE COLORS to this list for your second color requirement:
#
COLORS = [
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
    # Add more colors here! Examples:
    # (255, 0, 0),   # Blue
    # (0, 255, 255),  # Yellow
    # (255, 255, 0),  # Cyan
]

COLOR_NAMES = [
    "Green",
    "Red",
    # Add matching names here
]

current_color_index = 0


def get_distance(lm1, lm2, w, h):
    """Calculate pixel distance between two landmarks."""
    x1, y1 = int(lm1.x * w), int(lm1.y * h)
    x2, y2 = int(lm2.x * w), int(lm2.y * h)
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# ============================================================
# MAIN LOOP — Sensor → Process → Output
# ============================================================
print("\nAirCanvas is running!")
print(f"Pinch threshold: {PINCH_THRESHOLD}px")
print(f"Colors available: {COLOR_NAMES}")
print("Pinch to draw. 'c' to clear. 'q' to quit.")
print("Press number keys (1, 2...) to switch colors.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame
    frame = cv2.flip(frame, 1)

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    results = hands.process(rgb_frame)

    drawing_now = False

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw hand skeleton on frame (optional, looks cool)
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Get thumb and index tip positions
        landmarks = hand_landmarks.landmark
        thumb = landmarks[THUMB_TIP]
        index = landmarks[INDEX_TIP]

        # Calculate distance between thumb and index finger
        distance = get_distance(thumb, index, FRAME_W, FRAME_H)

        # Get index fingertip position in pixels (this is our "stylus")
        ix, iy = int(index.x * FRAME_W), int(index.y * FRAME_H)

        # Draw a circle at the fingertip so you can see where you're pointing
        cv2.circle(frame, (ix, iy), 8, COLORS[current_color_index], -1)

        # Check if pinching
        if distance < PINCH_THRESHOLD:
            drawing_now = True

            # Draw a line from previous position to current position
            # This creates smooth strokes instead of dots
            if was_drawing and prev_x is not None:
                cv2.line(canvas, (prev_x, prev_y), (ix, iy),
                         COLORS[current_color_index], thickness=4)

            prev_x, prev_y = ix, iy
        else:
            # Not pinching — reset previous position
            prev_x, prev_y = None, None

        # Display distance value
        cv2.putText(frame, f"Distance: {distance:.0f}px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Threshold: {PINCH_THRESHOLD}px", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    else:
        prev_x, prev_y = None, None

    was_drawing = drawing_now

    # Display status
    status = "DRAWING" if drawing_now else "NOT DRAWING"
    color = (0, 255, 0) if drawing_now else (100, 100, 100)
    cv2.putText(frame, status, (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    # Show current color
    cv2.putText(frame, f"Color: {COLOR_NAMES[current_color_index]}",
                (10, FRAME_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                COLORS[current_color_index], 2)

    # Merge canvas onto frame
    # Where canvas is not black, show the drawing
    mask = canvas > 0
    frame[mask] = canvas[mask]

    # Show the result
    cv2.imshow("AirCanvas - Day 02", frame)

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        # Clear canvas
        canvas = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        print("Canvas cleared!")
    elif ord('1') <= key <= ord('9'):
        # Switch color
        idx = key - ord('1')
        if idx < len(COLORS):
            current_color_index = idx
            print(f"Color: {COLOR_NAMES[current_color_index]}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("\nAirCanvas ended. See you tomorrow for Day 03!")
