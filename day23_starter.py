import cv2
import numpy as np
import sys
import time
import collections

# ============================================================
# CAMERA SETUP
# ============================================================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("ERROR: No webcam found.")
    sys.exit(1)

ret, frame = cap.read()
if not ret:
    print("ERROR: Could not read from webcam.")
    sys.exit(1)

FRAME_H, FRAME_W = frame.shape[:2]
CENTER_X, CENTER_Y = FRAME_W // 2, FRAME_H // 2

# ============================================================
# HSV Color Range
# ============================================================

HSV_LOWER = np.array([40, 50, 50])
HSV_UPPER = np.array([80, 255, 255])

MIN_CONTOUR_AREA = 500

# ============================================================
# PID CONTROLLER
# ============================================================

class PIDController:
    def __init__(self, Kp=0.45, Ki=0.015, Kd=0.08, integral_limit=500):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()

        self.integral_limit = integral_limit

    def update(self, error):
        now = time.time()
        dt = now - self.prev_time

        if dt <= 0:
            dt = 0.001

        # P term
        p = self.Kp * error

        # I term
        self.integral += error * dt
        self.integral = max(
            -self.integral_limit,
            min(self.integral, self.integral_limit)
        )
        i = self.Ki * self.integral

        # D term
        derivative = (error - self.prev_error) / dt
        d = self.Kd * derivative

        output = p + i + d

        self.prev_error = error
        self.prev_time = now

        return output, p, i, d

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()


pid_x = PIDController(Kp=0.45, Ki=0.015, Kd=0.08)
pid_y = PIDController(Kp=0.45, Ki=0.015, Kd=0.08)

# ============================================================
# COLOR SAMPLER
# ============================================================

frame = None

def mouse_callback(event, x, y, flags, param):
    global HSV_LOWER, HSV_UPPER, frame

    if event == cv2.EVENT_LBUTTONDOWN and frame is not None:
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = frame_hsv[y, x]

        print(f"\nSampled HSV at ({x},{y}): H={h}, S={s}, V={v}")

        h_margin = 15
        s_margin = 60
        v_margin = 60

        HSV_LOWER = np.array([
            max(0, h - h_margin),
            max(0, s - s_margin),
            max(0, v - v_margin)
        ])

        HSV_UPPER = np.array([
            min(179, h + h_margin),
            min(255, s + s_margin),
            min(255, v + v_margin)
        ])

        print(f"New HSV range: lower={HSV_LOWER}, upper={HSV_UPPER}")


cv2.namedWindow("ObjectFollower - Day 23")
cv2.setMouseCallback("ObjectFollower - Day 23", mouse_callback)

# ============================================================
# TRACKING HISTORY
# ============================================================

trajectory = collections.deque(maxlen=50)

# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def draw_crosshair(frame, x, y, size=20, color=(0, 255, 255), thickness=2):
    cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
    cv2.line(frame, (x, y - size), (x, y + size), color, thickness)
    cv2.circle(frame, (x, y), size // 2, color, 1)


def draw_error_vector(frame, cx, cy):
    cv2.arrowedLine(
        frame,
        (CENTER_X, CENTER_Y),
        (cx, cy),
        (0, 100, 255),
        2,
        tipLength=0.2
    )


def draw_pid_bars(frame, output_x, output_y, max_val=200):
    bar_y = FRAME_H - 60
    bar_h = 20

    bar_len_x = int(min(abs(output_x) / max_val * (FRAME_W // 3), FRAME_W // 3))
    color_x = (0, 180, 255) if output_x > 0 else (255, 100, 0)

    if output_x > 0:
        cv2.rectangle(frame, (CENTER_X, bar_y), (CENTER_X + bar_len_x, bar_y + bar_h), color_x, -1)
    else:
        cv2.rectangle(frame, (CENTER_X - bar_len_x, bar_y), (CENTER_X, bar_y + bar_h), color_x, -1)

    cv2.rectangle(frame, (0, bar_y), (FRAME_W, bar_y + bar_h), (80, 80, 80), 1)

    cv2.putText(
        frame,
        f"PID_X: {output_x:+.1f}",
        (10, bar_y + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    bar_y2 = FRAME_H - 35

    bar_len_y = int(min(abs(output_y) / max_val * (FRAME_W // 3), FRAME_W // 3))
    color_y = (0, 255, 100) if output_y > 0 else (255, 0, 100)

    if output_y > 0:
        cv2.rectangle(frame, (CENTER_X, bar_y2), (CENTER_X + bar_len_y, bar_y2 + bar_h), color_y, -1)
    else:
        cv2.rectangle(frame, (CENTER_X - bar_len_y, bar_y2), (CENTER_X, bar_y2 + bar_h), color_y, -1)

    cv2.rectangle(frame, (0, bar_y2), (FRAME_W, bar_y2 + bar_h), (80, 80, 80), 1)

    cv2.putText(
        frame,
        f"PID_Y: {output_y:+.1f}",
        (10, bar_y2 + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

# ============================================================
# MAIN LOOP
# ============================================================

print("\n" + "=" * 55)
print("ObjectFollower — Day 23")
print("=" * 55)
print("Click on your target object to sample its color.")
print("Default target: bright green objects")
print("'r' = reset PID | 'q' = quit\n")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Failed to read frame.")
        break

    frame = cv2.flip(frame, 1)
    display = frame.copy()

    draw_crosshair(display, CENTER_X, CENTER_Y, size=30, color=(100, 100, 100), thickness=1)
    cv2.circle(display, (CENTER_X, CENTER_Y), 5, (200, 200, 200), -1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    object_found = False
    output_x = 0.0
    output_y = 0.0

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > MIN_CONTOUR_AREA:
            object_found = True

            M = cv2.moments(largest)

            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = CENTER_X, CENTER_Y

            bx, by, bw, bh = cv2.boundingRect(largest)

            cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.drawContours(display, [largest], -1, (0, 200, 0), 1)

            draw_crosshair(display, cx, cy, size=15, color=(0, 255, 255), thickness=2)

            error_x = cx - CENTER_X
            error_y = cy - CENTER_Y

            output_x, px, ix, dx = pid_x.update(error_x)
            output_y, py, iy, dy = pid_y.update(error_y)

            draw_error_vector(display, cx, cy)

            trajectory.append((cx, cy))

            cv2.putText(display, f"Error X: {error_x:+d} px", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            cv2.putText(display, f"Error Y: {error_y:+d} px", (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            cv2.putText(display, f"Area: {area:.0f} px^2", (10, 86),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 255, 150), 1)

            cv2.putText(display, f"P/I/D X: {px:+.1f} {ix:+.1f} {dx:+.1f}", (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            cv2.putText(display, f"P/I/D Y: {py:+.1f} {iy:+.1f} {dy:+.1f}", (10, 138),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    else:
        pid_x.reset()
        pid_y.reset()

    pts = list(trajectory)
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        color = (int(255 * alpha), int(200 * alpha), 0)
        cv2.line(display, pts[i - 1], pts[i], color, 2)

    draw_pid_bars(display, output_x, output_y)

    status = "TRACKING" if object_found else "SEARCHING..."
    status_color = (0, 255, 0) if object_found else (0, 0, 255)

    cv2.putText(display, status, (FRAME_W - 170, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    cv2.putText(
        display,
        f"HSV: [{HSV_LOWER[0]}-{HSV_UPPER[0]}, {HSV_LOWER[1]}-{HSV_UPPER[1]}, {HSV_LOWER[2]}-{HSV_UPPER[2]}]",
        (10, FRAME_H - 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (180, 180, 180),
        1
    )

    cv2.putText(display, "Click=sample color | r=reset | q=quit",
                (10, FRAME_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    mask_small = cv2.resize(mask, (FRAME_W // 4, FRAME_H // 4))
    mask_color = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)

    x1 = FRAME_W - FRAME_W // 4 - 10
    y1 = 10
    x2 = FRAME_W - 10
    y2 = 10 + FRAME_H // 4

    display[y1:y2, x1:x2] = mask_color

    cv2.putText(display, "MASK",
                (x1, y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    cv2.imshow("ObjectFollower - Day 23", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("r"):
        pid_x.reset()
        pid_y.reset()
        trajectory.clear()
        print("PID reset.")

cap.release()
cv2.destroyAllWindows()

print("\nObjectFollower ended. See you tomorrow for Day 24!")
