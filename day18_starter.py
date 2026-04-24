import cv2
import numpy as np
import sys
import time

MODEL_TYPE = None
depth_model = None
depth_transform = None

def load_midas_torch():
    global depth_model, depth_transform, MODEL_TYPE
    try:
        import torch
        print("Loading MiDaS-small via torch hub (~80 MB first run)...")
        depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        depth_transform = transforms.small_transform
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        depth_model.to(device)
        depth_model.eval()
        MODEL_TYPE = "torch"
        print(f"✓ MiDaS-small loaded (device: {device})")
        return True
    except Exception as e:
        print(f"torch/MiDaS failed: {e}")
        return False

def load_onnx_fallback():
    global MODEL_TYPE
    MODEL_TYPE = "pseudo"
    print("⚠️  Using pseudo-depth fallback")
    return True

print("\n" + "=" * 55)
print("  📡 DepthMapper — Day 18")
print("=" * 55)

if not load_midas_torch():
    load_onnx_fallback()

def estimate_depth(frame_bgr):
    if MODEL_TYPE == "torch":
        import torch
        device = next(depth_model.parameters()).device
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_batch = depth_transform(rgb).to(device)

        with torch.no_grad():
            prediction = depth_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_bgr.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()
    else:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        depth = cv2.GaussianBlur(np.abs(laplacian), (51, 51), 0)

    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        depth = (depth - d_min) / (d_max - d_min)
    else:
        depth = np.zeros_like(depth)

    return depth

# TODO #2 complete: tuned colormap
COLORMAP = cv2.COLORMAP_PLASMA

def colorize_depth(depth_normalized):
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, COLORMAP)

def export_point_cloud(depth_map, frame_shape, filename="point_cloud.csv"):
    h, w = frame_shape[:2]
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))

    xs_flat = xs.flatten()
    ys_flat = ys.flatten()
    d_flat = depth_map.flatten()

    step = 4
    xs_s = xs_flat[::step]
    ys_s = ys_flat[::step]
    ds_s = d_flat[::step]

    import csv
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "depth"])
        for x, y, d in zip(xs_s, ys_s, ds_s):
            writer.writerow([x, y, f"{d:.4f}"])

    total_points = len(xs_s)
    print(f"✓ Saved {filename} ({total_points:,} points)")
    return filename

def show_histogram(depth_map):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.hist(depth_map.flatten(), bins=100, color="#0f7173", edgecolor="none")
    plt.xlabel("Depth (normalized 0=near, 1=far)")
    plt.ylabel("Pixel count")
    plt.title("Depth Distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# TODO #1 complete: center-region distance estimation
def estimate_center_depth(depth_map):
    """
    Average depth in the center 20% of the frame.
    Returns float from 0 to 1.
    """
    h, w = depth_map.shape

    center_y, center_x = h // 2, w // 2
    region_h, region_w = h // 5, w // 5

    y1 = max(0, center_y - region_h // 2)
    y2 = min(h, center_y + region_h // 2)
    x1 = max(0, center_x - region_w // 2)
    x2 = min(w, center_x + region_w // 2)

    center_region = depth_map[y1:y2, x1:x2]

    if center_region.size == 0:
        return 0.5

    return float(np.mean(center_region))

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("ERROR: No webcam found.")
    sys.exit(1)

print("\nDepthMapper running!")
print("Controls: 's'=save CSV | 'h'=histogram | 'q'=quit")
print("Move your hand closer/further — watch the heatmap change.\n")

last_depth = None
frame_count = 0
fps_start = time.time()
fps = 0.0
PROCESS_EVERY = 3

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % PROCESS_EVERY == 0:
        depth = estimate_depth(frame)
        last_depth = depth

        elapsed = time.time() - fps_start
        fps = PROCESS_EVERY / elapsed if elapsed > 0 else 0
        fps_start = time.time()

    if last_depth is None:
        continue

    heatmap = colorize_depth(last_depth)
    blended = cv2.addWeighted(frame, 0.3, heatmap, 0.7, 0)

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    rw, rh = w // 10, h // 10

    cv2.rectangle(blended, (cx - rw, cy - rh), (cx + rw, cy + rh), (255, 255, 255), 2)

    center_d = estimate_center_depth(last_depth)

    if center_d < 0.33:
        range_str = "NEAR"
        range_color = (0, 255, 0)
    elif center_d < 0.66:
        range_str = "MID"
        range_color = (0, 255, 255)
    else:
        range_str = "FAR"
        range_color = (0, 0, 255)

    cv2.putText(
        blended,
        f"Center: {range_str} ({center_d:.2f})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        range_color,
        2,
    )

    cv2.putText(
        blended,
        f"FPS: {fps:.1f} | Model: {MODEL_TYPE}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
    )

    cv2.putText(
        blended,
        "s=save CSV  h=histogram  q=quit",
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (150, 150, 150),
        1,
    )

    cv2.imshow("DepthMapper - Day 18", blended)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("s") and last_depth is not None:
        export_point_cloud(last_depth, frame.shape)
    elif key == ord("h") and last_depth is not None:
        show_histogram(last_depth)

cap.release()
cv2.destroyAllWindows()

print("\nDepthMapper ended. See you tomorrow for Day 19!")
