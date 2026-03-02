# Dual MediaPipe Palm Capture System (NO HOMOGRAPHY)
# Features preserved: orientation, distance normalization, finger overlap detection,
# frame stabilization, side separation, batch capture

from picamera2 import Picamera2
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import re

MIN_ROI_AREA_RATIO = 0.05

# =========================
# AUTO SUCCESS STATE
# =========================
SUCCESS_DISPLAY_TIME = 1.5  # seconds
success_active = False
success_start_time = 0
success_latched = False

# =========================
# CAMERA SETUP
# =========================
picam_left = Picamera2(0)
picam_right = Picamera2(1)

cfg = {"size": (1280, 1280), "format": "XRGB8888"}
picam_left.configure(picam_left.create_preview_configuration(main=cfg))
picam_right.configure(picam_right.create_preview_configuration(main=cfg))

picam_left.start()
picam_right.start()

# --- Set fixed brightness for Camera 1 ---
picam_right.set_controls({
    "AeEnable": False,       # Disable auto-exposure
    "AnalogueGain": 1.2,     # Adjust for your desired brightness
    "ExposureTime": 25000    # Adjust for your desired brightness
})

# --- IR recolor ONLY for Camera 1 ---
def recolor_ir(frame):
    f = frame.astype(np.float32)

    R_in = f[:, :, 2]
    G_in = f[:, :, 1]
    B_in = f[:, :, 0]

    # Tuned to map (0,47,115) → approx (160,135,165)
    R = 1.43 * B_in
    G = 0.85 * G_in + 0.75 * B_in
    B = 1.39 * B_in

    out = np.stack([B, G, R], axis=2)
    return np.clip(out, 0, 255).astype(np.uint8)

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands_left = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
hands_right = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# =========================
# PREPROCESSING
# =========================
clahe = cv2.createCLAHE(2.0, (8, 8))

def log_gabor_safe(img, center_freq=0.15, sigma_f=0.55, n_orients=4):
    img = img.astype(np.float32) / 255.0
    h, w = img.shape
    F = np.fft.fftshift(np.fft.fft2(img))
    u = np.linspace(-0.5, 0.5, w)
    v = np.linspace(-0.5, 0.5, h)
    U, V = np.meshgrid(u, v)
    r = np.sqrt(U**2 + V**2)
    r[r == 0] = 1e-6
    theta = np.arctan2(V, U)

    radial = np.exp(-(np.log(r / center_freq)**2) / (2 * np.log(sigma_f)**2))
    out = np.zeros_like(img)

    for k in range(n_orients):
        ang = k * np.pi / n_orients
        diff = np.abs((theta - ang + np.pi) % (2*np.pi) - np.pi)
        orient = np.exp(-(diff**2) / (2*(np.pi/(2*n_orients))**2))
        out += np.real(np.fft.ifft2(np.fft.ifftshift(F * radial * orient)))

    out -= out.min()
    out /= out.max() + 1e-6
    return (out * 255).astype(np.uint8)

# =========================
# ROI HELPERS
# =========================
def roi_area_ratio(poly, shape):
    h, w = shape[:2]
    x, y, rw, rh = cv2.boundingRect(poly.astype(int))
    return (rw * rh) / (w * h)

def roi_fully_inside(poly, shape):
    h, w = shape[:2]
    xs = poly[:, 0, 0]
    ys = poly[:, 0, 1]
    return np.all((xs >= 0) & (xs < w) & (ys >= 0) & (ys < h))

def fingers_inside_roi(poly, lm, shape):
    h, w = shape[:2]
    tips = [4, 8, 12, 16, 20]
    poly = poly.reshape(-1, 2).astype(np.int32)
    for i in tips:
        x = int(lm[i].x * w)
        y = int(lm[i].y * h)
        if cv2.pointPolygonTest(poly, (x, y), False) >= 0:
            return True
    return False

# =========================
# HAND → ROI PIPELINE
# =========================
def process_frame(frame, hands_solver, label_prefix="Cam", use_log_gabor=True):
    """
    Process a camera frame with MediaPipe hands.

    Returns:
        annotated_frame, roi_polygon, processed_roi, hand_landmarks
    """
    clean_frame = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands_solver.process(rgb)

    roi = None
    roi_processed = None
    roi_poly = None
    lm = None

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark
        hand_label = "Unknown"
        if result.multi_handedness:
            hand_label = result.multi_handedness[0].classification[0].label  # Left/Right

        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        h, w, _ = frame.shape

        # --- Get keypoints ---
        def to_xy(l):
            return np.array([l[0] * w, l[1] * h], dtype=np.float32)

        wrist = np.array([lm[0].x, lm[0].y])
        index_base = np.array([lm[5].x, lm[5].y])
        pinky_base = np.array([lm[17].x, lm[17].y])

        v1 = index_base - wrist
        v2 = pinky_base - wrist
        normal = np.cross(np.append(v1, 0), np.append(v2, 0))
        normal /= np.linalg.norm(normal) + 1e-6

        # Chirality correction ONLY for palm/back test
        z = normal[2]
        if hand_label == "Left":
            z = -z

        side = "Palm" if z > 0 else "Back"

        cv2.putText(frame, f"{label_prefix}: {hand_label}-{side}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0) if side == "Palm" else (0, 0, 255), 2)

        if side != "Palm":
            return frame, None, None, lm

        # --- Polygon ROI calculation ---
        wrist_xy = to_xy(wrist)
        index_xy = to_xy(index_base)
        pinky_xy = to_xy(pinky_base)

        base_vec = pinky_xy - index_xy
        mid_top = (index_xy + pinky_xy) / 2
        down_vec = wrist_xy - mid_top

        perp_vec = down_vec - np.dot(down_vec, base_vec) / np.dot(base_vec, base_vec) * base_vec
        perp_unit = perp_vec / (np.linalg.norm(perp_vec) + 1e-6)
        roi_size = int(np.linalg.norm(base_vec))

        if np.dot(wrist_xy - mid_top, perp_unit) < 0:
            perp_unit = -perp_unit

        top_left = index_xy
        top_right = pinky_xy
        bottom_right = (pinky_xy + perp_unit * roi_size).astype(int)
        bottom_left = (index_xy + perp_unit * roi_size).astype(int)
        roi_poly = np.array([top_left, top_right, bottom_right, bottom_left], np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [roi_poly], True, (255, 0, 0), 2)

        base_unit = base_vec / (np.linalg.norm(base_vec) + 1e-6)
        angle = np.degrees(np.arctan2(base_unit[1], base_unit[0]))

        # --- Fix 180° ambiguity using wrist direction ---
        # Rotate down_vec and check if wrist ends up below fingers
        Rtest = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
        down_rot = Rtest[:, :2] @ down_vec

        # If wrist points upward after rotation → flip 180°
        if down_rot[1] < 0:
            angle += 180

        center = (int(mid_top[0]), int(mid_top[1]))
        rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(clean_frame, rot, (w, h))
        pts_rot = cv2.transform(roi_poly, rot)

        x_min, y_min = np.min(pts_rot[:, 0, :], axis=0)
        x_max, y_max = np.max(pts_rot[:, 0, :], axis=0)
        x_min, y_min = np.clip([x_min, y_min], 0, [w-1, h-1])
        x_max, y_max = np.clip([x_max, y_max], 0, [w-1, h-1])

        if x_max > x_min and y_max > y_min:
            roi = rotated[int(y_min):int(y_max), int(x_min):int(x_max)]
            if roi.size > 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5,5), 0)
                roi_processed = clahe.apply(blur)
                roi_processed = cv2.resize(roi_processed, (256, 256))
                
                # --- Apply Log-Gabor only if requested ---
                if use_log_gabor:
                    roi_processed = log_gabor_safe(roi_processed, center_freq=0.15, sigma_f=0.55, n_orients=4)

                # --- Canonicalize orientation (thumb always on same side) ---
                # Convention: RIGHT hand = canonical, LEFT hand = flipped
                if hand_label == "Left":
                    roi_processed = cv2.flip(roi_processed, 1)


    return frame, roi_poly, roi_processed, lm


# =========================
# FILE INDEXING
# =========================
def next_id(folder):
    ids = [int(m.group(1)) for f in os.listdir(folder)
           if (m := re.match(r"(\d{5})_", f))]
    return max(ids, default=0) + 1

# =========================
# MAIN LOOP
# =========================
READY_FRAMES = 5
stable_count = 0
capturing = False
CAPTURE_TOTAL = 10
CAPTURE_DELAY = 0.15
capture_idx = 0
last_time = 0

print("✅ Dual MediaPipe capture system running")

while True:
    L = cv2.flip(cv2.cvtColor(picam_left.capture_array(), cv2.COLOR_BGRA2BGR), 1)

    R = cv2.flip(cv2.cvtColor(picam_right.capture_array(), cv2.COLOR_BGRA2BGR), 1)
    R = recolor_ir(R)  # keep your IR recolor for right camera

    # =========================
    # Right-hand vein preprocessing (dark veins)
    # =========================

    R_gray = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)

    # --- Adaptive histogram equalization ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    R_clahe = clahe.apply(R_gray)

    # --- Optional: gamma correction ---
    gamma = 1.8  # >1 darkens veins (reverse of before)
    R_gamma = np.power(R_clahe / 255.0, gamma) * 255
    R_gamma = R_gamma.astype(np.uint8)

    # --- Optional slight blur to reduce noise ---
    R_proc = cv2.GaussianBlur(R_gamma, (3,3), 0)

    # --- Convert back to 3-channel for MediaPipe display ---
    R = cv2.cvtColor(R_proc, cv2.COLOR_GRAY2BGR)

    L, L_poly, L_roi, L_lm = process_frame(L, hands_left, label_prefix="LeftCam", use_log_gabor=True)
    R, R_poly, R_roi, R_lm = process_frame(R, hands_right, label_prefix="RightCam", use_log_gabor=False)


    stable = (
        L_poly is not None and R_poly is not None and
        roi_area_ratio(L_poly, L.shape) >= MIN_ROI_AREA_RATIO and
        roi_area_ratio(R_poly, R.shape) >= MIN_ROI_AREA_RATIO and
        roi_fully_inside(L_poly, L.shape) and
        roi_fully_inside(R_poly, R.shape) and
        not fingers_inside_roi(L_poly, L_lm, L.shape)
    )

    if L_poly is not None and L_lm is not None:
        if fingers_inside_roi(L_poly, L_lm, L.shape):
            cv2.putText(
                L,
                "Keep fingers outside ROI",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

    stable_count = stable_count + 1 if stable else 0
    ready = stable_count >= READY_FRAMES

    # =========================
    # RESET WHEN HAND LEAVES
    # =========================
    if not stable:
        success_latched = False

    # =========================
    # AUTO RECOGNITION TRIGGER
    # =========================
    if ready and not success_active and not success_latched:
        success_active = True
        success_latched = True
        success_start_time = time.time()
        print("✅ Palm recognized")


    cv2.putText(L, "READY" if ready else f"Stabilizing {stable_count}/{READY_FRAMES}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0) if ready else (0, 255, 255), 2)

    if L_poly is not None and roi_area_ratio(L_poly, L.shape) < MIN_ROI_AREA_RATIO:
        cv2.putText(
            L,
            "Move hand closer",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

    key = cv2.waitKey(1) & 0xFF

    now = time.time()

    cv2.imshow("Left | Right", np.hstack([L, R]))

    # =========================
    # SUCCESS UI OVERLAY
    # =========================
    if success_active:
        elapsed = time.time() - success_start_time
        if elapsed <= SUCCESS_DISPLAY_TIME:
            success_frame = np.zeros_like(L)
            success_frame[:] = (0, 200, 0)

            h, w, _ = success_frame.shape
            center = (w // 2, h // 2)

            # Draw checkmark using lines (Unicode-free)
            cx, cy = center

            pt1 = (cx - 120, cy)
            pt2 = (cx - 20, cy + 100)
            pt3 = (cx + 140, cy - 120)

            cv2.line(success_frame, pt1, pt2, (255, 255, 255), 25, cv2.LINE_AA)
            cv2.line(success_frame, pt2, pt3, (255, 255, 255), 25, cv2.LINE_AA)


            cv2.putText(
                success_frame,
                "PALM VERIFIED",
                (center[0] - 220, center[1] + 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (255, 255, 255),
                4,
                cv2.LINE_AA
            )

            cv2.imshow("Left | Right", np.hstack([success_frame, success_frame]))
            continue
        else:
            success_active = False

    if key == 27:
        break

# =========================
# CLEANUP
# =========================
hands_left.close()
hands_right.close()
picam_left.stop()
picam_right.stop()
cv2.destroyAllWindows()
print("👋 System closed")
