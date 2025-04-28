# camera_module.py (Hybrid Commander Edition)

import cv2
import numpy as np
import flask
import threading
import json
from flask import Flask, jsonify

# === Setup Flask App ===
app = Flask(__name__)

# === Shared State ===
latest_fingers = []
lock = threading.Lock()

# === Camera Setup ===
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# === Hand Detection ===
def detect_fingers(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Tune this to your glove / lighting
    lower_black = (0, 0, 0)
    upper_black = (180, 255, 70)

    mask = cv2.inRange(hsv, lower_black, upper_black)
    blurred = cv2.GaussianBlur(mask, (7, 7), 0)

    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fingers = []

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area > 5000:  # Area threshold to ignore junk
            # Optional: convex hull to smooth hand shape
            hull = cv2.convexHull(largest_contour)

            # Find topmost point (smallest y)
            topmost = tuple(hull[hull[:, :, 1].argmin()][0])

            fingers.append({
                "type": "finger",
                "x": int(topmost[0]),
                "y": int(topmost[1])
            })

    return fingers


# === Camera Reading Thread ===
def camera_loop():
    global latest_fingers

    while True:
        ret, frame = camera.read()
        if not ret:
            continue

        frame = cv2.flip(frame, -1)  # Optional: flip vertically if needed

        fingers = detect_fingers(frame)

        with lock:
            latest_fingers = fingers


# === API Endpoint ===
@app.route('/touches', methods=['GET'])
def get_touches():
    with lock:
        return jsonify(latest_fingers)


# === Server Start ===
def start_server():
    app.run(host='0.0.0.0', port=5000, threaded=True)


# === Main Launcher ===
if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    start_server()