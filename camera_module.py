# camera_module.py

import cv2
import flask
import threading
import json
from flask import Flask, jsonify

# Setup Flask app
app = Flask(__name__)

# Shared state
latest_fingers = []
lock = threading.Lock()

# Camera settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Initialize camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Finger detection function (Detects Black Gloves)
def detect_fingers(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Lighter black: allow a bit higher brightness
    lower_black = (0, 0, 0)
    upper_black = (180, 255, 100)  # Value threshold relaxed to 100

    mask = cv2.inRange(hsv, lower_black, upper_black)
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fingers = []

    if contours:
        # Only use the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Optional: Filter by minimum area to avoid random noise
        if cv2.contourArea(largest_contour) > 1500:
            # Find extreme points
            ext_left = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
            ext_right = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
            ext_top = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
            ext_bottom = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])

            # Create "finger" points from extremes
            fingers.append({"type": "finger", "x": ext_left[0], "y": ext_left[1]})
            fingers.append({"type": "finger", "x": ext_right[0], "y": ext_right[1]})
            fingers.append({"type": "finger", "x": ext_top[0], "y": ext_top[1]})
            fingers.append({"type": "finger", "x": ext_bottom[0], "y": ext_bottom[1]})

    return fingers



# Camera reading loop
def camera_loop():
    global latest_fingers

    while True:
        ret, frame = camera.read()
        if not ret:
            continue

        fingers = detect_fingers(frame)

        with lock:
            latest_fingers = fingers

# API endpoint to get touches
@app.route('/touches', methods=['GET'])
def get_touches():
    with lock:
        return jsonify(latest_fingers)

# Start server
def start_server():
    app.run(host='0.0.0.0', port=5000, threaded=True)

# Main start
if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    start_server()
