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

    # Updated black detection — more strict
    lower_black = (0, 0, 0)
    upper_black = (180, 80, 50)  # Very dark only, not medium gray!

    mask = cv2.inRange(hsv, lower_black, upper_black)
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fingers = []

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # Area must be BIG enough to trust it's a real object
        if area > 5000:  # (boosted threshold from 3000 to 5000)
            # Find extreme points
            ext_left = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
            ext_right = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
            ext_top = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
            ext_bottom = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])

            fingers.append({"type": "finger", "x": int(ext_left[0]), "y": int(ext_left[1])})
            fingers.append({"type": "finger", "x": int(ext_right[0]), "y": int(ext_right[1])})
            fingers.append({"type": "finger", "x": int(ext_top[0]), "y": int(ext_top[1])})
            fingers.append({"type": "finger", "x": int(ext_bottom[0]), "y": int(ext_bottom[1])})
        else:
            # Ignore tiny blobs (like corners, noise)
            fingers = []

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
