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

    # Define a stricter HSV range for black
    lower_black = (0, 0, 0)
    upper_black = (180, 255, 50)  # Lowered Value upper limit for better black detection

    mask = cv2.inRange(hsv, lower_black, upper_black)
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fingers = []

    if contours:
        # Sort contours by area, biggest first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours[:2]:  # Only keep up to 2 largest blobs (optional)
            area = cv2.contourArea(cnt)
            if 1000 < area < 10000:  # Filter by size (area between 1k and 10k pixels)
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    fingers.append({"type": "finger", "x": cx, "y": cy})

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
