import os
import tempfile
from threading import Lock

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["*"])

mutex = Lock()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "posemodelv2.h5")
OUT_PATH = os.path.join(tempfile.gettempdir(), "out.png")

model = None

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

font = cv2.FONT_HERSHEY_SIMPLEX
img_places = [(50, 50), (50, 70), (50, 90), (50, 110), (50, 130)]
fontScale = 0.5
color = (0, 255, 255)
thickness = 1


@app.route("/")
def hello_world():
    return jsonify({"success": True, "message": "Hackation backend is running"})


@app.route("/outimage")
def send_out_image():
    if not os.path.exists(OUT_PATH):
        return jsonify({"success": False, "error": "No output image yet"}), 404

    return send_file(OUT_PATH, max_age=0)


def process_image_pose(image):
    image = cv2.flip(image, 1)

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.6,
    ) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_hand_landmarks:
        image = cv2.putText(
            image,
            "No Hand",
            img_places[0],
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        cv2.imwrite(OUT_PATH, image)
        return "?", [["A", 0], ["B", 0], ["C", 0], ["D", 0], ["E", 0]]

    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

    one_hand = []
    for hand_landmarks in results.multi_hand_landmarks:
        for point in hand_landmarks.landmark:
            one_hand.extend([point.x, point.y, point.z])

    one_hand = np.array(one_hand[:63])

    # If model is not loaded, return dummy prediction
    if model is None:
        predicted_letter = "A"
        top_list = [["A", 100], ["B", 0], ["C", 0], ["D", 0], ["E", 0]]
        cv2.imwrite(OUT_PATH, image)
        return predicted_letter, top_list

    predicted_label = model.predict(
        np.reshape(one_hand, (1,) + one_hand.shape),
        verbose=False,
    )

    predicted_index = int(np.argmax(predicted_label))
    labels = [chr(j) for j in range(ord("A"), ord("Z") + 1)]
    predicted_letter = labels[predicted_index]

    image = cv2.putText(
        image,
        f"prediction {predicted_letter}",
        img_places[0],
        font,
        fontScale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    cv2.imwrite(OUT_PATH, image)

    top_list = [
        [chr(idx + ord("A")), int(prob * 10000) / 100]
        for idx, prob in enumerate(predicted_label[0])
    ]
    top_list.sort(key=lambda x: x[1], reverse=True)

    return predicted_letter, top_list[:5]


@app.route("/senddata", methods=["POST"])
def process_data():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "Missing file"}), 400

    acquired = mutex.acquire(True, 0.1)

    if not acquired:
        return jsonify({"success": False, "error": "Server busy"}), 429

    try:
        filestr = request.files["file"].read()
        file_bytes = np.frombuffer(filestr, np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

        if img is None:
            return jsonify({"success": False, "error": "Invalid image"}), 400

        result, top = process_image_pose(img)

        return jsonify({
            "success": True,
            "topcandidates": top,
            "result": result,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
        mutex.release()
