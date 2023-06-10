#!/usr/bin/python3
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "anime_object_detection"))

import base64
import json
import numpy as np
import cv2
import io
from flask import Flask, request

from anime_object_detection.face import _DEFAULT_FACE_MODEL, _gr_detect_faces, _open_face_detect_model
from anime_object_detection.head import _DEFAULT_HEAD_MODEL, _gr_detect_heads, _open_head_detect_model
from anime_object_detection.person import _DEFAULT_PERSON_MODEL, _gr_detect_person, _open_person_detect_model
from imgutils.data import ImageTyping, load_image, rgb_encode
from onnx_ import _open_onnx_model
from plot import detection_visualize
from yolo_ import _image_preprocess, _data_postprocess

app = Flask(__name__)

# Load your model
def load_model():
    model = {
            "face": _open_face_detect_model(_DEFAULT_FACE_MODEL),
            "head": _open_head_detect_model(_DEFAULT_HEAD_MODEL),
            "person": _open_person_detect_model(_DEFAULT_PERSON_MODEL),
            "params": {
                "infer_size": 640, #  gr.Slider(480, 960, value=640, step=32, label='Max Infer Size')
                "face": {
                    "conf_threshold": 0.7, 
                    "iou_threshold": 0.25, 
                    },
                "head": {
                    "conf_threshold": 0.7, 
                    "iou_threshold": 0.3, 
                    },
                "person": {
                    "conf_threshold": 0.5, 
                    "iou_threshold": 0.3, 
                    },
                }
            }
    return model

# Perform object detection on the input image using YOLO
def perform_object_detection(input_bytes, model):
    # Convert the input bytes to an image
    nparr = np.frombuffer(input_bytes, np.uint8)
    # im0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = io.BytesIO(nparr)
    image = load_image(image, mode='RGB')
    max_infer_size = model["params"]["infer_size"]

    # Image convert stuff
    new_image, old_size, new_size = _image_preprocess(image, max_infer_size)
    data = rgb_encode(new_image)[None, ...]

    # Detect 
    results = []
    for target in ["person", "head", "face"]:
        output, = model[target].run(['output0'], {'images': data})
        _LABELS = [target]
        _result = _data_postprocess(output[0], 
                model["params"][target]["conf_threshold"], 
                model["params"][target]["iou_threshold"], 
                old_size, new_size, _LABELS)
        xs, ys = old_size
        _result = [((y0/ys, x0/xs, y1/ys, x1/xs), tag, conf) for (x0, y0, x1, y1), tag, conf in _result]
        results.extend(_result)

    # json
    tag_map = {"face":"face", "head":"head", "person":"body"}
    result = []
    for (y0, x0, y1, x1), tag, conf in results:
        result.append({
            "det_boxes":[y0, x0, y1, x1],
            "det_class":tag,
            "det_score":conf,
            })

    return result

@app.route('/detect', methods=['POST'])
def detect_objects():
    # Load YOLO model if not loaded already
    if not hasattr(app, 'model'):
        app.model = load_model()

    # Read the request JSON
    data = request.get_json()

    predictions = []
    for d in data['instances'] : 
        # Get the input image bytes from the request JSON
        b64 = d['input_bytes']['b64']
        input_bytes = base64.b64decode(b64)

        # # Debug
        # render_string = f'<img src="data:image/jpg;base64,{b64}"/>'
        # print(render_string)
        # test_string = 'curl --header "Content-Type: application/json" -d "{\\"instances\\":[{\\"input_bytes\\": {\\"b64\\": \\"'+b64+'\\"}} ]}" http://localhost:5000/detect'
        # print(test_string)

        # Perform object detection on the image
        predictions.append(perform_object_detection(input_bytes, app.model))

    # Prepare the response JSON
    response = { 'predictions': predictions[0] if len(predictions) == 1 else predictions }
    print(response)

    # Return the response as JSON
    return json.dumps(response)

if __name__ == '__main__':
    app.run()
