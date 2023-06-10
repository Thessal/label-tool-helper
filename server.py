import base64
import json
import numpy as np
# import cv2
from flask import Flask, request
from yolov5.detect import *
from yolov5.utils.augmentations import letterbox

app = Flask(__name__)

# Load your YOLO model
def load_yolo_model():
    weights = "./yolov5/yolov5s.onnx"
    data = "./yolov5/data/coco128.yaml"
    dnn = False
    device = ''
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=False)

    return model

# Perform object detection on the input image using YOLO
def perform_object_detection(input_bytes, model):
    imgsz=(640, 640)  # inference size (height, width)
    # conf_thres=0.25  # confidence threshold
    # iou_thres=0.45  # NMS IOU threshold
    conf_thres=0.1  # confidence threshold
    iou_thres=0.0  # NMS IOU threshold
    max_det=1000  # maximum detections per image
    view_img=False  # show results
    save_txt=False  # save results to *.txt
    save_conf=False  # save confidences in --save-txt labels
    save_crop=False  # save cropped prediction boxes
    nosave=False  # do not save images/videos
    classes=None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False  # class-agnostic NMS
    augment=False  # augmented inference
    update=False  # update all models
    name='exp'  # save results to project/name
    exist_ok=False  # existing project/name ok, do not increment
    line_thickness=3  # bounding box thickness (pixels)
    hide_labels=False  # hide labels
    hide_conf=False  # hide confidences
    dnn=False  # use OpenCV DNN for ONNX inference

    # Convert the input bytes to an image
    nparr = np.frombuffer(input_bytes, np.uint8)
    im0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # print(im0.shape)
    image = letterbox(im0, 640, stride=32, auto=True)[0]  # padded resize
    image = image.transpose((2, 0, 1))[::-1]
    image = np.ascontiguousarray(image)

    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Run inference
    bs = 1 
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    # for path, im, im0s, vid_cap, s in dataset:
    for im in [image]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = model(im, augment=augment, visualize=False)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        result = [] 
        for i, det in enumerate(pred):  # per image
            if len(det):
                # # Rescale boxes from img_size to im0 size
                # det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                # Rescale boxes from img_size to 1 
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], (1,1,3))

                for i in range(det.shape[0]):
                    result.append({
                        # "det_boxes":det[i, [2,3,0,1]].tolist(),
                        # [xm ym xM yM] -> [ym xm yM xM]
                        "det_boxes":det[i, [1,0,3,2]].tolist(),
                        "det_class":names[int(det[i, 5])],
                        "det_score":float(det[i, 4]), 
                    })
                    # print(result)
        return result

@app.route('/detect', methods=['POST'])
def detect_objects():
    # Load YOLO model if not loaded already
    if not hasattr(app, 'model'):
        app.model = load_yolo_model()

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

