# label-tool-helper
Object detection API for label-tool (https://github.com/Slava/label-tool)

YOLOv5 based

# Usage 
```
git clone git@github.com:Thessal/label-tool-helper.git
cd label-tool-helper
git clone https://github.com/ultralytics/yolov5.git 
git checkout a199480ba6bb527598df11abbc1d679ccda82670 
pip install -r requirements.txt
# python export.py --weights yolov5s.pt --include onnx --dynamic
python server.py
```
