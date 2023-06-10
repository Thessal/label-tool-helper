# label-tool-helper
Object detection API for label-tool (https://github.com/Slava/label-tool)

* server.py : YOLOv5 based
* server_anima.py : deepghs tuned model (https://huggingface.co/spaces/deepghs/anime_object_detection)

# Usage 

* Basic
```
git clone git@github.com:Thessal/label-tool-helper.git
pushd label-tool-helper
git clone https://github.com/ultralytics/yolov5.git 
git checkout a199480ba6bb527598df11abbc1d679ccda82670 
pip install -r requirements.txt
python export.py --weights yolov5s.pt --include onnx --dynamic
popd
python server.py
```

* Animation
```
git clone https://huggingface.co/spaces/deepghs/anime_object_detection
pushd anime_object_detection
git checkout ae08d38a667a83c19d76090ba0b8bdc49039acee
pip install -r requirements.txt
popd
python server_anime.py
```

endpoint : localhost:5000/detect
