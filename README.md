# IHA-YOLO

Official Pytorch implementation of IHA-YOLO: Inter-Head Attention for Real-time Cell Detection and Counting.

### Installation
```bash
conda create -n iha_yolo python=3.10
conda activate iha_yolo
pip3 install torch===2.3.0 torchvision torchaudio
pip install seaborn thop timm einops
cd ultralytics
pip install -e .
```

### Training
```bash
from ultralytics import YOLO
model = YOLO("IHA-YOLO/ultralytics/cfg/models/iha-yolo/iha-yolom.yaml").load("yolov10m.pt")
results = model.train(data="data.yaml", epochs=200)
```

### Acknowledgement
This repository is a modified version of the YOLOv10 code provided by [Ultralytics](https://github.com/ultralytics/ultralytics).
