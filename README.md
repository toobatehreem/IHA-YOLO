# IHA-YOLO

Official PyTorch implementation of **IHA-YOLO**: Inter-Head Attention for Real-time Cell Detection accepted at ISBI 2025.

## Abstract

Multiclass cell detection is a crucial task in numerous biomedical applications, particularly in cell biology. The development of YOLO object detection models has advanced the field of real-time detection, but it is still struggling with challenges in medical imaging due to limited data availability, overlapping tiny objects, diverse cell types, and class imbalances. In this paper, we introduce Inter-Head Attention (IHA)-YOLO, a novel model that proposes an inter-head self-attention block to enhance global representation learning, thereby improving the contextual understanding across feature maps and performing effective detection of small cells and sub-cell structures in medical images. Through extensive experiments on five publicly available datasets, IHA-YOLO outperforms the state-of-the-art methods, achieving an average absolute mAP50 improvement of 2.03% and a 13% faster inference rate. In addition to cell detection, we adapt IHA-YOLO for cell counting to demonstrate its effectiveness. 

## Overview
![image](https://github.com/user-attachments/assets/c49940c3-2460-47cd-98b6-65814c3fda7a)


## Performance Comparison

### Average mAP vs FPS
![image](https://github.com/user-attachments/assets/7b50f9da-4642-4597-b2d2-e67455367014)


## Results
### Cell Detection
![image](https://github.com/user-attachments/assets/f7bace74-506b-412e-8ee2-30f1f5b119e7)


### Cell Counting
![image](https://github.com/user-attachments/assets/67b08d2b-49a6-45c7-b271-e26e26b63b5d)


### Qualitative Visualization

![QualitativeResults1 (1)](https://github.com/user-attachments/assets/0f03c9da-4406-4f17-9c39-e83362b79e5c)

## Getting Started

### Installation

To set up the environment and install the required packages, run the following commands:

```bash
conda create -n iha_yolo python=3.10
conda activate iha_yolo
pip install torch===2.3.0 torchvision torchaudio
pip install seaborn thop timm einops
cd ultralytics
pip install -e .
```

### Training

To train the IHA-YOLO model, use the following code snippet:

```python
from ultralytics import YOLO

# Load the model configuration and weights
model = YOLO("IHA-YOLO/ultralytics/cfg/models/iha-yolo/iha-yolom.yaml").load("yolov10m.pt")

# Start training
results = model.train(data="data.yaml", epochs=200)
```

### Testing

To evaluate the trained model, you can use the following code:

```python
from ultralytics import YOLO

# Load the model configuration and weights
model = YOLO("IHA-YOLO/ultralytics/cfg/models/iha-yolo/iha-yolom.yaml").load("yolov10m.pt")

# Validate the model
metrics = model.val()

# Print evaluation metrics
print(f"Mean Average Precision @ .5:.95 : {metrics.box.map}")
print(f"Mean Average Precision @ .50   : {metrics.box.map50}")
print(f"Mean Average Precision @ .70   : {metrics.box.map75}")
```


## Acknowledgements

This repository is a modified version of the YOLOv10 code provided by [Ultralytics](https://github.com/ultralytics/ultralytics).

