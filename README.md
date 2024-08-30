# IHA-YOLO

Official PyTorch implementation of **IHA-YOLO**: Inter-Head Attention for Real-time Cell Detection and Counting.

## Abstract

Multi-class cell detection and counting are vital tasks in numerous biomedical applications, particularly in cell biology. The development of the YOLO object detection model has significantly advanced the field of real-time object detection, providing accurate and efficient multi-class detection. However, existing architectures often face challenges in precisely localizing and classifying small, densely clustered cells within complex biological images. Real-time and end-to-end cell detection and counting approaches face significant challenges due to limited data availability, overlapping tiny objects, diverse cell types, class imbalances, and subtle yet critical variations in cell size and shape. In this paper, we introduce Inter-Head Attention (IHA)-YOLO, a novel model that proposes an inter-head attention module to enhance global representation learning, thereby improving the model's ability to understand and process contextual information across the entire input feature map. This method is particularly effective in detecting small cells and sub-cell structures within constrained medical image datasets. Through extensive experiments on five publicly available datasets, IHA-YOLO demonstrates superior performance compared to state-of-the-art cell detection and counting methods, offering 13\% faster inference and an average absolute improvement of 2.03%  across five datasets.  Moreover, our model achieves relative mAP50:95 improvements of 5.33%,  6.74%, and 2.75% over the baseline YOLOv10 on the BOrg, MoNuSAC, and CoNSeP datasets, respectively, while maintaining a similar speed.

## Overview
![image](https://github.com/user-attachments/assets/a24c9230-f09a-4768-b32a-bb7e45a35936)

![image](https://github.com/user-attachments/assets/cb6f9f8b-a45e-4114-8cd3-ae10f41c8fe6)

![image](https://github.com/user-attachments/assets/22338e27-f653-40a4-83fb-d9be010f987e)


## Performance Comparison

### Detection Performance Comparison Table

![image](https://github.com/user-attachments/assets/a1592101-5de5-4368-b2a1-700c4ae1456a)


### Quantitative Results

![image](https://github.com/user-attachments/assets/de5ef97d-b49a-43b9-b54c-905552b7fdbb)

![image](https://github.com/user-attachments/assets/c6fb534b-928a-4bbb-ad71-576db872957d)

### Qualitative Results

![Qualitative Results](https://github.com/user-attachments/assets/890890f2-4f69-452e-9e0d-8e117bd13902)

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

