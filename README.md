Here's your modified Markdown content:

```markdown
# IHA-YOLO

Official PyTorch implementation of **IHA-YOLO**: Inter-Head Attention for Real-time Cell Detection and Counting.

## Installation

To set up the environment and install the required packages, run the following commands:

```bash
conda create -n iha_yolo python=3.10
conda activate iha_yolo
pip install torch===2.3.0 torchvision torchaudio
pip install seaborn thop timm einops
cd ultralytics
pip install -e .
```

## Training

To train the IHA-YOLO model, use the following code snippet:

```python
from ultralytics import YOLO

# Load the model configuration and weights
model = YOLO("IHA-YOLO/ultralytics/cfg/models/iha-yolo/iha-yolom.yaml").load("yolov10m.pt")

# Start training
results = model.train(data="data.yaml", epochs=200)
```

## Testing

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

## Performance Comparison

### Detection Performance Comparison Table

[Download Performance Chart](https://github.com/user-attachments/files/16749227/chart.11.1.1.pdf)

| **Datasets**               | **Mouse Embryos**                     |                                  |                                    | **BCCD**                        |                                  |                                    | **Average**   | **Model Complexity ($\downarrow$)** | **Model Complexity ($\downarrow$)** |
|----------------------------|---------------------------------------|----------------------------------|------------------------------------|----------------------------------|----------------------------------|------------------------------------|--------------|--------------------------------------|-------------------------------------|
| **Models**                 | **Precision ($\uparrow$)**            | **Recall ($\uparrow$)**          | **mAP50 ($\uparrow$)**             | **Precision ($\uparrow$)**       | **Recall ($\uparrow$)**          | **mAP50 ($\uparrow$)**             | **mAP50 ($\uparrow$)** | **GFLOPs**                          | **Parameters (M)**                 |
| Faster-RCNN                | 89.57                                 | 90.25                            | 90.82                              | 80.45                            | 83.24                            | 85.37                              | 58.08        | 134.38                               | 41.75                               |
| RetinaNet                  | 92.75                                 | 93.24                            | 93.89                              | 81.35                            | 81.87                            | 86.65                              | 57.41        | 273.00                               | 57.00                               |
| EfficientDet               | 88.92                                 | 90.52                            | 89.19                              | 78.41                            | 77.98                            | 79.81                              | 52.12        | 135.00                               | 34.00                               |
| DeGPR                      | 98.10                                 | 97.80                            | 99.15                              | 81.10                            | 89.60                            | 89.07                              | 68.71        | 72.60                                | 30.06                               |
| YOLOv8                     | 98.80                                 | 99.10                            | **99.34**                          | 82.50                            | 88.30                            | 87.55                              | 68.87        | 79.10                                | 25.86                               |
| YOLOv10                    | 98.30                                 | 97.60                            | 99.18                              | 86.20                            | 86.60                            | **90.83**                          | **69.82**    | 64.00                                | 16.49                               |
| Mamba YOLO                 | 97.90                                 | 98.10                            | 98.96                              | 83.30                            | 86.80                            | 90.49                              | 66.60        | 156.00                               | 57.60                               |
| RT-DETER                   | 98.80                                 | 98.80                            | 99.02                              | 80.40                            | 88.10                            | 88.28                              | 68.01        | 108.00                               | 38.81                               |
| **IHA-YOLO**               | 97.90                                 | 97.20                            | **99.22**                          | 87.10                            | 86.20                            | **90.92**                          | **70.74**    | 68.40                                | 22.16                               |

*Detection performance comparison of IHA-YOLO vs. various models on Mouse Embryos and BCCD datasets. The models listed in the last two sections were evaluated by us for the first time on these datasets. The highest values are highlighted in red, and the second highest in blue.*

![Qualitative Results](https://github.com/user-attachments/assets/890890f2-4f69-452e-9e0d-8e117bd13902)

## Acknowledgements

This repository is a modified version of the YOLOv10 code provided by [Ultralytics](https://github.com/ultralytics/ultralytics).
```

### Changes Made:
- Converted the table to a Markdown format that is supported by GitHub.
- Added a header for the performance comparison section.
- Ensured the correct Markdown table formatting with proper column alignment.
